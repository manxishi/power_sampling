import os

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets


import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers

from grader_utils.parse_utils import parse_answer
from constants import *

from vllm import LLM, SamplingParams
from vllm.sampling_params import LogitsProcessor

### DESCRIPTION ###
# power sampling to sample from p^{alpha}, where p is the base model
# takes in 1/alpha (temperature) as an argument (default 0.25), and mcmc_power_samp implements sampling from p^{alpha} 


class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    # returns log probs
    @torch.no_grad()
    def next_token(self, prefix):
        device = self.device
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=device)
        prefix_cond = torch_prefix if torch_prefix.size(1) <= self.block_size else torch_prefix[:, -self.block_size:]
        output = self.model(prefix_cond)
        logits = output.logits
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)



# returns probabilities (normed)
def normalize(dist):
    probs = F.softmax(dist, dim=-1)
    return probs

# returns sum of logits (product of distributions p*q)
def dist_product(logit_p, logit_q):
    return logit_p+logit_q

# returns logit scaled by temp (temperature scaling p^(1/tau))
def dist_temp_scale(logit_p, temp):
    return logit_p * torch.tensor(1 / temp, dtype=logit_p.dtype, device=logit_p.device)

# low-temperature sampling proposal distribution
def naive_temp(p : AutoregressiveSampler, context, temp, seq_len):
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]


    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    return prop, log_probs_norm, log_probs_unnorm


def naive_temp_vllm(llm, context_ids, temp, seq_len, seed=None):
    c = len(context_ids)
    max_new = seq_len - c

    # Capture unscaled log probs via logits processor (called BEFORE temperature scaling)
    captured_lp_base = []

    def capture_logits(token_ids, logits):
        # Called with raw logits before any temperature scaling
        lp = F.log_softmax(logits, dim=-1)
        captured_lp_base.append(lp.cpu())
        return logits  # return unmodified so vLLM applies temp scaling normally

    sampling_params = SamplingParams(
        temperature=temp,
        max_tokens=max_new,
        logprobs=1,
        logits_processors=[capture_logits],
        seed=seed,
    )

    outputs = llm.generate(
        prompt_token_ids=[context_ids],
        sampling_params=sampling_params,
    )

    out = outputs[0].outputs[0]
    generated_tokens = list(out.token_ids)
    prop = context_ids + generated_tokens

    log_probs_norm = [
        out.logprobs[i][generated_tokens[i]].logprob
        for i in range(len(generated_tokens))
    ]

    log_probs_unnorm = [
        (1.0 / temp) * captured_lp_base[i][generated_tokens[i]].item()
        for i in range(len(generated_tokens))
    ]

    return prop, log_probs_norm, log_probs_unnorm
    
# def naive_temp_vllm(llm, context_ids, temp, seq_len):
#     c = len(context_ids)
#     max_new = seq_len - c

#     # Pass 1: generate tokens + get lp_norm (log prob under tempered dist)
#     gen_params = SamplingParams(
#         temperature=temp,
#         max_tokens=max_new,
#         logprobs=1,
#         prompt_logprobs=0,
#     )
#     outputs = llm.generate(
#         prompt_token_ids=[context_ids],
#         sampling_params=gen_params,
#     )
#     out = outputs[0].outputs[0]
#     generated_tokens = list(out.token_ids)
#     prop = context_ids + generated_tokens

#     log_probs_norm = [
#         out.logprobs[i][generated_tokens[i]].logprob
#         for i in range(len(generated_tokens))
#     ]

#     # Pass 2: score at temp=1 to get lp_base, then lp_unnorm = (1/temp)*lp_base
#     # We score the full sequence; vLLM treats it as prompt_logprobs
#     score_params = SamplingParams(
#         temperature=1.0,
#         max_tokens=1,          # minimal generation, we only want prompt_logprobs
#         prompt_logprobs=1,
#     )
#     score_outputs = llm.generate(
#         prompt_token_ids=[prop],
#         sampling_params=score_params,
#     )
#     # prompt_logprobs[i] is the logprob of prop[i] given prop[:i]
#     # indices c..c+len(generated_tokens)-1 correspond to generated tokens
#     prompt_lps = score_outputs[0].prompt_logprobs  # list of dicts, index 0 is None
#     log_probs_unnorm = []
#     for i, token_id in enumerate(generated_tokens):
#         token_pos = c + i  # position in full sequence
#         lp_base = prompt_lps[token_pos][token_id].logprob
#         log_probs_unnorm.append((1.0 / temp) * lp_base)

#     return prop, log_probs_norm, log_probs_unnorm

# def naive_temp_vllm(llm, context_ids, temp, seq_len):
#     """
#     Drop-in replacement for naive_temp using vLLM.
#     Returns (prop, log_probs_norm, log_probs_unnorm)
#     """
#     c = len(context_ids)
#     max_new = seq_len - c

#     # vLLM needs prompt as token ids
#     sampling_params = SamplingParams(
#         temperature=temp,
#         max_tokens=max_new,
#         logprobs=1,          # returns log probs under the sampling distribution (norm)
#         prompt_logprobs=0,   # don't need prompt log probs
#     )

#     outputs = llm.generate(
#         prompt_token_ids=[context_ids],
#         sampling_params=sampling_params,
#     )

#     out = outputs[0].outputs[0]
#     generated_tokens = list(out.token_ids)
#     prop = context_ids + generated_tokens

#     # log_probs_norm: log prob under temperature-scaled distribution
#     log_probs_norm = []
#     log_probs_unnorm = []
#     for token_id, logprob_dict in zip(generated_tokens, out.logprobs):
#         lp_norm = logprob_dict[token_id].logprob  # log p(x|scaled)
#         # unnorm = (1/temp) * log p(x|unscaled) = lp_norm / temp * temp... 
#         # vLLM only gives scaled logprobs, so recover unscaled:
#         # log p_unscaled(x) = temp * lp_norm  (since lp_norm = (1/temp)*log p_unscaled)
#         lp_unnorm = temp * lp_norm
#         log_probs_norm.append(lp_norm)
#         log_probs_unnorm.append(lp_unnorm)

#     return prop, log_probs_norm, log_probs_unnorm

# alpha = infty power sampling; temp is for proposal distribution
def max_swap(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    c = len(context)
    print(f'Temp: {temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0


    for _ in tqdm(range(block_num)):
        # gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        gen, lp_norm, lp_unnorm = naive_temp_vllm(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            # prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            prop, log_prob_prop, target_log_prob_prop = naive_temp_vllm(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) - sum(target_log_prob_cur)

            if log_r > 0:
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

# power sampling with autoregressive mcmc
def mcmc_power_samp(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    c = len(context)
    print(f'alpha: {1/temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0


    for _ in tqdm(range(block_num)):
        # gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        gen, lp_norm, lp_unnorm = naive_temp_vllm(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            # prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            prop, log_prob_prop, target_log_prob_prop = naive_temp_vllm(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

def mcmc_power_samp_vllm(llm, tokenizer, context, temp, mcmc_steps, max_new_tokens, block_num=16, seed=None):
    c = len(context)
    gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []
    eos_id = tokenizer.eos_token_id
    attempts = 0
    acceptances = 0

    assert max_new_tokens % block_num == 0
    jump_size = max_new_tokens // block_num

    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp_vllm(llm, gen, temp=temp, seq_len=jump_size + len(gen), seed=seed)
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts += 1
            t = len(gen)
            idx = random.randint(c, t - 1)
            prop, log_prob_prop, target_log_prob_prop = naive_temp_vllm(llm, gen[:idx], temp=temp, seq_len=t, seed=seed)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx - c : s - c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx - c : s - c]
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c:] = log_prob_prop.copy()
                log_probs_unnorm[idx - c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if eos_id in gen:
            eos_idx = gen.index(eos_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            return gen, log_probs_norm, log_probs_unnorm, acceptances / attempts

    return gen, log_probs_norm, log_probs_unnorm, acceptances / attempts


def format_prompt(question, model, tokenizer, cot=True):
    if model == "qwen":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_math":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_math_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "tulu":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    return format_str
