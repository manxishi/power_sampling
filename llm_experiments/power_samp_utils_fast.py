"""
Optimized power sampling with KV cache reuse for MCMC.
Same functionality as power_samp_utils.py but faster via:
- KV cache reuse on MCMC reject (avoids recomputing prefix)
- Optional verbose flag to reduce I/O in hot paths
"""

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

### DESCRIPTION ###
# power sampling to sample from p^{alpha}, where p is the base model
# takes in 1/alpha (temperature) as an argument (default 0.25), and mcmc_power_samp implements sampling from p^{alpha}
# FAST VERSION: Uses KV cache reuse when MCMC rejects to avoid recomputing prefix.


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


def _copy_and_crop_cache(past_key_values, crop_length):
    """Clone and slice KV cache tensors to crop_length.

    Uses explicit .clone() on each layer's key/value tensors instead of
    copy.deepcopy, which is prohibitively slow on GPU tensors.
    """
    if past_key_values is None:
        return None

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    try:
        if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
            new_cache = DynamicCache()
            for layer_idx in range(len(past_key_values.key_cache)):
                new_cache.update(
                    past_key_values.key_cache[layer_idx][:, :, :crop_length, :].clone(),
                    past_key_values.value_cache[layer_idx][:, :, :crop_length, :].clone(),
                    layer_idx,
                )
            return new_cache

        if isinstance(past_key_values, (tuple, list)):
            return tuple(
                (k[:, :, :crop_length, :].clone(), v[:, :, :crop_length, :].clone())
                for k, v in past_key_values
            )

        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
            new_cache = type(past_key_values)()
            for layer_idx in range(len(past_key_values.key_cache)):
                new_cache.update(
                    past_key_values.key_cache[layer_idx][:, :, :crop_length, :].clone(),
                    past_key_values.value_cache[layer_idx][:, :, :crop_length, :].clone(),
                    layer_idx,
                )
            return new_cache
    except Exception:
        pass

    return None


def _get_cache_seq_length(cache):
    """Get sequence length of cache. Works with Cache, DynamicCache, or legacy tuple format."""
    if hasattr(cache, 'get_seq_length'):
        return cache.get_seq_length()
    if hasattr(cache, 'self_attention_cache') and hasattr(cache.self_attention_cache, 'get_seq_length'):
        return cache.self_attention_cache.get_seq_length()
    if isinstance(cache, (tuple, list)) and len(cache) > 0:
        first = cache[0]
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            return first[0].shape[-2]
    return None


def _naive_temp_with_cache(p, context, temp, seq_len, past_key_values=None, verbose=False):
    """
    Low-temperature sampling. Returns (prop, log_probs_norm, log_probs_unnorm, past_key_values).
    When past_key_values is provided (truncated to len(context)), reuses cache instead of recomputing prefix.
    Uses manual generation loop to capture cache (generate() does not return it).
    """
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    model = p.model
    num_new_tokens = seq_len - c

    if num_new_tokens <= 0:
        return context.copy(), [], [], past_key_values

    prop = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []

    if past_key_values is None:
        # Full context forward to fill cache, then decode
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids, use_cache=True)
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
    else:
        # Reuse truncated cache: pass last context token to get logits for next
        cache = _copy_and_crop_cache(past_key_values, c)
        if cache is None:
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
        else:
            input_ids = torch.tensor([[context[-1]]], dtype=torch.long, device=device)
            try:
                cache_len = _get_cache_seq_length(cache) or c
                cache_position = torch.tensor([cache_len], dtype=torch.long, device=device)
                outputs = model(input_ids=input_ids, past_key_values=cache, cache_position=cache_position, use_cache=True)
            except TypeError:
                outputs = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

    for step in range(num_new_tokens):
        unscaled = logits
        scaled = logits / temp
        probs = F.softmax(scaled, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        lp_unnorm = (1 / temp * F.log_softmax(unscaled, dim=-1)[0, next_token]).item()
        lp_norm = F.log_softmax(scaled, dim=-1)[0, next_token].item()
        log_probs_unnorm.append(lp_unnorm)
        log_probs_norm.append(lp_norm)
        prop.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

        input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
        try:
            cache_len = _get_cache_seq_length(cache) or (c + len(prop) - 1)
            cache_position = torch.tensor([cache_len], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, past_key_values=cache, cache_position=cache_position, use_cache=True)
        except (TypeError, AttributeError):
            outputs = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    return prop, log_probs_norm, log_probs_unnorm, cache


# Original naive_temp for compatibility (uses generate, no cache return)
def naive_temp(p: AutoregressiveSampler, context, temp, seq_len):
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


# alpha = infty power sampling; temp is for proposal distribution
def max_swap(p: AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16, verbose=True):
    c = len(context)
    if verbose:
        print(f'Temp: {temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []
    curr_kv = None

    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    if verbose:
        print(max_new_tokens)
        print(jump_size)
    attempts = 0
    acceptances = 0

    block_iter = tqdm(range(block_num), desc="blocks") if verbose else range(block_num)
    for _ in block_iter:
        gen, lp_norm, lp_unnorm, curr_kv = _naive_temp_with_cache(p, gen, temp=temp, seq_len=jump_size+len(gen), past_key_values=curr_kv, verbose=verbose)
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        mcmc_iter = tqdm(range(mcmc_steps), desc="mcmc") if verbose else range(mcmc_steps)
        for _ in mcmc_iter:
            attempts += 1
            t = len(gen)
            idx = random.randint(c, t-1)
            # Reuse truncated cache when we have one (on reject we keep curr_kv)
            prefix_kv = _copy_and_crop_cache(curr_kv, idx) if curr_kv is not None else None
            prop, log_prob_prop, target_log_prob_prop, prop_kv = _naive_temp_with_cache(p, gen[:idx], temp=temp, seq_len=t, past_key_values=prefix_kv, verbose=verbose)
            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx
            log_prob_cur = log_probs_norm[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm[idx-c:s-c]
            log_r = sum(target_log_prob_prop) - sum(target_log_prob_cur)

            if log_r > 0:
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()
                curr_kv = prop_kv

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


# power sampling with autoregressive mcmc (FAST: KV cache reuse)
def mcmc_power_samp(p: AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16, verbose=True):
    c = len(context)
    if verbose:
        print(f'alpha: {1/temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []
    curr_kv = None

    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    if verbose:
        print(max_new_tokens)
        print(jump_size)
    attempts = 0
    acceptances = 0

    block_iter = tqdm(range(block_num), desc="blocks") if verbose else range(block_num)
    for _ in block_iter:
        gen, lp_norm, lp_unnorm, curr_kv = _naive_temp_with_cache(p, gen, temp=temp, seq_len=jump_size+len(gen), past_key_values=curr_kv, verbose=verbose)
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        mcmc_iter = tqdm(range(mcmc_steps), desc="mcmc") if verbose else range(mcmc_steps)
        for _ in mcmc_iter:
            attempts += 1
            t = len(gen)
            idx = random.randint(c, t-1)
            prefix_kv = _copy_and_crop_cache(curr_kv, idx) if curr_kv is not None else None
            prop, log_prob_prop, target_log_prob_prop, prop_kv = _naive_temp_with_cache(p, gen[:idx], temp=temp, seq_len=t, past_key_values=prefix_kv, verbose=verbose)
            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx
            log_prob_cur = log_probs_norm[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm[idx-c:s-c]
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()
                curr_kv = prop_kv

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def format_prompt(question, model, tokenizer, cot=True):
    if model == "qwen":
        format_str = PROMPT + question
        if cot:
            format_str += COT
        else:
            format_str += BASE

    elif model == "qwen_math":
        format_str = PROMPT + question
        if cot:
            format_str += COT
        else:
            format_str += BASE

    elif model == "qwen_math_grpo":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi_grpo":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "tulu":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    return format_str
