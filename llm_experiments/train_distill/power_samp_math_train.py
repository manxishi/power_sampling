import os
import sys
import time
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
import torch
import transformers
from vllm import LLM, SamplingParams

# Shared utilities live one level up
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from grader_utils.parse_utils import parse_answer
from constants import *
from power_samp_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", type=str, default="results/")
    parser.add_argument("--model", type=str, default="qwen_math",
                        choices=["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--mcmc_steps", type=int, default=10)
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model = args.model
    temp = args.temperature
    mcmc_steps = args.mcmc_steps

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)

    model_str = "/home/gridsan/mshi/MAML-Soljacic_shared/DeepSeek_models/DeepSeek-R1-Distill-Qwen-7B"

    # Load MATH training set (7500 problems across all subjects)
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'MATH_train.json')
    dataset = json.load(open(data_path))
    print(f"Loaded MATH train: {len(dataset)} problems")

    print(f"Loading model from: {model_str}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_str, trust_remote_code=True, local_files_only=True)
    llm = LLM(model=model_str, trust_remote_code=True, dtype="half", enforce_eager=True,
              enable_chunked_prefill=False, max_model_len=32768)
    print("Model loaded.")

    start = args.batch_size * args.batch_idx
    end = args.batch_size * (args.batch_idx + 1)
    batch = dataset[start:end]
    print(f"Running batch {args.batch_idx}: problems {start}–{end-1} ({len(batch)} problems)")

    results = []
    timing_stats = {'naive_times': [], 'std_times': [], 'mcmc_times': []}

    for data in tqdm(batch, desc=f"Batch {args.batch_idx}"):
        question = data["prompt"]
        answer = data["answer"]
        level = data.get("level", "")
        prob_type = data.get("type", "")

        input_text = format_prompt(question, model, tokenizer, cot=True)
        prefx = tokenizer.encode(input_text)

        # Naive temperature sampling
        t0 = time.time()
        naive_out = llm.generate(
            prompt_token_ids=[prefx],
            sampling_params=SamplingParams(temperature=temp, max_tokens=3072, seed=args.seed),
        )[0].outputs[0]
        naive_time = time.time() - t0
        timing_stats['naive_times'].append(naive_time)
        naive_token_ids = list(naive_out.token_ids)
        naive_completion = tokenizer.decode(naive_token_ids, skip_special_tokens=True)
        print(f"naive done ({naive_time:.2f}s)")

        # Standard sampling (temp=1.0)
        t0 = time.time()
        std_out = llm.generate(
            prompt_token_ids=[prefx],
            sampling_params=SamplingParams(temperature=1.0, max_tokens=3072, seed=args.seed),
        )[0].outputs[0]
        std_time = time.time() - t0
        timing_stats['std_times'].append(std_time)
        std_token_ids = list(std_out.token_ids)
        std_completion = tokenizer.decode(std_token_ids, skip_special_tokens=True)
        print(f"std done ({std_time:.2f}s)")

        # MCMC power sampling
        t0 = time.time()
        mcmc_out, _, _, acceptance_ratio = mcmc_power_samp_vllm(
            llm, tokenizer, prefx, temp, mcmc_steps, max_new_tokens=3072, seed=args.seed)
        mcmc_time = time.time() - t0
        timing_stats['mcmc_times'].append(mcmc_time)
        mcmc_token_ids = mcmc_out[len(prefx):]
        mcmc_completion = tokenizer.decode(mcmc_token_ids, skip_special_tokens=True)
        print(f"mcmc done ({mcmc_time:.2f}s), acceptance={acceptance_ratio:.3f}")
        print(f"MCMC slowdown: {mcmc_time/naive_time:.2f}x naive, {mcmc_time/std_time:.2f}x std")

        naive_answer = parse_answer(naive_completion)
        std_answer = parse_answer(std_completion)
        mcmc_answer = parse_answer(mcmc_completion)

        print(f"answers: naive={naive_answer} | std={std_answer} | mcmc={mcmc_answer} | correct={answer}")

        results.append({
            "id": data.get("id", ""),
            "question": question,
            "prompt": input_text,
            "correct_answer": answer,
            "level": level,
            "type": prob_type,
            "naive_completion": naive_completion,
            "naive_answer": naive_answer,
            "naive_correct": naive_answer == answer,
            "naive_time_sec": naive_time,
            "std_completion": std_completion,
            "std_answer": std_answer,
            "std_correct": std_answer == answer,
            "std_time_sec": std_time,
            "mcmc_completion": mcmc_completion,
            "mcmc_answer": mcmc_answer,
            "mcmc_correct": mcmc_answer == answer,
            "mcmc_time_sec": mcmc_time,
            "acceptance_ratio": acceptance_ratio,
        })

    # Timing summary
    naive_times = np.array(timing_stats['naive_times'])
    std_times = np.array(timing_stats['std_times'])
    mcmc_times = np.array(timing_stats['mcmc_times'])

    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    print(f"Naive (T={temp}): mean={naive_times.mean():.2f}s, total={naive_times.sum():.2f}s")
    print(f"Std (T=1.0):     mean={std_times.mean():.2f}s, total={std_times.sum():.2f}s")
    print(f"MCMC ({mcmc_steps} steps): mean={mcmc_times.mean():.2f}s, total={mcmc_times.sum():.2f}s")
    print(f"Slowdown: {(mcmc_times/naive_times).mean():.2f}x naive, {(mcmc_times/std_times).mean():.2f}x std")

    df = pd.DataFrame(results)
    n = len(df)
    print(f"\nAccuracy on batch ({n} problems):")
    print(f"  Naive: {df['naive_correct'].sum()}/{n} = {df['naive_correct'].mean():.1%}")
    print(f"  Std:   {df['std_correct'].sum()}/{n} = {df['std_correct'].mean():.1%}")
    print(f"  MCMC:  {df['mcmc_correct'].sum()}/{n} = {df['mcmc_correct'].mean():.1%}")
    print("="*60)

    fname = f"{model}_train_power_samp_{mcmc_steps}steps_{temp}temp_batch{args.batch_idx}_seed{args.seed}.csv"
    out_path = os.path.join(save_str, fname)
    df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")
