import os
import time

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np


import torch
import transformers
from vllm import LLM, SamplingParams

from grader_utils.parse_utils import parse_answer
from constants import *
from power_samp_utils import *





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "results/",  dest = "save_str")
    parser.add_argument("--model", action = "store", default = "qwen", type = str, choices = ["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument("--temperature", action = "store", default = 0.25, type = float, dest = "temperature")
    parser.add_argument("--dataset", action = "store", default = "MATH", type = str)
    parser.add_argument("--cot", action = "store", type = bool, default = True)
    parser.add_argument("--mcmc_steps", action = "store", type = int, default = 10)
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)


    model = args.model
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)


    print(model)
    print(mcmc_steps)
    
    # Use local models from shared directory
    if model == "qwen":
        model_str = "/home/gridsan/mshi/MAML-Soljacic_shared/DeepSeek_models/DeepSeek-R1-Distill-Qwen-7B"
    elif model == "qwen_math":
        model_str = "/home/gridsan/mshi/MAML-Soljacic_shared/DeepSeek_models/DeepSeek-R1-Distill-Qwen-7B"
    elif model == "qwen_math_grpo":
        model_str = "/home/gridsan/mshi/MAML-Soljacic_shared/DeepSeek_models/DeepSeek-R1-Distill-Qwen-7B"
    elif model == "phi":
        model_str = "/home/gridsan/mshi/MAML-Soljacic_shared/DeepSeek_models/DeepSeek-R1-Distill-Qwen-7B"
    elif model == "tulu":
        model_str = "/home/gridsan/mshi/MAML-Soljacic_shared/DeepSeek_models/DeepSeek-R1-Distill-Qwen-7B"

    if dataset_name == "MATH":
        json_file = 'data/MATH500.json'
        dataset = json.load(open(json_file, "r"))



    print("dataset done")
    print(f"Loading model from: {model_str}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True, local_files_only=True)
    llm = LLM(model=model_str, trust_remote_code=True, dtype="half", enforce_eager=True,
              enable_chunked_prefill=False, max_model_len=32768)

    print("loaded models")
    results = []
    
    # Timing statistics
    timing_stats = {
        'naive_times': [],
        'std_times': [],
        'mcmc_times': []
    }

    start = 100*args.batch_idx
    end = 100*(args.batch_idx+1)

    for problem, data in tqdm(enumerate(dataset[start:end]), desc = "Benchmark on MATH"):
        question = data["prompt"]
        print(question)
        answer = data["answer"]

        input_text = format_prompt(question, model, tokenizer, cot)
        prefx = tokenizer.encode(input_text)

        # Time naive temperature sampling
        start_time = time.time()
        naive_out = llm.generate(
            prompt_token_ids=[prefx],
            sampling_params=SamplingParams(temperature=temp, max_tokens=3072, seed=args.seed),
        )[0].outputs[0]
        naive_time = time.time() - start_time
        timing_stats['naive_times'].append(naive_time)
        naive_token_ids = list(naive_out.token_ids)
        print(tokenizer.decode(naive_token_ids, skip_special_tokens=True))
        print(f"naive done (took {naive_time:.2f}s)")

        # Time standard sampling (temperature=1.0)
        start_time = time.time()
        std_out = llm.generate(
            prompt_token_ids=[prefx],
            sampling_params=SamplingParams(temperature=1.0, max_tokens=3072, seed=args.seed),
        )[0].outputs[0]
        std_time = time.time() - start_time
        timing_stats['std_times'].append(std_time)
        std_token_ids = list(std_out.token_ids)
        print(tokenizer.decode(std_token_ids, skip_special_tokens=True))
        print(f"std done (took {std_time:.2f}s)")

        # Time MCMC power sampling
        start_time = time.time()
        mcmc_power_samp_output, _, _, acceptance_ratio = mcmc_power_samp_vllm(llm, tokenizer, prefx, temp, mcmc_steps, max_new_tokens=3072, seed=args.seed)
        mcmc_time = time.time() - start_time
        timing_stats['mcmc_times'].append(mcmc_time)

        mcmc_token_ids = mcmc_power_samp_output[len(prefx):]
        print(len(std_token_ids))
        print(len(naive_token_ids))
        print(len(mcmc_token_ids))
        print(tokenizer.decode(mcmc_token_ids, skip_special_tokens=True))
        print(f"mcmc done (took {mcmc_time:.2f}s)")
        print(f"MCMC slowdown: {mcmc_time/naive_time:.2f}x vs naive, {mcmc_time/std_time:.2f}x vs std")

        naive_completion = tokenizer.decode(naive_token_ids, skip_special_tokens=True)
        std_completion = tokenizer.decode(std_token_ids, skip_special_tokens=True)
        mcmc_completion = tokenizer.decode(mcmc_token_ids, skip_special_tokens=True)

        naive_answer = parse_answer(naive_completion)
        std_answer = parse_answer(std_completion)
        mcmc_answer = parse_answer(mcmc_completion)
        
        print(naive_answer)
        print(std_answer)
        print(mcmc_answer)
        print(question)
        print(answer)
        print(f'Acceptance: {acceptance_ratio}')


        results.append({
            "question": question,
            "correct_answer": answer,
            "naive_completion": naive_completion,
            "naive_answer": naive_answer,
            "naive_time_sec": naive_time,
            "std_completion": std_completion,
            "std_answer": std_answer,
            "std_time_sec": std_time,
            "mcmc_completion": mcmc_completion,
            "mcmc_answer": mcmc_answer,
            "mcmc_time_sec": mcmc_time,
            "acceptance_ratio": acceptance_ratio,
        })

    # Print timing summary
    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    
    naive_times = np.array(timing_stats['naive_times'])
    std_times = np.array(timing_stats['std_times'])
    mcmc_times = np.array(timing_stats['mcmc_times'])
    
    print(f"\nNaive Temperature Sampling (T={temp}):")
    print(f"  Mean: {naive_times.mean():.2f}s | Median: {np.median(naive_times):.2f}s")
    print(f"  Min: {naive_times.min():.2f}s | Max: {naive_times.max():.2f}s")
    print(f"  Total: {naive_times.sum():.2f}s")
    
    print(f"\nStandard Sampling:")
    print(f"  Mean: {std_times.mean():.2f}s | Median: {np.median(std_times):.2f}s")
    print(f"  Min: {std_times.min():.2f}s | Max: {std_times.max():.2f}s")
    print(f"  Total: {std_times.sum():.2f}s")
    
    print(f"\nMCMC Power Sampling ({mcmc_steps} steps):")
    print(f"  Mean: {mcmc_times.mean():.2f}s | Median: {np.median(mcmc_times):.2f}s")
    print(f"  Min: {mcmc_times.min():.2f}s | Max: {mcmc_times.max():.2f}s")
    print(f"  Total: {mcmc_times.sum():.2f}s")
    
    print(f"\nSlowdown (MCMC vs baselines):")
    print(f"  vs Naive: {(mcmc_times / naive_times).mean():.2f}x (mean)")
    print(f"  vs Std: {(mcmc_times / std_times).mean():.2f}x (mean)")
    
    print("="*60 + "\n")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_str, model+"_math_base_power_samp_results_" + str(mcmc_steps) + "_" + str(temp) + "_" + str(args.batch_idx)  + "_" + str(args.seed) + ".csv"), index=False)
    
    # Also save timing summary as JSON
    timing_summary = {
        'mcmc_steps': mcmc_steps,
        'temperature': temp,
        'batch_idx': args.batch_idx,
        'seed': args.seed,
        'num_problems': len(naive_times),
        'naive': {
            'mean': float(naive_times.mean()),
            'median': float(np.median(naive_times)),
            'min': float(naive_times.min()),
            'max': float(naive_times.max()),
            'total': float(naive_times.sum())
        },
        'std': {
            'mean': float(std_times.mean()),
            'median': float(np.median(std_times)),
            'min': float(std_times.min()),
            'max': float(std_times.max()),
            'total': float(std_times.sum())
        },
        'mcmc': {
            'mean': float(mcmc_times.mean()),
            'median': float(np.median(mcmc_times)),
            'min': float(mcmc_times.min()),
            'max': float(mcmc_times.max()),
            'total': float(mcmc_times.sum())
        },
        'slowdown_vs_naive': float((mcmc_times / naive_times).mean()),
        'slowdown_vs_std': float((mcmc_times / std_times).mean())
    }
    
    timing_file = os.path.join(save_str, model+"_timing_summary_" + str(mcmc_steps) + "_" + str(temp) + "_" + str(args.batch_idx)  + "_" + str(args.seed) + ".json")
    with open(timing_file, 'w') as f:
        json.dump(timing_summary, f, indent=2)
    print(f"Timing summary saved to: {timing_file}")
    












        













