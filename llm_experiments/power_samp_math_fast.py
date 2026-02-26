"""
Optimized power sampling for MATH benchmark.
Uses power_samp_utils_fast (KV cache reuse) + model-level optimizations:
- Flash Attention 2 (optional, if available - no pip install required)
- BF16/FP16 dtype
- Optional torch.compile (--compile-model flag)
- Reduced verbose output for faster I/O

Run like power_samp_math.py. Saves to *_fast* filenames to avoid overwriting.
"""

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
from power_samp_utils_fast import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action="store", type=str, default="results/", dest="save_str")
    parser.add_argument("--model", action="store", default="qwen", type=str,
                        choices=["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument("--temperature", "--temp", action="store", default=0.25, type=float, dest="temperature")
    parser.add_argument("--dataset", action="store", default="MATH", type=str)
    parser.add_argument("--cot", action="store", type=bool, default=True)
    parser.add_argument("--mcmc_steps", action="store", type=int, default=10)
    parser.add_argument("--device", action="store", type=str, dest="device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_idx", action="store", type=int, default=0)
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument("--compile-model", action="store_true", help="Use torch.compile (experimental)")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable Flash Attention 2")
    parser.add_argument("--verbose", action="store_true", help="Extra print/tqdm in MCMC")
    args = parser.parse_args()

    random.seed(0)

    model = args.model
    device = args.device
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)

    print(model)
    print(device)
    print(mcmc_steps)

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
        json_file = "data/MATH500.json"
        dataset = json.load(open(json_file, "r"))

    print("dataset done")
    print(f"Loading model from: {model_str}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True, local_files_only=True)

    use_flash = not args.no_flash_attn
    load_kw = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    if use_flash:
        try:
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                model_str, attn_implementation="flash_attention_2", **load_kw
            )
            print("Using Flash Attention 2")
        except Exception as e:
            print(f"Flash Attention not available, falling back to default attn: {e}")
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, **load_kw)
    else:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, **load_kw)
    hf_model = hf_model.to(device)
    if args.compile_model:
        try:
            hf_model = torch.compile(hf_model, mode="reduce-overhead")
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    print("loaded models")
    results = []
    timing_stats = {"naive_times": [], "std_times": [], "mcmc_times": []}

    start = 100 * args.batch_idx
    end = 100 * (args.batch_idx + 1)
    verbose_mcmc = args.verbose

    for problem, data in tqdm(enumerate(dataset[start:end]), desc="Benchmark on MATH"):
        question = data["prompt"]
        answer = data["answer"]

        input_text = format_prompt(question, model, tokenizer, cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefx = [idx.item() for idx in input_ids[0]]

        start_time = time.time()
        naive_temp_output = hf_model.generate(
            input_ids, max_new_tokens=3072,
            return_dict_in_generate=True, output_scores=True, do_sample=True, temperature=temp
        )
        naive_time = time.time() - start_time
        timing_stats["naive_times"].append(naive_time)
        if verbose_mcmc:
            print(tokenizer.decode(naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
            print(f"naive done (took {naive_time:.2f}s)")

        start_time = time.time()
        std_output = hf_model.generate(
            input_ids, max_new_tokens=3072,
            return_dict_in_generate=True, output_scores=True, do_sample=True
        )
        std_time = time.time() - start_time
        timing_stats["std_times"].append(std_time)
        if verbose_mcmc:
            print(tokenizer.decode(std_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
            print(f"std done (took {std_time:.2f}s)")

        start_time = time.time()
        mcmc_power_samp_output, _, _, acceptance_ratio = mcmc_power_samp(
            autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=3072, verbose=verbose_mcmc
        )
        mcmc_time = time.time() - start_time
        timing_stats["mcmc_times"].append(mcmc_time)

        naive_generated_ids = naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        std_generated_ids = std_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        mcmc_power_samp_ids = torch.tensor([mcmc_power_samp_output], dtype=torch.long, device=device).squeeze().to("cpu")

        naive_completion = tokenizer.decode(naive_generated_ids, skip_special_tokens=True)
        std_completion = tokenizer.decode(std_generated_ids, skip_special_tokens=True)
        mcmc_completion = tokenizer.decode(mcmc_power_samp_ids, skip_special_tokens=True)

        naive_answer = parse_answer(naive_completion)
        std_answer = parse_answer(std_completion)
        mcmc_answer = parse_answer(mcmc_completion)

        if verbose_mcmc:
            print(naive_answer, std_answer, mcmc_answer, question, answer, f"Acceptance: {acceptance_ratio}")

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

    naive_times = np.array(timing_stats["naive_times"])
    std_times = np.array(timing_stats["std_times"])
    mcmc_times = np.array(timing_stats["mcmc_times"])

    print("\n" + "=" * 60)
    print("TIMING SUMMARY (FAST)")
    print("=" * 60)
    print(f"\nNaive: mean={naive_times.mean():.2f}s total={naive_times.sum():.2f}s")
    print(f"Std:   mean={std_times.mean():.2f}s total={std_times.sum():.2f}s")
    print(f"MCMC:  mean={mcmc_times.mean():.2f}s total={mcmc_times.sum():.2f}s")
    print(f"Slowdown vs naive: {(mcmc_times / naive_times).mean():.2f}x")
    print(f"Slowdown vs std:   {(mcmc_times / std_times).mean():.2f}x")
    print("=" * 60 + "\n")

    base_name = f"{model}_math_base_power_samp_results_{mcmc_steps}_{temp}_{args.batch_idx}_{args.seed}_fast"
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_str, base_name + ".csv"), index=False)

    timing_summary = {
        "mcmc_steps": mcmc_steps, "temperature": temp, "batch_idx": args.batch_idx, "seed": args.seed,
        "num_problems": len(naive_times),
        "naive": {"mean": float(naive_times.mean()), "total": float(naive_times.sum())},
        "std": {"mean": float(std_times.mean()), "total": float(std_times.sum())},
        "mcmc": {"mean": float(mcmc_times.mean()), "total": float(mcmc_times.sum())},
        "slowdown_vs_naive": float((mcmc_times / naive_times).mean()),
        "slowdown_vs_std": float((mcmc_times / std_times).mean()),
    }
    timing_file = os.path.join(save_str, f"{model}_timing_summary_{mcmc_steps}_{temp}_{args.batch_idx}_{args.seed}_fast.json")
    with open(timing_file, "w") as f:
        json.dump(timing_summary, f, indent=2)
    print(f"Results saved to: {save_str}")
    print(f"Timing summary: {timing_file}")
