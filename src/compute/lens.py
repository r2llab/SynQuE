import numpy as np
import argparse
import os
from glob import glob
import json
import pandas as pd

from src.compute.utils import compute_correlation


SEEDS = [42, 43, 44, 45, 46]

# web navigation domain constants
WEBSITE_NAMES = [
    'allrecipes',
    'amazon',
    'apple',
    'arxiv',
    'bbc',
    'coursera',
    'dictionary.cambridge',
    'espn',
    'github',
    'google_maps',
    'google_search',
    'huggingface',
    'wolframalpha',
]
PARTITIONS = [0, 1, 2, 3, 4]

EPSILON = 1e-6


def compute_lens_score_normalized(synth_examples, real_examples):
    """Computes a balanced LLM score that considers both synthetic and real examples.
    
    This function calculates a weighted average of two error scores:
    1. How often synthetic examples are mistakenly classified as real
    2. How often real examples are mistakenly classified as synthetic
    
    The function uses normalization factors derived from real examples to adjust
    the raw scores before computing probabilities.
    
    Args:
        synth_examples (list): The synthetic examples with LLM judgments.
        real_examples (list): The real examples with LLM judgments.
        
    Returns:
        float: A balanced LLM score representing the overall error rate across
               both synthetic and real examples, weighted by their respective counts.
    """    
    SCORE_MAP = {
        'very likely': 4,
        'likely': 3,
        'unsure': 2,
        'unlikely': 1,
        'very unlikely': 0,
    }
    
    # compute normalization factors for synth examples
    score_real_given_synth_loc_A_judgement_factor_synth = np.mean([SCORE_MAP[item['score_real_given_synth_loc_A_judgement']] for item in real_examples])
    score_real_given_synth_loc_B_judgement_factor_synth = np.mean([SCORE_MAP[item['score_real_given_synth_loc_B_judgement']] for item in real_examples])
    score_synth_given_synth_loc_A_judgement_factor_synth = np.mean([SCORE_MAP[item['score_synth_given_synth_loc_A_judgement']] for item in real_examples])
    score_synth_given_synth_loc_B_judgement_factor_synth = np.mean([SCORE_MAP[item['score_synth_given_synth_loc_B_judgement']] for item in real_examples])
    
    # compute normalization factors for real examples
    score_real_given_synth_loc_A_judgement_factor_real = np.mean([SCORE_MAP[item['score_real_given_synth_loc_A_judgement']] for item in synth_examples])
    score_real_given_synth_loc_B_judgement_factor_real = np.mean([SCORE_MAP[item['score_real_given_synth_loc_B_judgement']] for item in synth_examples])
    score_synth_given_synth_loc_A_judgement_factor_real = np.mean([SCORE_MAP[item['score_synth_given_synth_loc_A_judgement']] for item in synth_examples])
    score_synth_given_synth_loc_B_judgement_factor_real = np.mean([SCORE_MAP[item['score_synth_given_synth_loc_B_judgement']] for item in synth_examples])

    score_real_given_synth_loc_A_judgement_factor_real = 1 if score_real_given_synth_loc_A_judgement_factor_real == 0 else score_real_given_synth_loc_A_judgement_factor_real
    score_real_given_synth_loc_B_judgement_factor_real = 1 if score_real_given_synth_loc_B_judgement_factor_real == 0 else score_real_given_synth_loc_B_judgement_factor_real
    score_synth_given_synth_loc_A_judgement_factor_real = 1 if score_synth_given_synth_loc_A_judgement_factor_real == 0 else score_synth_given_synth_loc_A_judgement_factor_real
    score_synth_given_synth_loc_B_judgement_factor_real = 1 if score_synth_given_synth_loc_B_judgement_factor_real == 0 else score_synth_given_synth_loc_B_judgement_factor_real

    score_real_given_synth_loc_A_judgement_factor_synth = max(score_real_given_synth_loc_A_judgement_factor_synth, EPSILON)
    score_real_given_synth_loc_B_judgement_factor_synth = max(score_real_given_synth_loc_B_judgement_factor_synth, EPSILON)
    score_synth_given_synth_loc_A_judgement_factor_synth = max(score_synth_given_synth_loc_A_judgement_factor_synth, EPSILON)
    score_synth_given_synth_loc_B_judgement_factor_synth = max(score_synth_given_synth_loc_B_judgement_factor_synth, EPSILON)

    score_real_given_synth_loc_A_judgement_factor_synth = max(score_real_given_synth_loc_A_judgement_factor_synth, EPSILON)
    score_real_given_synth_loc_B_judgement_factor_synth = max(score_real_given_synth_loc_B_judgement_factor_synth, EPSILON)
    score_synth_given_synth_loc_A_judgement_factor_synth = max(score_synth_given_synth_loc_A_judgement_factor_synth, EPSILON)
    score_synth_given_synth_loc_B_judgement_factor_synth = max(score_synth_given_synth_loc_B_judgement_factor_synth, EPSILON)
    
    # compute llm scores
    error_scores = []
    error_scores_real = []
    for item in synth_examples:
        # real given loc A
        h_real_given_loc_A = SCORE_MAP[item['score_real_given_synth_loc_A_judgement']] / score_real_given_synth_loc_A_judgement_factor_synth
        h_synth_given_loc_A = SCORE_MAP[item['score_synth_given_synth_loc_A_judgement']] / score_synth_given_synth_loc_A_judgement_factor_synth
        p_real_given_loc_A = h_real_given_loc_A / (h_real_given_loc_A + h_synth_given_loc_A + EPSILON)

        # real given loc B
        h_real_given_loc_B = SCORE_MAP[item['score_real_given_synth_loc_B_judgement']] / score_real_given_synth_loc_B_judgement_factor_synth
        h_synth_given_loc_B = SCORE_MAP[item['score_synth_given_synth_loc_B_judgement']] / score_synth_given_synth_loc_B_judgement_factor_synth
        p_real_given_loc_B = h_real_given_loc_B / (h_real_given_loc_B + h_synth_given_loc_B + EPSILON)

        error_scores.append((p_real_given_loc_A + p_real_given_loc_B) / 2)
    
    for item in real_examples:
        # real given loc A
        h_real_given_loc_A = SCORE_MAP[item['score_real_given_synth_loc_A_judgement']] / score_real_given_synth_loc_A_judgement_factor_real
        h_synth_given_loc_A = SCORE_MAP[item['score_synth_given_synth_loc_A_judgement']] / score_synth_given_synth_loc_A_judgement_factor_real
        p_synth_given_loc_A = h_synth_given_loc_A / (h_real_given_loc_A + h_synth_given_loc_A + EPSILON)

        # real given loc B
        h_real_given_loc_B = SCORE_MAP[item['score_real_given_synth_loc_B_judgement']] / score_real_given_synth_loc_B_judgement_factor_real
        h_synth_given_loc_B = SCORE_MAP[item['score_synth_given_synth_loc_B_judgement']] / score_synth_given_synth_loc_B_judgement_factor_real
        p_synth_given_loc_B = h_synth_given_loc_B / (h_real_given_loc_B + h_synth_given_loc_B + EPSILON)

        error_scores_real.append((p_synth_given_loc_A + p_synth_given_loc_B) / 2)

    return (np.mean(error_scores) * len(synth_examples) + np.mean(error_scores_real) * len(real_examples)) / (len(synth_examples) + len(real_examples))


def compute_lens_score(examples, all_scores=False, real_data=False) -> float:
    """Computes the LLM score for the given examples. This only uses four scores to normalize.

    Args:
        examples (_type_): List of examples
        all_scores (bool, optional): Whether to return all scores. Defaults to False.
        real_data (bool, optional): Whether to return real data scores. Defaults to False.

    Returns:
        float: LLM score
    """    
    SCORE_MAP = {
        'very likely': 4,
        'likely': 3,
        'unsure': 2,
        'unlikely': 1,
        'very unlikely': 0,
    }
    llm_scores = []
    score_real_given_synth_loc_A_ls = []
    score_synth_given_synth_loc_A_ls = []
    score_real_given_synth_loc_B_ls = []
    score_synth_given_synth_loc_B_ls = []
    for item in examples:
        score_real_given_synth_loc_A = SCORE_MAP[item['score_real_given_synth_loc_A_judgement']]
        score_synth_given_synth_loc_A = SCORE_MAP[item['score_synth_given_synth_loc_A_judgement']]
        score_real_given_synth_loc_B = SCORE_MAP[item['score_real_given_synth_loc_B_judgement']]
        score_synth_given_synth_loc_B = SCORE_MAP[item['score_synth_given_synth_loc_B_judgement']]
        p_real_given_loc_A = score_real_given_synth_loc_A / max(1e-6, score_real_given_synth_loc_A + score_synth_given_synth_loc_A)
        p_real_given_loc_B = score_real_given_synth_loc_B / max(1e-6, score_real_given_synth_loc_B + score_synth_given_synth_loc_B)
        p_synth_given_loc_A = score_synth_given_synth_loc_A / max(1e-6, score_real_given_synth_loc_A + score_synth_given_synth_loc_A)
        p_synth_given_loc_B = score_synth_given_synth_loc_B / max(1e-6, score_real_given_synth_loc_B + score_synth_given_synth_loc_B)
        if real_data:
            llm_score = (p_synth_given_loc_A + p_synth_given_loc_B) / 2
        else:
            llm_score = (p_real_given_loc_A + p_real_given_loc_B) / 2
        score_real_given_synth_loc_A_ls.append(score_real_given_synth_loc_A)
        score_synth_given_synth_loc_A_ls.append(score_synth_given_synth_loc_A)
        score_real_given_synth_loc_B_ls.append(score_real_given_synth_loc_B)
        score_synth_given_synth_loc_B_ls.append(score_synth_given_synth_loc_B)
        llm_scores.append(llm_score)
    if all_scores:
        return np.mean(llm_scores), np.mean(score_real_given_synth_loc_A_ls), np.mean(score_synth_given_synth_loc_A_ls), np.mean(score_real_given_synth_loc_B_ls), np.mean(score_synth_given_synth_loc_B_ls)
    return np.mean(llm_scores)



def main(args):
    results = []
    model_name = args.data_path.split("/")[-1]
    if args.task == "sentiment_analysis":
        for seed in SEEDS:
            data_path = os.path.join(args.data_path, f"seed={seed}")
            real_data_path = os.path.join(data_path, "real_data")
            synth_data_path = os.path.join(data_path, "synthetic_data")
            dataset_paths = glob(os.path.join(synth_data_path, "*.json"))
            for dataset_path in dataset_paths:
                dataset_name = dataset_path.split("/")[-1].replace(".json", "")
                synth_examples = json.load(open(dataset_path, "rt"))
                scores = compute_lens_score(synth_examples)
                if os.path.exists(os.path.join(real_data_path, f"{dataset_name}.json")):
                    real_examples = json.load(open(os.path.join(real_data_path, f"{dataset_name}.json"), "rt"))
                    debiased_score = compute_lens_score_normalized(synth_examples, real_examples)
                else:
                    debiased_score = None
                results.append({
                    "seed": seed,
                    "dataset_name": dataset_name,
                    "lens_score": scores,
                    "debiased_lens_score": debiased_score
                })
    elif args.task == "text2sql":
        for seed in SEEDS:
            data_path = os.path.join(args.data_path, f"seed={seed}")
            real_data_path = os.path.join(data_path, "real_data")
            synth_data_path = os.path.join(data_path, "synthetic_data")
            dataset_paths = glob(os.path.join(synth_data_path, "*_*", "*.json"))
            for dataset_path in dataset_paths:
                db_id = dataset_path.split("/")[-2]
                dataset_name = dataset_path.split("/")[-1].replace(".json", "")
                synth_examples = json.load(open(dataset_path, "rt"))
                scores = compute_lens_score(synth_examples)
                if os.path.exists(os.path.join(real_data_path, db_id, f"{dataset_name}.json")):
                    real_examples = json.load(open(os.path.join(real_data_path, db_id, f"{dataset_name}.json"), "rt"))
                    debiased_score = compute_lens_score_normalized(synth_examples, real_examples)
                else:
                    debiased_score = None
                results.append({
                    "seed": seed,
                    "db_id": db_id,
                    "dataset_name": dataset_name,
                    "debiased_lens_score": debiased_score
                })
    elif args.task == "web":
        # For each website
        for website in WEBSITE_NAMES:
            for seed in SEEDS:
                for partition in PARTITIONS:
                    # Build base paths
                    data_path = os.path.join(args.data_path, f"seed={seed}")
                    real_data_dir = os.path.join(data_path, "real_data", website)
                    synth_data_dir = os.path.join(data_path, "synthetic_data", website)
                    synth_file_pattern = f"nnetnav_live_site={website}_num_tasks=*_portion={partition}.json"
                    real_file_pattern = synth_file_pattern

                    synth_file_path = os.path.join(synth_data_dir, synth_file_pattern)
                    real_file_path = os.path.join(real_data_dir, real_file_pattern)

                    synth_files = glob(synth_file_path)
                    if len(synth_files) == 0:
                        raise ValueError(f"No synthetic data for {website} with seed {seed} and partition {partition}. Expected file: {synth_file_path}")

                    synth_file = synth_files[0]
                    synth_examples = json.load(open(synth_file, "rt"))
                    score = compute_lens_score(synth_examples)

                    # Check for the corresponding real data. Only compute normalized if it exists.
                    real_files = glob(real_file_path)
                    if len(real_files) > 0 and os.path.exists(real_files[0]):
                        real_file = real_files[0]
                        real_examples = json.load(open(real_file, "rt"))
                        debiased_score = compute_lens_score_normalized(synth_examples, real_examples)
                    else:
                        debiased_score = None

                    results.append({
                        "website": website,
                        "seed": seed,
                        "partition": partition,
                        "debiased_lens_score": debiased_score
                    })
    df_results = pd.DataFrame(results)
    # df_results.to_csv(os.path.join(args.output_path, f"{args.task}_{model_name}_lens_scores.csv"), index=False)
    df_corr_results = compute_correlation(df_results, "debiased_lens", args.task, args.task_performance_path)
    df_corr_results.to_csv(os.path.join(args.output_path, f"{args.task}_{model_name}_lens_correlation.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the generated scores")
    parser.add_argument("--task_performance_path", type=str, help="Path to the task performance csv file")
    parser.add_argument("--task", type=str, required=True, choices=["sentiment_analysis", "text2sql", "web", "image"], help="Name of the task")
    parser.add_argument("--output_path", type=str, default="./results", help="Path to save the results")
    args = parser.parse_args()

    main(args)