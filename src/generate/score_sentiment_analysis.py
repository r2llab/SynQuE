from src.generate.inference.batch_inference import SentimentAnalysisScoringPipeline
from src.generate.utils import get_model_name
import argparse
import os

PROMPT_ROOT_PATH = "./prompt_templates/sentiment_analysis/scoring"
DATASET_NAMES = [
    'qwen2.5-7b_zero-shot_v1',
    'qwen2.5-32b_few-shot_bg_train-time-info_v1',
    'llama3.3-70b_zero-shot_bg_train-time-info_v1',
    'qwen2.5-32b_zero-shot_bg_v1',
    'qwen2.5-7b_few-shot_bg_test-time-info_v1',
    'qwen2.5-32b_few-shot_bg_test-time-info_v1',
    'llama3.3-70b_zero-shot_bg_v1',
    'llama3.1-8b_zero-shot_v1',
    'qwen2.5-7b_zero-shot_bg_train-time-info_v1',
    'qwen2.5-32b_zero-shot_bg_test-time-info_v1',
    'qwen2.5-32b_zero-shot_v1',
    'llama3.3-70b_few-shot_bg_v1',
    'qwen2.5-32b_zero-shot_bg_train-time-info_v1',
    'qwen2.5-32b_few-shot_v1',
    'qwen2.5-7b_few-shot_bg_v1',
    'llama3.3-70b_zero-shot_v1',
    'llama3.3-70b_few-shot_v1',
    'llama3.1-8b_zero-shot_bg_v1',
    'llama3.1-8b_few-shot_bg_test-time-info_v1',
    'llama3.1-8b_zero-shot_bg_train-time-info_v1',
    'qwen2.5-7b_zero-shot_bg_test-time-info_v1',
    'llama3.1-8b_few-shot_v1',
    'llama3.3-70b_zero-shot_bg_test-time-info_v1',
    'qwen2.5-7b_few-shot_v1',
    'llama3.1-8b_zero-shot_bg_test-time-info_v1',
    'qwen2.5-32b_few-shot_bg_v1',
    'qwen2.5-7b_few-shot_bg_train-time-info_v1',
    'llama3.3-70b_few-shot_bg_test-time-info_v1',
    'qwen2.5-7b_zero-shot_bg_v1',
    'llama3.1-8b_few-shot_bg_v1',
    'llama3.1-8b_few-shot_bg_train-time-info_v1',
    'llama3.3-70b_few-shot_bg_train-time-info_v1'
]
def main(args):
    # this is dummy code
    rubric_path = f"data/sentiment_analysis/LENS/rubrics/rubric.sentiment_analysis.deepseek-reasoner_num_samples=200_num_points=10_real_seed={args.seed}.json"
    display_model_name = get_model_name(args.scoring_model_path)
    if args.synthetic_data:
        output_folder_path = os.path.join(args.output_folder_path, display_model_name, "seed={}".format(args.seed), "synthetic_data")
        source_data_path = os.path.join("./data/sentiment_analysis/synthetic_data", DATASET_NAMES[0] + ".json")
    else:
        output_folder_path = os.path.join(args.output_folder_path, display_model_name, "seed={}".format(args.seed), "real_data")
        source_data_path = os.path.join("./data/sentiment_analysis/real_data/balanced_real_seed={}.json".format(args.seed))

    folder_path = output_folder_path
    prompt_template_path = os.path.join(PROMPT_ROOT_PATH, "score.txt")

    
    pipeline = SentimentAnalysisScoringPipeline(
        rubric_key=DATASET_NAMES[0],
        rubric_path=rubric_path,
        scoring_model_path=args.scoring_model_path,
        original_data_path=source_data_path,
        resume_path=args.resume_path,
        verbose=args.verbose,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_num_seqs=args.max_num_seqs,
        temperature=args.temperature,
        prompt_template_path=prompt_template_path,
    )

    for dataset_name in DATASET_NAMES:
        if args.synthetic_data:
            # synthetic data output path
            source_data_path = os.path.join("./data/sentiment_analysis/synthetic_data", dataset_name + ".json")
            final_path = os.path.join(folder_path, dataset_name + ".json")
        else:
            # real data output path
            source_data_path = os.path.join("./data/sentiment_analysis/real_data/balanced_real_seed={}.json".format(args.seed))
            final_path = os.path.join(folder_path, dataset_name + ".json")
        if os.path.exists(final_path):
            print(f"Results already exist for {final_path}. Skipping...")
            continue

        pipeline.update_data(rubric_path, dataset_name, source_data_path)
        results = pipeline.generate(args.num_examples)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        pipeline.save_results(
            examples=results,
            output_path=final_path,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--num_examples", type=int, help="Number of examples to generate (optional)")
    parser.add_argument("--resume_path", type=str, help="Path to resume from previous run (optional)")
    parser.add_argument("--synthetic_data", action="store_true", default=False, help="Whether the data is synthetic data")
    parser.add_argument("--seed", type=int, required=True, help="Seed for the synthetic data")
    
    # model args
    parser.add_argument("--scoring_model_path", type=str, required=True, help="Path to the judgement model")
    parser.add_argument("--output_folder_path", type=str, default="data/sentiment_analysis/LENS/scores", help="Path to save the results")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose output")
    parser.add_argument("--temperature", type=float, default=0.00, help="Temperature")
    
    # generation args
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size to use for vLLM generation")
    parser.add_argument("--max_num_seqs", type=int, default=500, help="Maximum number of sequences to parallelize for generation")
    
    args = parser.parse_args()
    main(args)
