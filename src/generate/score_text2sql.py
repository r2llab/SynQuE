from src.generate.inference.batch_inference import Text2SQLScoringPipeline
import argparse
import os
from src.generate.utils import get_model_name

PROMPT_ROOT_PATH = "./prompt_templates/text2sql/scoring"

DATASET_NAMES = [
    'llama3.1-8b_1000_few-shot_bg_test-time-info_v1',
    'llama3.1-8b_1000_few-shot_bg_v1',
    'llama3.1-8b_1000_zero-shot_bg_test-time-info_v1',
    'llama3.1-8b_1000_zero-shot_bg_v1',
    'qwen2.5-coder-7b_1000_few-shot_bg_test-time-info_v1',
    'qwen2.5-coder-7b_1000_few-shot_bg_v1',
    'qwen2.5-coder-7b_1000_zero-shot_bg_test-time-info_v1',
    'qwen2.5-coder-7b_1000_zero-shot_bg_v1'
]

def main(args):
    # dummy data
    rubric_path = f"data/text2sql/LENS/rubrics/rubric.text2sql.deepseek-reasoner_num_points=10_num_samples=30_seed={args.seed}.json"
    display_model_name = get_model_name(args.scoring_model_path)
    if args.synthetic_data:
        root_output_folder_path = os.path.join(args.output_folder_path, display_model_name, "seed={}".format(args.seed), "synthetic_data")
        source_data_path = os.path.join("./data/text2sql", "data", "movie_platform", DATASET_NAMES[0] + ".json")
    else:
        root_output_folder_path = os.path.join(args.output_folder_path, display_model_name, "seed={}".format(args.seed), "real_data")
        source_data_path = os.path.join("./data/text2sql/data", "real", f"dev_movie_platform_seed={args.seed}.json")
    
    prompt_template_path = os.path.join(PROMPT_ROOT_PATH, "score.txt")

    pipeline = Text2SQLScoringPipeline(
        rubric_path=rubric_path,
        rubric_key=DATASET_NAMES[0], # dummy data
        scoring_model_path=args.scoring_model_path,
        source_data_path=source_data_path,
        db_id="movie_platform", # dummy data
        resume_path=args.resume_path,
        verbose=args.verbose,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_num_seqs=args.max_num_seqs,
        prompt_template_path=prompt_template_path,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    for dataset_name in DATASET_NAMES:
        for db_id in ["app_store", "movie_platform", "computer_student"]:
            output_folder_path = os.path.join(root_output_folder_path, db_id)
            final_path = os.path.join(output_folder_path, dataset_name + ".json")

            if os.path.exists(final_path):
                print(f"Skipping {dataset_name}:{db_id} because it already exists")
                continue

            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            
            if args.synthetic_data:
                source_data_path = os.path.join("./data/text2sql", "data", db_id, dataset_name + ".json")
            else:
                source_data_path = os.path.join("./data/text2sql/data", "real", f"dev_{db_id}_seed={args.seed}.json")

            pipeline.update_data(rubric_path, db_id, dataset_name, source_data_path)
            results = pipeline.generate(args.num_examples)
            
            pipeline.save_results(
                examples=results,
                output_path=final_path
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--num_examples", type=int, help="Number of examples to generate (optional)")
    parser.add_argument("--resume_path", type=str, help="Path to resume from previous run (optional)")
    parser.add_argument("--seed", type=int, required=True, help="Seed of the data split")

    # model args
    parser.add_argument("--scoring_model_path", type=str, required=True, help="Path to the judgement model")
    parser.add_argument("--output_folder_path", type=str, default="./data/text2sql/LENS/scores", help="Path to save the results")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose output")
   
    # generation args
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size to use for vLLM generation")
    parser.add_argument("--max_num_seqs", type=int, default=500, help="Maximum number of sequences to parallelize for generation")
    parser.add_argument("--synthetic_data", action="store_true", help="Whether to use synthetic data")
    parser.add_argument("--temperature", type=float, default=0.00, help="Temperature to use")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p to use")

    args = parser.parse_args()
    main(args)
