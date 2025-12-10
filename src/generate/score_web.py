from src.generate.inference.batch_inference import WebAgentScoringPipeline
import argparse
import os
import re
from src.generate.utils import get_model_name

PROMPT_ROOT_PATH = "./prompt_templates/web_agent/scoring"
ROOT_PATH = "./data/web_agent"
DATASET_NAMES = [
    'nnetnav_live_site=allrecipes_num_tasks=79_portion=0',
    'nnetnav_live_site=allrecipes_num_tasks=79_portion=1',
    'nnetnav_live_site=allrecipes_num_tasks=79_portion=2',
    'nnetnav_live_site=allrecipes_num_tasks=79_portion=3',
    'nnetnav_live_site=allrecipes_num_tasks=79_portion=4',
    'nnetnav_live_site=amazon_num_tasks=63_portion=0',
    'nnetnav_live_site=amazon_num_tasks=63_portion=1',
    'nnetnav_live_site=amazon_num_tasks=63_portion=2',
    'nnetnav_live_site=amazon_num_tasks=63_portion=3',
    'nnetnav_live_site=amazon_num_tasks=63_portion=4',
    'nnetnav_live_site=apple_num_tasks=70_portion=0',
    'nnetnav_live_site=apple_num_tasks=70_portion=1',
    'nnetnav_live_site=apple_num_tasks=70_portion=2',
    'nnetnav_live_site=apple_num_tasks=70_portion=3',
    'nnetnav_live_site=apple_num_tasks=70_portion=4',
    'nnetnav_live_site=arxiv_num_tasks=80_portion=0',
    'nnetnav_live_site=arxiv_num_tasks=80_portion=1',
    'nnetnav_live_site=arxiv_num_tasks=80_portion=2',
    'nnetnav_live_site=arxiv_num_tasks=80_portion=3',
    'nnetnav_live_site=arxiv_num_tasks=80_portion=4',
    'nnetnav_live_site=bbc_num_tasks=69_portion=0',
    'nnetnav_live_site=bbc_num_tasks=69_portion=1',
    'nnetnav_live_site=bbc_num_tasks=69_portion=2',
    'nnetnav_live_site=bbc_num_tasks=69_portion=3',
    'nnetnav_live_site=bbc_num_tasks=69_portion=4',
    'nnetnav_live_site=coursera_num_tasks=72_portion=0',
    'nnetnav_live_site=coursera_num_tasks=72_portion=1',
    'nnetnav_live_site=coursera_num_tasks=72_portion=2',
    'nnetnav_live_site=coursera_num_tasks=72_portion=3',
    'nnetnav_live_site=coursera_num_tasks=72_portion=4',
    'nnetnav_live_site=dictionary.cambridge_num_tasks=54_portion=0',
    'nnetnav_live_site=dictionary.cambridge_num_tasks=54_portion=1',
    'nnetnav_live_site=dictionary.cambridge_num_tasks=54_portion=2',
    'nnetnav_live_site=dictionary.cambridge_num_tasks=54_portion=3',
    'nnetnav_live_site=dictionary.cambridge_num_tasks=54_portion=4',
    'nnetnav_live_site=espn_num_tasks=62_portion=0',
    'nnetnav_live_site=espn_num_tasks=62_portion=1',
    'nnetnav_live_site=espn_num_tasks=62_portion=2',
    'nnetnav_live_site=espn_num_tasks=62_portion=3',
    'nnetnav_live_site=espn_num_tasks=62_portion=4',
    'nnetnav_live_site=github_num_tasks=71_portion=0',
    'nnetnav_live_site=github_num_tasks=71_portion=1',
    'nnetnav_live_site=github_num_tasks=71_portion=2',
    'nnetnav_live_site=github_num_tasks=71_portion=3',
    'nnetnav_live_site=github_num_tasks=71_portion=4',
    'nnetnav_live_site=google_maps_num_tasks=75_portion=0',
    'nnetnav_live_site=google_maps_num_tasks=75_portion=1',
    'nnetnav_live_site=google_maps_num_tasks=75_portion=2',
    'nnetnav_live_site=google_maps_num_tasks=75_portion=3',
    'nnetnav_live_site=google_maps_num_tasks=75_portion=4',
    'nnetnav_live_site=google_search_num_tasks=72_portion=0',
    'nnetnav_live_site=google_search_num_tasks=72_portion=1',
    'nnetnav_live_site=google_search_num_tasks=72_portion=2',
    'nnetnav_live_site=google_search_num_tasks=72_portion=3',
    'nnetnav_live_site=google_search_num_tasks=72_portion=4',
    'nnetnav_live_site=huggingface_num_tasks=76_portion=0',
    'nnetnav_live_site=huggingface_num_tasks=76_portion=1',
    'nnetnav_live_site=huggingface_num_tasks=76_portion=2',
    'nnetnav_live_site=huggingface_num_tasks=76_portion=3',
    'nnetnav_live_site=huggingface_num_tasks=76_portion=4',
    'nnetnav_live_site=wolframalpha_num_tasks=66_portion=0',
    'nnetnav_live_site=wolframalpha_num_tasks=66_portion=1',
    'nnetnav_live_site=wolframalpha_num_tasks=66_portion=2',
    'nnetnav_live_site=wolframalpha_num_tasks=66_portion=3',
    'nnetnav_live_site=wolframalpha_num_tasks=66_portion=4'
]

def main(args):
    # dummy data
    rubric_path = os.path.join(ROOT_PATH, "LENS/rubrics", f"rubric.webvoyager.deepseek-reasoner.axtree_points=10_seed={args.seed}.json")
    model_name = get_model_name(args.scoring_model_path)
    if args.synthetic_data:
        source_data_path = os.path.join(ROOT_PATH, "original/synthetic", DATASET_NAMES[0] + ".json")
        output_folder_path = os.path.join(args.output_dir, model_name, f"seed={args.seed}", "synthetic_data", "allrecipes")
    else:
        # real data
        source_data_path = os.path.join(ROOT_PATH, f"sampled/real", "allrecipes" + ".json")
        output_folder_path = os.path.join(args.output_dir, model_name, f"seed={args.seed}", "real_data", "allrecipes")
    
    prompt_template_path = os.path.join(PROMPT_ROOT_PATH, "score.txt")

    pipeline = WebAgentScoringPipeline(
        domain_name="allrecipes",
        rubric_path=rubric_path,
        rubric_key=DATASET_NAMES[0],
        scoring_model_path=args.scoring_model_path,
        source_data_path=source_data_path,
        resume_path=args.resume_path,
        verbose=args.verbose,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_num_seqs=args.max_num_seqs,
        temperature=args.temperature,
        prompt_template_path=prompt_template_path,
    )

    for dataset_name in DATASET_NAMES:
        domain = re.search(r'nnetnav_live_site=(.*)_num_tasks=.*', dataset_name).group(1)
        if args.synthetic_data:
            source_data_path = os.path.join(ROOT_PATH, "original/synthetic", dataset_name + ".json")
            output_folder_path = os.path.join(args.output_dir, model_name, f"seed={args.seed}", f"synthetic_data", domain)
        else:
            # real data
            source_data_path = os.path.join(ROOT_PATH, f"sampled/real", domain + ".json")
            output_folder_path = os.path.join(args.output_dir, model_name, f"seed={args.seed}", f"real_data", domain)
        
        final_path = os.path.join(output_folder_path, dataset_name + ".json")

        # skip if the file already exists
        if os.path.exists(final_path):
            print(f"Skipping {dataset_name} because it already exists")
            continue

        # create output folder if it doesn't exist
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        pipeline.update_data(rubric_path, dataset_name, domain, source_data_path)
        results = pipeline.generate()
        pipeline.save_results(
            examples=results,
            output_path=final_path,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--resume_path", type=str, help="Path to resume from previous run (optional)")
    parser.add_argument("--seed", type=int, choices=[42, 43, 44, 45, 46], help="Seed to use [42-46]", required=True)

    # model args
    parser.add_argument("--scoring_model_path", type=str, required=True, help="Path to the judgement model")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose output")

    # generation args
    parser.add_argument("--temperature", type=float, default=0.00, help="Temperature to use")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size to use")
    parser.add_argument("--max_num_seqs", type=int, default=100, help="Maximum number of sequences to generate")
    parser.add_argument("--synthetic_data", action="store_true", help="Whether to use synthetic data")
    parser.add_argument("--output_dir", type=str, default="./data/web_agent/LENS/scores", help="Output directory")
    args = parser.parse_args()
    main(args)
