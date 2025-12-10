from src.generate.inference.batch_inference.pipelines import ImageClassificationScoringPipeline
import argparse
import os
from utils import get_model_name

PROMPT_ROOT_PATH = "./prompt_templates/image_classification/scoring"
DATASET_NAMES = ["unmet_v11_label_background", "unmet_v11_label_only", "unmet_v11_label_relation", "unmet_v15_label_only", "unmet_v15_label_background", "unmet_v15_label_relation"]

def get_source_data_path(dataset_name, split, seed=None, synthetic_data=False):
    if synthetic_data:
        # real data
        source_data_path = os.path.join(f"./data/image_classification/raw_json/synthetic/{split}", dataset_name + ".json")
    else:
        source_data_path = os.path.join(f"data/image_classification/raw_csv/real/imagenet_{split}_balanced_seed={seed}.csv")
    return source_data_path

def main(args):
    args.output_folder_path = args.output_folder_path+f"_seed={args.seed}"
    if args.synthetic_data:
        output_folder_path = os.path.join(args.output_folder_path, "synthetic_data")
    else:
        output_folder_path = os.path.join(args.output_folder_path, "real_data")
            
    output_folder_path = os.path.join(output_folder_path, f"{get_model_name(args.scoring_model_path)}")
    metadata_folder_path = os.path.join(output_folder_path, "metadata")
    
    prompt_template_path = os.path.join(PROMPT_ROOT_PATH, "score.txt")
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    if not os.path.exists(metadata_folder_path):
        os.makedirs(metadata_folder_path)

    pipeline = ImageClassificationScoringPipeline(
        rubric_path=args.rubric_path,
        dataset=DATASET_NAMES[0],
        scoring_model_path=args.scoring_model_path,
        source_data_path=get_source_data_path(DATASET_NAMES[0], args.split, args.seed, args.synthetic_data),
        image_key=args.image_key,
        resume_path=args.resume_path,
        verbose=args.verbose,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_num_seqs=args.max_num_seqs,
        prompt_template_path=prompt_template_path,
        max_context_window=args.max_context_window,
        max_generate_tokens=args.max_generate_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        multimodal=True
    )
    for dataset_name in DATASET_NAMES:
        source_data_path = get_source_data_path(dataset_name, args.split, args.seed, args.synthetic_data)
        final_path = os.path.join(output_folder_path, dataset_name + ".json")
        metadata_path = os.path.join(metadata_folder_path, dataset_name + "_metadata.json")
        if os.path.exists(final_path):
            print(f"Skipping {dataset_name} because it already exists")
            continue
        pipeline.update_data(args.rubric_path, dataset_name, source_data_path)
        results = pipeline.generate(args.num_examples)
        
        pipeline.save_results(
            examples=results,
            output_path=final_path,
            metadata_path=metadata_path,
            metadata={**args.__dict__}
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--split", type=str, default=None, help="Split to use for the dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--image_key", type=str, default="image_path", help="Key for image path in data")
    parser.add_argument("--label_key", type=str, default="label", help="Key for label in data")
    parser.add_argument("--num_examples", type=int, help="Number of examples to generate (optional)")
    parser.add_argument("--resume_path", type=str, help="Path to resume from previous run (optional)")
    parser.add_argument("--seed", type=int, default=0, help="Seed to use for the dataset")

    # model args
    parser.add_argument("--scoring_model_path", type=str, required=True, help="Path to the judgement model")
    parser.add_argument("--output_folder_path", type=str, default="./data/image_classification/SynQuE/scoring", help="Path to save the results")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose output")
    parser.add_argument("--rubric_path", type=str, required=True, help="Path to the rubric")
   
    # generation args
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size to use")
    parser.add_argument("--max_num_seqs", type=int, default=50, help="Maximum number of sequences to generate")
    parser.add_argument("--synthetic_data", action="store_true", help="Whether to use synthetic data")
    
    # model generation parameters
    parser.add_argument("--max_context_window", type=int, default=8192, help="Maximum number of tokens in the context window")
    parser.add_argument("--max_generate_tokens", type=int, default=64, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.00, help="Temperature for the generation model")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for the generation model")
    
    args = parser.parse_args()
    main(args)
