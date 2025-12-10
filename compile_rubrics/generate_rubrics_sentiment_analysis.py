import json
import os
import random
from collections import defaultdict
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from omegaconf import DictConfig
from tqdm import auto as tqdm

from utils import fix_double_quotes


# ============================================================================
# Helper Functions
# ============================================================================

def sample_balanced_data(data, num_samples, label_key="sentiment"):
    """
    Returns a class-balanced sample of the data.
    
    Args:
        data: list of dicts, each with keys "headline" and "sentiment" (0,1,2)
        num_samples: total number of samples to return (should be divisible by 3 for perfect balance)
        label_key: key to use for grouping by sentiment
    
    Returns:
        list: balanced sample of data
    """
    random.seed(42)
    
    # Group data by sentiment
    sentiment_groups = defaultdict(list)
    for item in data:
        sentiment = int(item[label_key])
        sentiment_groups[sentiment].append(item)
    
    # Determine number of samples per class
    num_classes = 3
    samples_per_class = num_samples // num_classes
    
    # Sample from each class
    balanced_samples = []
    for sentiment in range(num_classes):
        group = sentiment_groups[sentiment]
        if len(group) < samples_per_class:
            raise ValueError(
                f"Not enough samples for sentiment {sentiment}: "
                f"requested {samples_per_class}, available {len(group)}"
            )
        balanced_samples.extend(random.sample(group, samples_per_class))
    
    # Shuffle the final result
    random.shuffle(balanced_samples)
    return balanced_samples


def preprocess_sentiment_analysis(x, real=False):
    """Extract text from data item based on whether it's real or synthetic."""
    if real:
        return x['text']
    else:
        return x['headline']


def load_synthetic_data(cfg):
    """Load synthetic datasets."""
    syn_db_dataset_map = {}  # dataset_name -> dataset
    data_paths = Path(cfg.data_root_path).glob('*.json')
    
    for path in data_paths:
        dataset_name = path.stem
        if dataset_name not in syn_db_dataset_map:
            syn_db_dataset_map[dataset_name] = []
        
        with path.open('rt') as f:
            data = json.load(f)
        
        data = sample_balanced_data(data, cfg.num_samples, label_key="sentiment")
        filtered_data = map(lambda x: preprocess_sentiment_analysis(x, real=False), data)
        syn_db_dataset_map[dataset_name].extend(list(filtered_data))
        print(f"Loaded {len(syn_db_dataset_map[dataset_name])} examples for {dataset_name}")
    
    return syn_db_dataset_map


def load_real_data(cfg):
    """Load real dataset."""
    print("-" * 100)
    print("Real data")
    
    real_data_path = Path(f'{cfg.real_data_path}/balanced_real_seed={cfg.real_seed}.json')
    real_dataset = []
    
    with real_data_path.open('rt') as f:
        data = json.load(f)
    
    # Real data is already balanced and sampled
    real_dataset.extend(list(map(lambda x: preprocess_sentiment_analysis(x, real=True), data)))
    print(f"Loaded {len(real_dataset)} examples")
    
    return real_dataset


def load_langchain_model(cfg):
    """Load and configure the LangChain model and prompt chains."""
    model = ChatOpenAI(
        model_name=cfg.model.name,
        temperature=0.0
    )
    
    # Load prompt templates
    sim_prompt_path = Path('./prompt_templates/sentiment_analysis/rubric_compilation/sim.txt')
    diff_prompt_path = Path('./prompt_templates/sentiment_analysis/rubric_compilation/diff.txt')
    
    with sim_prompt_path.open('r') as f:
        sim_prompt = f.read()
    
    with diff_prompt_path.open('r') as f:
        diff_prompt = f.read()
    
    similar_prompt_template = ChatPromptTemplate.from_messages([
        ("system", sim_prompt),
        ("user", "Samples from A:\n{A}\n\nSamples from B:\n{B}")
    ])
    
    diff_prompt_template = ChatPromptTemplate.from_messages([
        ("system", diff_prompt),
        ("user", "Similar characteristics between A and B:\n{similar_points}\n\nSamples from A:\n{A}\n\nSamples from B:\n{B}")
    ])
    
    json_parser = JsonOutputParser()
    
    similar_chain = similar_prompt_template | model
    diff_chain = diff_prompt_template | model
    
    return similar_chain, diff_chain, json_parser


def generate_rubrics(
    syn_db_dataset_map,
    real_dataset,
    similar_chain,
    diff_chain,
    json_parser,
    cfg
):
    """Generate rubrics for all datasets."""
    # Determine output file path
    output_filename = (
        f'rubric.sentiment_analysis.{cfg.model.name.replace("/", "--")}'
        f'_num_samples={cfg.num_samples}_num_points={cfg.num_points}_real_seed={cfg.real_seed}.json'
    )
    output_filepath = os.path.join(cfg.output_path, output_filename)
    partial_filepath = output_filepath.replace('.json', '.partial.json')
    
    # Create output directory if it doesn't exist
    output_path = Path(cfg.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize rubric dictionaries
    sims = {}  # dataset_name -> sims
    diffs_synth_from_real = {}  # dataset_name -> diffs
    diffs_real_from_synth = {}  # dataset_name -> diffs
    
    # Load existing rubrics if available
    if os.path.isfile(partial_filepath):
        print(f"Loading existing rubrics from {partial_filepath}")
        with open(partial_filepath, 'rt') as f:
            rubrics = json.load(f)
        sims = rubrics['sims']
        diffs_synth_from_real = rubrics['diffs_synth_from_real']
        diffs_real_from_synth = rubrics['diffs_real_from_synth']
    elif os.path.isfile(output_filepath):
        print(f"Loading existing rubrics from {output_filepath}")
        with open(output_filepath, 'rt') as f:
            rubrics = json.load(f)
        sims = rubrics['sims']
        diffs_synth_from_real = rubrics['diffs_synth_from_real']
        diffs_real_from_synth = rubrics['diffs_real_from_synth']
    
    # Generate rubrics for each dataset
    try:
        for dataset_name in tqdm.tqdm(syn_db_dataset_map.keys(), initial=len(sims)):
            print(f"Generating rubrics for {dataset_name}...")
            
            # We want to make sure for difference generation, we must have sims
            no_sims = True
            
            # Generate similarities
            if dataset_name not in sims or len(sims.get(dataset_name, {})) == 0:
                result = similar_chain.invoke(dict(
                    feedback='similar to',
                    num=cfg.num_points,
                    A=json.dumps(real_dataset),
                    B=json.dumps(syn_db_dataset_map[dataset_name])
                )).content
                sims[dataset_name] = json_parser.parse(fix_double_quotes(result))
            else:
                no_sims = False
                print(f'skipping {dataset_name} sims')
            
            # Generate differences: synthetic from real
            if dataset_name not in diffs_synth_from_real or len(diffs_synth_from_real.get(dataset_name, {})) == 0 or no_sims:
                similar_points = "\n".join(sims[dataset_name])
                result = diff_chain.invoke(dict(
                    feedback='different from',
                    num=cfg.num_points,
                    A=json.dumps(real_dataset),
                    B=json.dumps(syn_db_dataset_map[dataset_name]),
                    similar_points=similar_points
                )).content
                diffs_synth_from_real[dataset_name] = json_parser.parse(fix_double_quotes(result))
            else:
                print(f'skipping {dataset_name} synth from real')
            
            # Generate differences: real from synthetic
            if dataset_name not in diffs_real_from_synth or len(diffs_real_from_synth.get(dataset_name, {})) == 0 or no_sims:
                similar_points = "\n".join(sims[dataset_name])
                result = diff_chain.invoke(dict(
                    feedback='different from',
                    num=cfg.num_points,
                    B=json.dumps(real_dataset),
                    A=json.dumps(syn_db_dataset_map[dataset_name]),
                    similar_points=similar_points
                )).content
                diffs_real_from_synth[dataset_name] = json_parser.parse(fix_double_quotes(result))
            else:
                print(f'skipping {dataset_name} real from synth')
                
    except Exception as e:
        print(f"Error: {e}")
        # Save partial results
        with open(partial_filepath, 'wt') as f:
            json.dump(dict(
                sims=sims,
                diffs_synth_from_real=diffs_synth_from_real,
                diffs_real_from_synth=diffs_real_from_synth
            ), f, indent=2)
        raise e
    
    # Save final results
    with open(output_filepath, 'wt') as f:
        json.dump(dict(
            sims=sims,
            diffs_synth_from_real=diffs_synth_from_real,
            diffs_real_from_synth=diffs_real_from_synth
        ), f, indent=2)


# ============================================================================
# Main Function
# ============================================================================

@hydra.main(version_base=None, config_path="conf", config_name="sentiment_analysis.yaml")
def main(cfg: DictConfig) -> None:
    """Main execution function."""
    print(OmegaConf.to_yaml(cfg))
    cfg = cfg.sentiment_analysis.rubric_generation

    
    similar_chain, diff_chain, json_parser = load_langchain_model(cfg)
    syn_db_dataset_map = load_synthetic_data(cfg)
    real_dataset = load_real_data(cfg)
    
    generate_rubrics(
        syn_db_dataset_map,
        real_dataset,
        similar_chain,
        diff_chain,
        json_parser,
        cfg
    )


if __name__ == "__main__":
    main()
