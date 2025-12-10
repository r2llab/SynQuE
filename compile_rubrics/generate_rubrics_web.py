import json
import os
import re
from pathlib import Path

import hydra
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from omegaconf import DictConfig
from tqdm import auto as tqdm

from utils import clean_text_output


# ============================================================================
# Helper Functions
# ============================================================================

def load_axtrees(cfg):
    """Load real and synthetic axtree maps from JSON files."""
    REAL_AXTREE_PATH = Path(cfg.axtree_real_path)
    SYN_AXTREE_PATH = Path(cfg.axtree_synthetic_path)
    
    with REAL_AXTREE_PATH.open('rt') as f:
        real_axtree_map = json.load(f)
    
    with SYN_AXTREE_PATH.open('rt') as f:
        syn_axtree_map = json.load(f)
    
    return real_axtree_map, syn_axtree_map


def load_datasets(cfg):
    """
    Load synthetic and real datasets.
    
    We use the full synthetic dataset and sampled real dataset.
    """
    DATA_ROOT_PATH = Path(cfg.synthetic_data_root)
    REAL_DATA_PATH = Path(cfg.real_data_path)
    
    # Load synthetic datasets
    print("Processing Synthetic data")
    syn_dataset_map = {}  # website -> dataset_name -> data
    data_paths = DATA_ROOT_PATH.glob('*.json')
    
    for path in data_paths:
        website = re.search(r'site=(.+?)_num_tasks', path.name).group(1)
        if website not in syn_dataset_map:
            syn_dataset_map[website] = {}
        
        with path.open('rt') as f:
            data = json.load(f)
        
        syn_dataset_map[website][path.stem] = data
        print(f"Loaded {len(syn_dataset_map[website][path.stem])} examples for {website} -> {path.stem}")
    
    assert len(syn_dataset_map.keys()) == cfg.num_websites, \
        f"Expected {cfg.num_websites} websites, got {len(syn_dataset_map.keys())}"
    
    total_num_examples = sum(len(dataset) for dataset in syn_dataset_map.values())
    assert total_num_examples == cfg.total_num_examples, \
        f"Expected {cfg.total_num_examples} datasets, got {total_num_examples}"
    
    # Load real datasets
    print("-" * 100)
    print("Processing Real data")
    
    real_dataset_map = {}  # website -> data
    for path in REAL_DATA_PATH.glob('*.json'):
        website = path.stem.lower().split('_seed')[0]
        seed = re.search(r'seed=(\d+)', path.stem)
        
        if seed is not None:
            seed = int(seed.group(1))
        else:
            continue
        
        if seed != cfg.seed:
            continue
        
        print(f"Processing {website} with seed {seed}")
        with path.open('rt') as f:
            real_dataset = json.load(f)
        
        real_dataset = list(real_dataset.values())
        real_dataset_map[website] = real_dataset
        print(f"Loaded {len(real_dataset_map[website])} examples for {website}")
    
    assert len(real_dataset_map.keys()) == cfg.num_websites, \
        f"Expected {cfg.num_websites} websites, got {len(real_dataset_map.keys())}"
    
    # Assert all websites from synthetic and the real data are the same set
    assert set(syn_dataset_map.keys()) == set(real_dataset_map.keys()), \
        f"Expected the same websites, got {set(syn_dataset_map.keys())} and {set(real_dataset_map.keys())}"
    
    return syn_dataset_map, real_dataset_map


def load_langchain_model(cfg):
    """Load and configure the LangChain model and prompt chains."""
    model = ChatOpenAI(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE'),
        model_name=cfg.model.name,
        temperature=0.0
    )
    
    # Load prompt templates
    sim_prompt_path = Path('./prompt_templates/web_agent/rubric_compilation/sim.txt')
    diff_prompt_path = Path('./prompt_templates/web_agent/rubric_compilation/diff.txt')
    
    with sim_prompt_path.open('r') as f:
        sim_prompt_text = f.read()
    
    with diff_prompt_path.open('r') as f:
        diff_prompt_text = f.read()
    
    similar_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a world class data analyst on analyzing user intents in web navigation tasks."),
        ("user", sim_prompt_text)
    ])
    
    diff_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a world class data analyst on analyzing user intents in web navigation tasks."),
        ("user", diff_prompt_text)
    ])
    
    json_parser = JsonOutputParser()
    
    similar_chain = similar_prompt_template | model
    diff_chain = diff_prompt_template | model
    
    return similar_chain, diff_chain, json_parser


def generate_rubrics(
    syn_dataset_map,
    real_dataset_map,
    real_axtree_map,
    syn_axtree_map,
    similar_chain,
    diff_chain,
    json_parser,
    cfg
):
    """Generate rubrics for all websites and datasets."""
    output_filename = (
        f'rubric.webvoyager.{cfg.model.name.replace("/", "--")}'
        f'.axtree_points={cfg.num_points}_{cfg.prompt_version}_seed={cfg.seed}.json'
    )
    
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)
    
    output_filepath = os.path.join(cfg.output_path, output_filename)
    partial_filepath = output_filepath.replace('.json', '.partial.json')
    
    # Initialize rubric dictionaries
    sims = {}  # website -> dataset_name -> sims
    diffs_synth_from_real = {}  # website -> dataset_name -> diffs
    diffs_real_from_synth = {}  # website -> dataset_name -> diffs
    
    # Load existing rubrics if available
    if os.path.isfile(output_filepath):
        print(f"Loading existing rubrics from {output_filepath}")
        with open(output_filepath, 'rt') as f:
            rubrics = json.load(f)
        sims = rubrics['sims']
        diffs_synth_from_real = rubrics['diffs_synth_from_real']
        diffs_real_from_synth = rubrics['diffs_real_from_synth']
    elif os.path.isfile(partial_filepath):
        print(f"Loading partial rubrics from {partial_filepath}")
        with open(partial_filepath, 'rt') as f:
            rubrics = json.load(f)
        sims = rubrics['sims']
        diffs_synth_from_real = rubrics['diffs_synth_from_real']
        diffs_real_from_synth = rubrics['diffs_real_from_synth']
    else:
        print("No rubrics found, starting from scratch")
    
    # Generate rubrics for each website and dataset
    try:
        for website in tqdm.tqdm(syn_dataset_map.keys(), initial=len(sims)):
            # Initialize website dictionaries if needed
            if website not in sims:
                sims[website] = {}
            if website not in diffs_synth_from_real:
                diffs_synth_from_real[website] = {}
            if website not in diffs_real_from_synth:
                diffs_real_from_synth[website] = {}
            
            print(f"Generating rubrics for {website}...")
            
            for dataset_name in tqdm.tqdm(syn_dataset_map[website].keys(), initial=len(sims)):
                print(f"Generating rubrics for {dataset_name}...")
                
                # Skip if all rubrics already exist and are non-empty
                if (dataset_name in sims[website] and 
                    dataset_name in diffs_synth_from_real[website] and 
                    dataset_name in diffs_real_from_synth[website]):
                    if (len(sims[website][dataset_name]) > 0 and 
                        len(diffs_synth_from_real[website][dataset_name]) > 0 and 
                        len(diffs_real_from_synth[website][dataset_name]) > 0):
                        print(f'skipping {dataset_name} because it already exists\n')
                        continue
                
                # Initialize empty strings if not present
                sims[website][dataset_name] = (
                    "" if dataset_name not in sims[website] else sims[website][dataset_name]
                )
                diffs_synth_from_real[website][dataset_name] = (
                    "" if dataset_name not in diffs_synth_from_real[website] 
                    else diffs_synth_from_real[website][dataset_name]
                )
                diffs_real_from_synth[website][dataset_name] = (
                    "" if dataset_name not in diffs_real_from_synth[website] 
                    else diffs_real_from_synth[website][dataset_name]
                )
                
                # Compute similarities
                if sims[website][dataset_name] == "":
                    sim_output = similar_chain.invoke(dict(
                        feedback='similar to',
                        num=cfg.num_points,
                        A=json.dumps(real_dataset_map[website]),
                        B=json.dumps(syn_dataset_map[website][dataset_name]),
                        A_tree=json.dumps(real_axtree_map[website]),
                        B_tree=json.dumps(syn_axtree_map[dataset_name])
                    )).content
                    
                    cleaned_sim_output = clean_text_output(sim_output)
                    try:
                        sims[website][dataset_name] = json_parser.parse(cleaned_sim_output)
                    except Exception as e:
                        print(f"Error parsing {website} {dataset_name}: {e}")
                        print(f"Cleaned output: {cleaned_sim_output}")
                        sims[website][dataset_name] = cleaned_sim_output
                
                # Compute differences using the similarities
                if isinstance(sims[website][dataset_name], str):
                    similar_points = sims[website][dataset_name]
                elif isinstance(sims[website][dataset_name], list):
                    similar_points = "\n".join(sims[website][dataset_name])
                else:
                    continue
                
                # Compute synth from real differences
                if diffs_synth_from_real[website][dataset_name] == "":
                    diff_output = diff_chain.invoke(dict(
                        feedback='different from',
                        num=cfg.num_points,
                        A=json.dumps(real_dataset_map[website]),
                        B=json.dumps(syn_dataset_map[website][dataset_name]),
                        similar_points=similar_points,
                        A_tree=json.dumps(real_axtree_map[website]),
                        B_tree=json.dumps(syn_axtree_map[dataset_name])
                    )).content
                    
                    cleaned_diff_output = clean_text_output(diff_output)
                    try:
                        diffs_synth_from_real[website][dataset_name] = json_parser.parse(cleaned_diff_output)
                    except Exception as e:
                        print(f"Error parsing {website} {dataset_name}: {e}")
                        print(f"Cleaned output: {cleaned_diff_output}")
                        diffs_synth_from_real[website][dataset_name] = cleaned_diff_output
                
                # Compute real from synth differences
                if diffs_real_from_synth[website][dataset_name] == "":
                    diff_output = diff_chain.invoke(dict(
                        feedback='different from',
                        num=cfg.num_points,
                        B=json.dumps(real_dataset_map[website]),
                        A=json.dumps(syn_dataset_map[website][dataset_name]),
                        similar_points=similar_points,
                        A_tree=json.dumps(real_axtree_map[website]),
                        B_tree=json.dumps(syn_axtree_map[dataset_name])
                    )).content
                    
                    cleaned_diff_output = clean_text_output(diff_output)
                    try:
                        diffs_real_from_synth[website][dataset_name] = json_parser.parse(cleaned_diff_output)
                    except Exception as e:
                        print(f"Error parsing {website} {dataset_name}: {e}")
                        print(f"Cleaned output: {cleaned_diff_output}")
                        diffs_real_from_synth[website][dataset_name] = cleaned_diff_output
                        
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

@hydra.main(version_base=None, config_path="conf", config_name="web.yaml")
def main(cfg: DictConfig) -> None:
    """Main execution function."""
    cfg = cfg.web.rubric_generation
    
    similar_chain, diff_chain, json_parser = load_langchain_model(cfg)
    syn_dataset_map, real_dataset_map = load_datasets(cfg)
    real_axtree_map, syn_axtree_map = load_axtrees(cfg)
    
    generate_rubrics(
        syn_dataset_map,
        real_dataset_map,
        real_axtree_map,
        syn_axtree_map,
        similar_chain,
        diff_chain,
        json_parser,
        cfg
    )


if __name__ == "__main__":
    main()
