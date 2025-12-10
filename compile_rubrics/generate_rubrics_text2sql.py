import json
import os
from pathlib import Path

import hydra
import numpy as np
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from omegaconf import DictConfig
from tqdm import auto as tqdm

from utils import fix_double_quotes


# ============================================================================
# Helper Functions
# ============================================================================

def load_synthetic_data(cfg):
    """Load synthetic datasets."""
    syn_db_dataset_map = {}  # db_id -> dataset_name -> dataset
    data_root_paths = Path(cfg.data_root).glob('*_*')
    
    np.random.seed(42)  # Just for synthetic data
    for db_path in data_root_paths:
        # For each db_id
        db_id = db_path.stem
        syn_db_dataset_map[db_id] = {}
        
        for dataset_path in db_path.glob('*.json'):
            dataset_name = dataset_path.stem
            if dataset_name not in syn_db_dataset_map[db_id]:
                syn_db_dataset_map[db_id][dataset_name] = []
            
            with dataset_path.open('rt') as f:
                data = json.load(f)
                filtered_data = map((lambda x: x['question']), data)
                syn_db_dataset_map[db_id][dataset_name].extend(
                    np.random.choice(
                        list(filtered_data), 
                        cfg.num_synthetic_samples, 
                        replace=False
                    ).tolist()
                )
                print(f"Loaded {len(syn_db_dataset_map[db_id][dataset_name])} examples for {dataset_name} in {db_id}")
    
    return syn_db_dataset_map


def load_real_data(cfg):
    """Load real datasets."""
    print("-" * 100)
    print("Real data")
    
    real_dataset_map = {}  # db_id -> dataset
    real_data_paths = Path(f'{cfg.data_root}/real').glob(f'*seed={cfg.real_seed}.json')
    
    for path in real_data_paths:
        db_id = path.stem.replace('dev_', '').replace(f'_seed={cfg.real_seed}', '')
        with path.open('rt') as f:
            data = json.load(f)
        real_dataset_map[db_id] = list(map(lambda x: x['question'], data))  # Data is already sampled
        print(f"Loaded {len(real_dataset_map[db_id])} examples for {db_id}")
    
    assert len(real_dataset_map.keys()) == 3, \
        f"Expected 3 real datasets, got {len(real_dataset_map.keys())}"
    
    return real_dataset_map


def load_langchain_model(cfg):
    """Load and configure the LangChain model and prompt chains."""
    model = ChatOpenAI(
        model_name=cfg.model.name,
        temperature=0.0
    )
    
    # Load prompt templates
    sim_prompt_path = Path('./prompt_templates/text2sql/rubric_compilation/sim.txt')
    diff_prompt_path = Path('./prompt_templates/text2sql/rubric_compilation/diff.txt')
    
    with sim_prompt_path.open('r') as f:
        sim_prompt_text = f.read()
    
    with diff_prompt_path.open('r') as f:
        diff_prompt_text = f.read()
    
    similar_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a world class data analyst on database queries in natural language."),
        ("user", sim_prompt_text)
    ])
    
    diff_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a world class data analyst on database queries in natural language."),
        ("user", diff_prompt_text)
    ])
    
    json_parser = JsonOutputParser()
    
    similar_chain = similar_prompt_template | model
    diff_chain = diff_prompt_template | model
    
    return similar_chain, diff_chain, json_parser


def generate_rubrics(
    syn_db_dataset_map,
    real_dataset_map,
    similar_chain,
    diff_chain,
    json_parser,
    cfg
):
    """Generate rubrics for all databases and datasets."""
    # Determine output filename
    output_filename = (
        f'rubric.text2sql.{cfg.model.name.replace("/", "--")}'
        f'_num_points={cfg.num_points}_num_real_samples=30'
        f'_num_synthetic_samples={cfg.num_synthetic_samples}_seed={cfg.real_seed}.json'
    )
    output_filepath = os.path.join(cfg.output_path, output_filename)
    partial_filepath = output_filepath.replace('.json', '.partial.json')
    
    # Create output directory if it doesn't exist
    os.makedirs(cfg.output_path, exist_ok=True)
    
    # Initialize rubric dictionaries
    sims = {}  # db_id -> dataset_name -> sims
    diffs_synth_from_real = {}  # db_id -> dataset_name -> diffs
    diffs_real_from_synth = {}  # db_id -> dataset_name -> diffs
    
    # Load existing rubrics if available
    if os.path.isfile(partial_filepath):
        print(f"Found partial results in {partial_filepath}\nloading partial results...")
        with open(partial_filepath, 'rt') as f:
            tmp = json.load(f)
            sims = tmp['sims']
            diffs_synth_from_real = tmp['diffs_synth_from_real']
            diffs_real_from_synth = tmp['diffs_real_from_synth']
    elif os.path.isfile(output_filepath):
        print(f"Loading existing rubrics from {output_filepath}")
        with open(output_filepath, 'rt') as f:
            tmp = json.load(f)
            sims = tmp['sims']
            diffs_synth_from_real = tmp['diffs_synth_from_real']
            diffs_real_from_synth = tmp['diffs_real_from_synth']
    
    # Sanity check
    assert syn_db_dataset_map.keys() == real_dataset_map.keys(), \
        f"syn_db_dataset_map.keys() != real_dataset_map.keys(): {syn_db_dataset_map.keys()} != {real_dataset_map.keys()}"
    
    db_ids = list(syn_db_dataset_map.keys())
    
    # Generate rubrics for each database and dataset
    try:
        for db_id in tqdm.tqdm(db_ids):
            # Initialize database dictionaries if needed
            if db_id not in sims:
                sims[db_id] = {}
            if db_id not in diffs_synth_from_real:
                diffs_synth_from_real[db_id] = {}
            if db_id not in diffs_real_from_synth:
                diffs_real_from_synth[db_id] = {}
            
            for dataset_name, dataset in tqdm.tqdm(list(syn_db_dataset_map[db_id].items())):
                # Generate similarities
                if dataset_name not in sims[db_id]:
                    output = similar_chain.invoke(dict(
                        feedback='similar to',
                        num=cfg.num_points,
                        A=json.dumps(real_dataset_map[db_id]),
                        B=json.dumps(dataset)
                    )).content
                    
                    cleaned_output = fix_double_quotes(output)
                    try:
                        sims[db_id][dataset_name] = json_parser.parse(cleaned_output)
                    except Exception as e:
                        print(f"Error parsing {dataset_name} sims: {e}")
                        sims[db_id][dataset_name] = cleaned_output
                else:
                    print(f'skipping {dataset_name} sims')
                
                # Generate differences: synthetic from real
                if dataset_name not in diffs_synth_from_real[db_id]:
                    similar_points = "\n".join(sims[db_id][dataset_name])
                    output = diff_chain.invoke(dict(
                        feedback='different from',
                        num=cfg.num_points,
                        A=json.dumps(real_dataset_map[db_id]),
                        B=json.dumps(dataset),
                        similar_points=similar_points
                    )).content
                    
                    cleaned_output = fix_double_quotes(output)
                    try:
                        diffs_synth_from_real[db_id][dataset_name] = json_parser.parse(cleaned_output)
                    except Exception as e:
                        print(f"Error parsing {dataset_name} synth from real: {e}")
                        diffs_synth_from_real[db_id][dataset_name] = cleaned_output
                else:
                    print(f'skipping {dataset_name} synth from real')
                
                # Generate differences: real from synthetic
                if dataset_name not in diffs_real_from_synth[db_id]:
                    similar_points = "\n".join(sims[db_id][dataset_name])
                    output = diff_chain.invoke(dict(
                        feedback='different from',
                        num=cfg.num_points,
                        B=json.dumps(real_dataset_map[db_id]),
                        A=json.dumps(dataset),
                        similar_points=similar_points
                    )).content
                    
                    cleaned_output = fix_double_quotes(output)
                    try:
                        diffs_real_from_synth[db_id][dataset_name] = json_parser.parse(cleaned_output)
                    except Exception as e:
                        print(f"Error parsing {dataset_name} real from synth: {e}")
                        diffs_real_from_synth[db_id][dataset_name] = cleaned_output
                else:
                    print(f'skipping {dataset_name} real from synth')
                    
    except (Exception, KeyboardInterrupt) as e:
        print(f"At {db_id}, {dataset_name}: Error: {e}")
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

@hydra.main(version_base=None, config_path="conf", config_name="text2sql.yaml")
def main(cfg: DictConfig) -> None:
    """Main execution function."""
    cfg = cfg.text2sql.rubric_generation
    
    similar_chain, diff_chain, json_parser = load_langchain_model(cfg)
    syn_db_dataset_map = load_synthetic_data(cfg)
    real_dataset_map = load_real_data(cfg)
    
    generate_rubrics(
        syn_db_dataset_map,
        real_dataset_map,
        similar_chain,
        diff_chain,
        json_parser,
        cfg
    )


if __name__ == "__main__":
    main()
