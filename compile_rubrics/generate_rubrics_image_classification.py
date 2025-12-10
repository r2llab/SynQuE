import base64
import io
import json
import random
from pathlib import Path
from string import Template
from typing import List

import hydra
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from omegaconf import DictConfig
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import auto as tqdm


# ============================================================================
# Data Models
# ============================================================================

class Similarity(BaseModel):
    """Model for similarity characteristics."""
    similar_characteristics: List[str] = Field(
        description="List of similar characteristics (e.g., Both datasets ...)"
    )


class Difference(BaseModel):
    """Model for difference characteristics."""
    different_characteristics: List[str] = Field(
        description="List of different characteristics (e.g., dataset B is ... dataset A is ...)."
    )


# ============================================================================
# Helper Functions
# ============================================================================

def assert_class_balance(dataset_file_map, drop_failed=False):
    """
    Check if all datasets have balanced classes.
    
    Args:
        dataset_file_map: Dictionary mapping dataset names to class file maps
        drop_failed: If True, drop imbalanced datasets instead of raising an error
    """
    print("\nChecking class balance:")
    imbalanced_datasets = []
    
    for dataset_name, class_files in list(dataset_file_map.items()):
        # Get counts for each class
        class_counts = {class_name: len(files) for class_name, files in class_files.items()}
        
        # Check if all classes have the same number of images
        counts = list(class_counts.values())
        is_balanced = len(set(counts)) <= 1  # If all counts are the same, set will have length 1
        
        if not is_balanced:
            error_msg = f"Dataset '{dataset_name}' has imbalanced classes: {class_counts}"
            print(f"❌ ERROR: {error_msg}")
            
            if drop_failed:
                print(f"   Dropping dataset '{dataset_name}' due to class imbalance")
                imbalanced_datasets.append(dataset_name)
            else:
                raise ValueError(error_msg)
        else:
            print(f"✅ Dataset '{dataset_name}' is balanced with {counts[0]} images per class")
    
    # Remove imbalanced datasets if drop_failed is True
    for dataset_name in imbalanced_datasets:
        del dataset_file_map[dataset_name]
    
    if imbalanced_datasets and drop_failed:
        print(f"Dropped {len(imbalanced_datasets)} imbalanced datasets: {imbalanced_datasets}")


def load_datasets(cfg):
    """
    Load real and synthetic datasets.
    
    Args:
        cfg: Configuration object containing data paths and settings
    
    Returns:
        tuple: (real_data, dataset_file_map) where dataset_file_map is 
               dataset -> class -> file_name
    """
    print("Processing Real data")
    real_data = pd.read_csv(cfg.real_data_df_path)
    real_data['image_path'] = real_data['image_path'].apply(
        lambda x: str(Path(cfg.real_data_root) / x)
    )
    
    print("Processing Synthetic data")
    dataset_file_map = {}  # dataset -> class -> file_name
    
    # Process each dataset specified in the config
    for dataset in cfg.synthetic_data:
        dataset_name = dataset.dataset_name
        data_path = dataset.data_root_path
        
        if not Path(data_path).exists():
            raise ValueError(f"Data path {data_path} does not exist")
        
        print(f"- Processing dataset: {dataset_name}")
        jpg_paths = list(Path(data_path).glob('**/*.jpg'))
        png_paths = list(Path(data_path).glob('**/*.png'))
        data_paths = jpg_paths + png_paths
        
        if dataset_name not in dataset_file_map:
            dataset_file_map[dataset_name] = {}
        
        for path in data_paths:
            # Extract class name from path
            # Assuming class is the second-to-last directory in the path
            class_name = path.parts[-2]
            
            if class_name not in dataset_file_map[dataset_name]:
                dataset_file_map[dataset_name][class_name] = []
            
            dataset_file_map[dataset_name][class_name].append(str(path))
    
    total_images = sum(
        len(files) for dataset in dataset_file_map.values() for files in dataset.values()
    )
    print(f"Found {total_images} images in {len(dataset_file_map)} datasets")
    
    # Assert class balance for each dataset
    assert_class_balance(dataset_file_map, drop_failed=cfg.drop_imbalanced_datasets)
    
    return real_data, dataset_file_map


def get_n_balanced_samples(df, n, seed=42):
    """
    Sample n rows from the dataframe, balanced across all classes.
    
    Args:
        df: DataFrame with 'label' column
        n: Total number of samples to return
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame: Balanced sample of the input dataframe
    """
    labels = sorted(df['label'].unique())
    samples_per_class = n // len(labels)
    print(f"Sampling {samples_per_class} samples per class")
    
    balanced_samples = []
    for label in labels:
        class_samples = df[df['label'] == label].sample(
            min(samples_per_class, sum(df['label'] == label)), 
            random_state=seed
        )
        balanced_samples.append(class_samples)
    
    result = pd.concat(balanced_samples)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def encode_image(image_path, max_short_side_size=768, max_long_side_size=1024):
    """
    Encode image to base64 string with optional resizing.
    
    Args:
        image_path: Path to the image file
        max_short_side_size: Maximum size for the shorter side
        max_long_side_size: Maximum size for the longer side
    
    Returns:
        str: Base64 encoded image string
    """
    img = Image.open(image_path)
    pixel_size = img.size[0] * img.size[1]
    
    if pixel_size > max_short_side_size * max_long_side_size:
        img = img.resize((max_short_side_size, max_long_side_size))
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def construct_prompt_files(
    dataset_A_paths, 
    dataset_B_paths, 
    prompt, 
    max_short_side_size=768, 
    max_long_side_size=1024
):
    """
    Construct prompt with encoded images for API calls.
    
    Args:
        dataset_A_paths: List of image paths for dataset A
        dataset_B_paths: List of image paths for dataset B
        prompt: Text prompt to include
        max_short_side_size: Maximum size for the shorter side of images
        max_long_side_size: Maximum size for the longer side of images
    
    Returns:
        list: Messages formatted for OpenAI API
    """
    whole_content_img_B = []
    for img_path in dataset_B_paths:
        b64_img = encode_image(img_path, max_short_side_size, max_long_side_size)
        whole_content_img_B.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_img}"},
        })
    
    whole_content_img_A = []
    for img_path in dataset_A_paths:
        b64_img = encode_image(img_path, max_short_side_size, max_long_side_size)
        whole_content_img_A.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_img}"},
        })
    
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": f"Below are {len(dataset_B_paths)} images from dataset B"}] +
                whole_content_img_B +
                [{"type": "text", "text": f"\n\nBelow are {len(dataset_A_paths)} images from dataset A"}] +
                whole_content_img_A +
                [{"type": "text", "text": prompt}]
            ),
        },
    ]
    return messages


def load_langchain_model(cfg):
    """
    Load OpenAI model and prompt templates.
    
    Args:
        cfg: Configuration object containing model and service settings
    
    Returns:
        tuple: (client, similar_system_prompt, different_system_prompt)
    """
    with open('./secrets.json') as f:
        secrets = json.load(f)
    
    client = OpenAI(
        api_key=secrets[cfg.service]['api_key'],
        base_url=secrets[cfg.service]['api_endpoint']
    )
    
    sim_prompt_path = Path('./prompt_templates/image_classification/rubric_compilation/sim.txt')
    diff_prompt_path = Path('./prompt_templates/image_classification/rubric_compilation/diff.txt')
    
    with sim_prompt_path.open('r') as f:
        similar_system_prompt = Template(f.read())
    
    with diff_prompt_path.open('r') as f:
        different_system_prompt = Template(f.read())
    
    return client, similar_system_prompt, different_system_prompt


def get_price(response, tokens_price_input_1m, tokens_price_output_1m):
    """
    Calculate the price of the API call based on token usage.
    
    Args:
        response: The response from the API call
        tokens_price_input_1m: Price per million input tokens
        tokens_price_output_1m: Price per million output tokens
    
    Returns:
        float: The estimated price of the API call
    """
    input_text_tokens = response.usage.prompt_tokens
    input_tokens = input_text_tokens
    output_tokens = response.usage.completion_tokens
    
    # Calculate price (convert from per million tokens to per token)
    price_per_token_input = tokens_price_input_1m / 1_000_000
    price_per_token_output = tokens_price_output_1m / 1_000_000
    total_price = (input_tokens * price_per_token_input + output_tokens * price_per_token_output)
    
    return total_price


# ============================================================================
# Rubric Generation
# ============================================================================

def generate_rubrics(
    real_data, 
    dataset_file_map, 
    client, 
    similar_system_prompt, 
    different_system_prompt, 
    cfg, 
    sim_parser, 
    diff_parser
):
    """
    Generate rubrics for image classification datasets.
    
    Args:
        real_data: DataFrame containing real data
        dataset_file_map: Dictionary mapping dataset names to class file maps
        client: OpenAI client instance
        similar_system_prompt: Template for similarity prompts
        different_system_prompt: Template for difference prompts
        cfg: Configuration object
        sim_parser: Parser for similarity responses
        diff_parser: Parser for difference responses
    """
    # Determine output filename
    if cfg.split is not None:
        output_filename = (
            f'rubric.image_classification.'
            f'{cfg.model.name.replace("/", "--")}.{cfg.data_type}.'
            f'generate_{cfg.num_points}points_num_examples_{cfg.num_examples}.'
            f'{cfg.split}.real_data_seed={cfg.seed}.json'
        )
    else:
        output_filename = (
            f'rubric.image_classification.'
            f'{cfg.model.name.replace("/", "--")}.{cfg.data_type}.'
            f'generate_{cfg.num_points}points_num_examples_{cfg.num_examples}.'
            f'real_data_seed={cfg.seed}.json'
        )
    
    output_path = Path(cfg.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    output_filepath = output_path / output_filename
    partial_filepath = output_filepath.with_suffix('.partial.json')
    
    # Initialize rubric dictionaries
    sims = {}  # dataset -> sims
    diffs_synth_from_real = {}  # dataset -> diffs
    diffs_real_from_synth = {}  # dataset -> diffs
    
    # Load existing rubrics if available
    if output_filepath.exists() or partial_filepath.exists():
        if output_filepath.exists():
            print(f"Loading existing rubrics from {output_filepath}...")
            with open(output_filepath, 'rt') as f:
                rubrics = json.load(f)
            sims = rubrics['sims']
            diffs_synth_from_real = rubrics['diffs_synth_from_real']
            diffs_real_from_synth = rubrics['diffs_real_from_synth']
        else:
            print(f"Loading existing rubrics from {partial_filepath}...")
            with open(partial_filepath, 'rt') as f:
                rubrics = json.load(f)
            sims = rubrics['sims']
            diffs_synth_from_real = rubrics['diffs_synth_from_real']
            diffs_real_from_synth = rubrics['diffs_real_from_synth']
    else:
        print(f"No rubrics at {output_filepath} or {partial_filepath} found, starting from scratch")
    
    # Generate rubrics for each dataset
    try:
        for dataset_name in tqdm(dataset_file_map.keys()):
            acc_price = 0
            
            # Initialize dataset dictionaries if needed
            if dataset_name not in sims:
                sims[dataset_name] = []
            if dataset_name not in diffs_synth_from_real:
                diffs_synth_from_real[dataset_name] = []
            if dataset_name not in diffs_real_from_synth:
                diffs_real_from_synth[dataset_name] = []
            
            print(f"Generating rubrics for {dataset_name}...")
            
            # Sample balanced data
            # real_data_sampled = get_n_balanced_samples(real_data, cfg.num_examples, cfg.seed)
            real_data_sampled = real_data  # We've already balanced and sampled the data
            
            # Sample synthetic data
            synthetic_files = []
            synthetic_labels = []
            for class_name, files in dataset_file_map[dataset_name].items():
                num_samples = cfg.num_examples // len(dataset_file_map[dataset_name])
                # We set the seed fixed for synthetic data
                random.seed(42)
                sampled_files = random.sample(files, min(num_samples, len(files)))
                synthetic_files.extend(sampled_files)
                synthetic_labels.extend([class_name] * len(sampled_files))
            
            synthetic_data = pd.DataFrame({
                'image_path': synthetic_files,
                'label': synthetic_labels
            })
            
            # Generate similarities
            if len(sims[dataset_name]) == 0:
                similar_messages = construct_prompt_files(
                    synthetic_data['image_path'].tolist(),
                    real_data_sampled['image_path'].tolist(),
                    similar_system_prompt.substitute(
                        num_points=cfg.num_points, 
                        format_instructions=sim_parser.get_format_instructions()
                    ),
                    cfg.max_short_side_size,
                    cfg.max_long_side_size
                )
                
                similar_response = client.chat.completions.create(
                    model=cfg.model.name,
                    messages=similar_messages,
                    response_format={"type": "json_object"},
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens
                )
                
                acc_price += get_price(
                    similar_response, 
                    cfg.model.tokens_price_input_1m, 
                    cfg.model.tokens_price_output_1m
                )
                sim_output = similar_response.choices[0].message.content
                
                try:
                    similar_characteristics = sim_parser.parse(sim_output)
                except Exception as e:
                    print(f"Error parsing similarity: {e}")
                    print(f"Response: {similar_response}")
                    print(f"Output: {sim_output}")
                    similar_characteristics = sim_output
                
                sims[dataset_name] = similar_characteristics
            
            # Generate differences: synthetic from real
            if len(diffs_synth_from_real[dataset_name]) == 0:
                different_messages = construct_prompt_files(
                    real_data_sampled['image_path'].tolist(),  # dataset A
                    synthetic_data['image_path'].tolist(),  # dataset B
                    different_system_prompt.substitute(
                        num_points=cfg.num_points,
                        similar_characteristics=json.dumps(sims[dataset_name]),
                        format_instructions=diff_parser.get_format_instructions()
                    ),
                    cfg.max_short_side_size,
                    cfg.max_long_side_size
                )
                
                different_response = client.chat.completions.create(
                    model=cfg.model.name,
                    messages=different_messages,
                    response_format={"type": "json_object"},
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens
                )
                
                acc_price += get_price(
                    different_response, 
                    cfg.model.tokens_price_input_1m, 
                    cfg.model.tokens_price_output_1m
                )
                
                diff_output = different_response.choices[0].message.content
                try:
                    different_characteristics = diff_parser.parse(diff_output)
                except Exception as e:
                    print(f"Error parsing difference: {e}")
                    print(f"Response: {different_response}")
                    print(f"Output: {diff_output}")
                    different_characteristics = diff_output
                
                diffs_synth_from_real[dataset_name] = different_characteristics
            
            # Generate differences: real from synthetic
            if len(diffs_real_from_synth[dataset_name]) == 0:
                different_messages = construct_prompt_files(
                    synthetic_data['image_path'].tolist(),  # dataset A
                    real_data_sampled['image_path'].tolist(),  # dataset B
                    different_system_prompt.substitute(
                        num_points=cfg.num_points,
                        similar_characteristics=json.dumps(sims[dataset_name]),
                        format_instructions=diff_parser.get_format_instructions()
                    ),
                    cfg.max_short_side_size,
                    cfg.max_long_side_size
                )
                
                different_response = client.chat.completions.create(
                    model=cfg.model.name,
                    messages=different_messages,
                    response_format={"type": "json_object"},
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens
                )
                
                acc_price += get_price(
                    different_response, 
                    cfg.model.tokens_price_input_1m, 
                    cfg.model.tokens_price_output_1m
                )
                
                diff_output = different_response.choices[0].message.content
                try:
                    different_characteristics = diff_parser.parse(diff_output)
                except Exception as e:
                    print(f"Error parsing difference: {e}")
                    print(f"Response: {different_response}")
                    print(f"Output: {diff_output}")
                    different_characteristics = diff_output
                
                diffs_real_from_synth[dataset_name] = different_characteristics
            
            # print(f"Total price: ${acc_price:.2f}")  # TODO: fix this as it only considers text tokens
            
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

@hydra.main(version_base=None, config_path="conf", config_name="image.yaml")
def main(cfg: DictConfig) -> None:
    """Main execution function."""
    cfg = cfg.image_classification.rubric_generation
    sim_parser = JsonOutputParser()
    diff_parser = JsonOutputParser()
    
    # print(OmegaConf.to_yaml(cfg))
    client, similar_system_prompt, different_system_prompt = load_langchain_model(cfg)
    real_data, dataset_file_map = load_datasets(cfg)
    
    generate_rubrics(
        real_data, 
        dataset_file_map, 
        client, 
        similar_system_prompt, 
        different_system_prompt, 
        cfg, 
        sim_parser, 
        diff_parser
    )


if __name__ == "__main__":
    main()
