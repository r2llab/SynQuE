from typing import List, Dict, Any, Optional

from .vllm_pipeline import VLLMPipeline
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import json
from ..data_models import ScoreModel

class ImageClassificationScoringPipeline(VLLMPipeline):
    def __init__(
        self,
        dataset: str,
        scoring_model_path: str,
        source_data_path: str,
        prompt_template_path: str,
        rubric_path: str,
        image_key: str = "image_path",
        sample_set_size: int = 1,
        resume_path: Optional[str] = None,
        max_context_window: int = 1536,
        max_generate_tokens: int = 64,
        temperature: float = 0.00,
        top_p: float = 0.95,
        **kwargs
    ):
        """Initialize the AnswerRubricImageClassificationPipeline
        
        Args:
            generation_model_name (str): Dataset name
            scoring_model_path (str): Path to the judgement model
            source_data_path (str): Path to the original data
            rubric_path (str): Path to the R1 rubric
            image_key (str, optional): Key for image path in data. Defaults to "image_path".
            sample_set_size (int, optional): Number of samples to include in the prompt. Defaults to 1.
            resume_path (Optional[str], optional): Path to resume from previous run. Defaults to None.
            max_context_window (int, optional): Maximum number of tokens in the context window. Defaults to 1536.
            max_generate_tokens (int, optional): Maximum number of tokens to generate. Defaults to 64.
            temperature (float, optional): Temperature for the generation model. Defaults to 0.00.
            top_p (float, optional): Top-p for the generation model. Defaults to 0.95.
        """
        self.image_key = image_key
        self.sample_set_size = sample_set_size
        self.resume_examples = []
        self.rubric = json.load(open(rubric_path))
        self.dataset = dataset
        self.scoring_model_path = scoring_model_path
        self.processor = AutoProcessor.from_pretrained(scoring_model_path, use_fast=True)

        if resume_path:
            with open(resume_path, "r") as f:
                self.resume_examples = json.load(f)
        
        super().__init__(
            model_path=self.scoring_model_path,
            prompt_template_path=prompt_template_path,
            pydantic_model=ScoreModel,
            aux_data_path=source_data_path,
            max_context_window=max_context_window,
            max_generate_tokens=max_generate_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )

    def preprocess_aux_data(self, aux_data_path: str) -> List[Dict[str, Any]]:
        """Filter out examples that have already been processed if resuming and encode images"""
        if aux_data_path.endswith(".json"):
            loaded_data = json.load(open(aux_data_path))
        elif aux_data_path.endswith(".csv"):
            import pandas as pd
            loaded_data = pd.read_csv(aux_data_path).to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported file type: {aux_data_path}")
        
        if self.resume_examples:
            resume_images = set(ex[self.image_key] for ex in self.resume_examples)
            loaded_data = [ex for ex in loaded_data if ex[self.image_key] not in resume_images]
        
        # we divide the data into sample_set_size chunks
        loaded_data = [loaded_data[i:i+self.sample_set_size] for i in range(0, len(loaded_data), self.sample_set_size)]
        return loaded_data
    
    def update_data(self, rubric_path: str, dataset: str, source_data_path: str) -> None:
        """Update the auxiliary data so we can reuse the same pipeline for different datasets"""
        self.rubric = json.load(open(rubric_path))
        self.dataset = dataset
        self.aux_data = self.preprocess_aux_data(source_data_path)

    def _get_texts(self, scoring_prompt: str, all_image_paths: List[str], get_image_inputs: bool = False) -> List[Dict[str, Any]]:
        messages = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": all_image_paths[i]},
                    {"type": "text", "text": scoring_prompt}
                ]
            }]
            for i in range(len(all_image_paths))
        ]
        texts = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if get_image_inputs:
            image_inputs = self._get_image_inputs(messages)
            return texts, image_inputs
        else:
            return texts

    def _get_image_inputs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [process_vision_info(message)[0] for message in messages]

    def _get_prompts(self, texts: List[str], image_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompts = [
            {
                "prompt": texts[i],
                "multi_modal_data": {
                    "image": image_inputs[i]
                }
            }
            for i in range(len(texts))
        ]
        return prompts
    
    def generate_prompts(self, num_examples: Optional[int] = None) -> List[str]:
        """Generate prompts from the original data"""
        similar_characteristics_str = "\n".join(self.rubric["sims"][self.dataset])
        sfr_diffs_str = "\n".join(self.rubric["diffs_synth_from_real"][self.dataset])
        rfs_diffs_str = "\n".join(self.rubric["diffs_real_from_synth"][self.dataset])

        if num_examples:
            self.aux_data = self.aux_data[:num_examples]

        # Given synthetic data is from dataset B
        # predict how likely is this data from dataset A
        prompts = []
        all_image_paths = [sample_set[0]['image_path'] for sample_set in self.aux_data]

        prompt_score_real_given_synth_loc_B = self.prompt_template.invoke(dict(
                    similar_characteristics=similar_characteristics_str,
                    differences=sfr_diffs_str,
                    prediction="A"
                )).text

        texts_score_real_given_synth_loc_B, all_image_inputs = self._get_texts(prompt_score_real_given_synth_loc_B, 
                                                                               all_image_paths, 
                                                                               get_image_inputs=True)
    
        prompts.extend(self._get_prompts(texts_score_real_given_synth_loc_B, all_image_inputs))

        # predict how likely is this data from dataset B
        prompt_score_synth_given_synth_loc_B = self.prompt_template.invoke(dict(
                    similar_characteristics=similar_characteristics_str,
                    differences=sfr_diffs_str,
                    prediction="B"
                )).text

        texts_score_synth_given_synth_loc_B = self._get_texts(prompt_score_synth_given_synth_loc_B, 
                                                                               all_image_paths, 
                                                                               get_image_inputs=False)
    
        prompts.extend(self._get_prompts(texts_score_synth_given_synth_loc_B, all_image_inputs))

        # Given synthetic data is from dataset A
        # predict how likely is this data from dataset A

        prompt_score_synth_given_synth_loc_A = self.prompt_template.invoke(dict(
                    similar_characteristics=similar_characteristics_str,
                    differences=rfs_diffs_str,
                    prediction="A"
                )).text

        texts_score_synth_given_synth_loc_A = self._get_texts(prompt_score_synth_given_synth_loc_A, 
                                                                               all_image_paths, 
                                                                               get_image_inputs=False)
    
        prompts.extend(self._get_prompts(texts_score_synth_given_synth_loc_A, all_image_inputs))

        # predict how likely is this data from dataset B
        prompt_score_real_given_synth_loc_A = self.prompt_template.invoke(dict(
                    similar_characteristics=similar_characteristics_str,
                    differences=rfs_diffs_str,
                    prediction="B"
            )).text
        
        texts_score_real_given_synth_loc_A = self._get_texts(prompt_score_real_given_synth_loc_A, 
                                                                               all_image_paths, 
                                                                               get_image_inputs=False)
    
        prompts.extend(self._get_prompts(texts_score_real_given_synth_loc_A, all_image_inputs))

        self.aux_data = [{"idx": i, "mode": "score_real_given_synth_loc_B"} for i in range(len(self.aux_data))] + \
                        [{"idx": i, "mode": "score_synth_given_synth_loc_B"} for i in range(len(self.aux_data), 2 * len(self.aux_data))] + \
                        [{"idx": i, "mode": "score_synth_given_synth_loc_A"} for i in range(2 * len(self.aux_data), 3 * len(self.aux_data))] + \
                        [{"idx": i, "mode": "score_real_given_synth_loc_A"} for i in range(3 * len(self.aux_data), 4 * len(self.aux_data))]
        return prompts

    def postprocess_each_output(self, parsed_output: Dict[str, Any], input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add the original image and label to the parsed output"""
        if input_data["mode"] == "score_real_given_synth_loc_B":
            parsed_output["score_real_given_synth_loc_B_judgement"] = parsed_output["judgement"]
        elif input_data["mode"] == "score_synth_given_synth_loc_B":
            parsed_output["score_synth_given_synth_loc_B_judgement"] = parsed_output["judgement"]
        elif input_data["mode"] == "score_synth_given_synth_loc_A":
            parsed_output["score_synth_given_synth_loc_A_judgement"] = parsed_output["judgement"]
        elif input_data["mode"] == "score_real_given_synth_loc_A":
            parsed_output["score_real_given_synth_loc_A_judgement"] = parsed_output["judgement"]    
        return parsed_output

    def postprocess_all_results(self, results: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Postprocess all results"""
        # we have a list of results that has double the number of images
        # we need to merge them into a single list
        final_results = {}
        for idx, result in results.items():
            del result["judgement"]
            final_results[idx] = result
        
        output_results = []
        final_length = len(self.aux_data) // 4
        for idx, result in final_results.items():
            if idx < final_length:
                output_results.append({**result, 
                                       **final_results[idx + final_length], 
                                       **final_results[idx + 2 * final_length], 
                                       **final_results[idx + 3 * final_length]})
                
        if len(output_results) != len(self.aux_data) // 4:
            raise ValueError(f"Output results length {len(output_results)} does not match aux data length {len(self.aux_data)}")
        return output_results
                
    def check_output_format(self, parsed_output: Dict[str, Any]) -> bool:
        """Check if all required fields are present and have valid values"""
        if parsed_output == "":
            return False
        valid_values = ["very unlikely", "unlikely", "unsure", "likely", "very likely"]
        
        if parsed_output["judgement"] not in valid_values:
            return False
        return True