from typing import List, Dict, Any, Optional
from .vllm_pipeline import VLLMPipeline
import json
from ..data_models import ScoreModel

class WebAgentScoringPipeline(VLLMPipeline):
    def __init__(
        self,
        domain_name: str,
        rubric_key: str,
        scoring_model_path: str,
        source_data_path: str,
        prompt_template_path: str,
        rubric_path: str,
        sample_set_size: int = 1,
        resume_path: Optional[str] = None,
        max_context_window: int = 1536,
        max_generate_tokens: int = 64,
        temperature: float = 0.00,
        top_p: float = 0.95,
        **kwargs
    ):
        """Initialize the AnswerRubricPipeline
        
        Args:
            domain_name (str): Name of the domain to filter the data by
            generation_model_name (str): Name of the generation model
            scoring_model_path (str): Path to the judgement model
            source_data_path (str): Path to the original data
            content_type (str): Type of content included in rubric generation
            rubric_path (str): Path to the R1 rubric
            sample_set_size (int, optional): Number of samples to include in the prompt. Defaults to 1.
            resume_path (Optional[str], optional): Path to resume from previous run. Defaults to None.
            max_context_window (int, optional): Maximum number of tokens in the context window. Defaults to 1536.
            max_generate_tokens (int, optional): Maximum number of tokens to generate. Defaults to 64.
            temperature (float, optional): Temperature for the generation model. Defaults to 0.00.
            top_p (float, optional): Top-p for the generation model. Defaults to 0.95.
        """
        self.sample_set_size = sample_set_size
        self.domain_name = domain_name
        self.resume_examples = []
        self.rubric = json.load(open(rubric_path))
        self.rubric_key = rubric_key
        self.scoring_model_path = scoring_model_path

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
        """Filter out examples that have already been processed if resuming"""
        loaded_data = json.load(open(aux_data_path))
        loaded_data = list(loaded_data.values())
        if self.resume_examples:
            return self.resume_examples
        
        # we divide the data into sample_set_size chunks
        loaded_data = [loaded_data[i:i+self.sample_set_size] for i in range(0, len(loaded_data), self.sample_set_size)]
        return loaded_data
    
    def update_data(self, rubric_path: str, rubric_key: str, domain_name: str, source_data_path: str) -> None:
        """Update the auxiliary data so we can reuse the same pipeline for different datasets"""
        self.rubric = json.load(open(rubric_path))
        self.rubric_key = rubric_key
        self.domain_name = domain_name
        self.aux_data = self.preprocess_aux_data(source_data_path)


    def generate_prompts(self, num_examples: Optional[int] = None) -> List[str]:
        """Generate prompts from the original data"""
        website = self.domain_name
        if website not in self.rubric["sims"]:
            raise ValueError(f"Website {website} not found in rubric")
        similar_characteristics_str = "\n".join(self.rubric["sims"][website][self.rubric_key])
        sfr_diffs_str = "\n".join(self.rubric["diffs_synth_from_real"][website][self.rubric_key])
        rfs_diffs_str = "\n".join(self.rubric["diffs_real_from_synth"][website][self.rubric_key])

        if num_examples:
            self.aux_data = self.aux_data[:num_examples]

        # Given synthetic data is from dataset B
        # predict how likely is this data from dataset A
        prompts = [
            self.prompt_template.invoke(dict(example="\n".join(sample_set), 
                                            similar_characteristics=similar_characteristics_str, 
                                            differences=sfr_diffs_str,
                                            prediction="A")).text
            for sample_set in self.aux_data
        ]
        # predict how likely is this data from dataset B
        prompts.extend([
            self.prompt_template.invoke(dict(example="\n".join(sample_set), 
                                            similar_characteristics=similar_characteristics_str, 
                                            differences=sfr_diffs_str,
                                            prediction="B")).text
            for sample_set in self.aux_data
        ])
        # Given synthetic data is from dataset A
        # predict how likely is this data from dataset A
        prompts.extend([
            self.prompt_template.invoke(dict(example="\n".join(sample_set), 
                                            similar_characteristics=similar_characteristics_str, 
                                            differences=rfs_diffs_str,
                                            prediction="A")).text
            for sample_set in self.aux_data
        ])
        # predict how likely is this data from dataset B
        prompts.extend([
            self.prompt_template.invoke(dict(example="\n".join(sample_set), 
                                            similar_characteristics=similar_characteristics_str, 
                                            differences=rfs_diffs_str,
                                            prediction="B")).text
            for sample_set in self.aux_data
        ])


        self.aux_data = [{"idx": i, "mode": "score_real_given_synth_loc_B"} for i in range(len(self.aux_data))] + \
                        [{"idx": i, "mode": "score_synth_given_synth_loc_B"} for i in range(len(self.aux_data), 2 * len(self.aux_data))] + \
                        [{"idx": i, "mode": "score_synth_given_synth_loc_A"} for i in range(2 * len(self.aux_data), 3 * len(self.aux_data))] + \
                        [{"idx": i, "mode": "score_real_given_synth_loc_A"} for i in range(3 * len(self.aux_data), 4 * len(self.aux_data))]
        return prompts

    def postprocess_each_output(self, parsed_output: Dict[str, Any], input_data: Dict[str, Any], prob: float=None) -> List[Dict[str, Any]]:
        """Add the original tweet and label to the parsed output"""
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
        # we have a list of results that has double the number of tweets
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