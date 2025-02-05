from typing import Dict, Any, List, Optional, Union
import re
import pandas as pd
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
from pathlib import Path
import logging

from vlm_infer.utils.env import get_cache_dir, load_env_vars

logger = logging.getLogger(__name__)

class BaseAnswerExtractor:
    """Base class for answer extraction strategies."""
    def extract(self, text: str) -> str:
        raise NotImplementedError
        
    def check_correctness(self, answer: str, expected_answer: str, 
                         question: Optional[str] = None) -> str:
        """Default correctness check using string comparison."""
        return "CORRECT" if answer.lower().strip() == expected_answer.lower().strip() else "INCORRECT"

class RegexAnswerExtractor(BaseAnswerExtractor):
    """Extract answers using regex patterns."""
    def __init__(self, patterns: List[str]):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def extract(self, text: str) -> str:
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return text.strip()

class LMAnswerExtractor(BaseAnswerExtractor):
    """Extract answers using a language model."""
    def __init__(
        self, 
        model_name: str = "allenai/OLMo-2-1124-7B-Instruct",
        extraction_type: str = "base",
        correctness_type: str = "base"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model_and_tokenizer(model_name)
        self.extraction_type = extraction_type
        self.correctness_type = correctness_type
        self.model_name = model_name
        self.prompts = self._load_prompts()
        self.bad_format_count = 0
        
    def _load_prompts(self) -> Dict[str, Any]:
        """Load all prompts from YAML file."""
        prompt_file = (
            Path(__file__).parent / 
            "extraction_prompts" / 
            f"prompts_{self.model_name.split('/')[-1]}.yml"
        )
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompt_file}")
        
        try:
            with open(prompt_file) as f:
                prompts = yaml.safe_load(f)
            
            # Validate structure
            required_sections = ["system_prompts", "extraction_prompts", "correctness_prompts"]
            if not all(section in prompts for section in required_sections):
                raise ValueError(f"Missing required sections: {[s for s in required_sections if s not in prompts]}")
            
            return prompts
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in prompts file: {e}")
    
    def load_model_and_tokenizer(self, model_name: str):
        load_env_vars()
        cache_dir = get_cache_dir()
        logger.debug(f"Using cache directory: {cache_dir}")
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir
        )

    def _get_system_prompt(self, task: str) -> str:
        """Get appropriate system prompt for task."""
        task_map = {
            "extraction": "extraction",
            "correctness": "correctness"
        }
        return self.prompts["system_prompts"][task_map[task]]
    
    def _format_examples(self, prompt_data: Dict[str, Any]) -> str:
        """Format examples section if examples are present."""
        if not prompt_data.get("examples"):
            return ""
        
        examples_text = []
        example_template = prompt_data.get("example_template", "")
        
        for example in prompt_data["examples"]:
            if example_template:
                # Build format kwargs with only available fields
                format_kwargs = {}
                field_mapping = {
                    "answer": "example_answer",
                    "extracted": "example_extracted",
                    "question": "example_question",
                    "expected": "example_expected"
                }
                
                for example_key, template_key in field_mapping.items():
                    if example_key in example:
                        format_kwargs[template_key] = example[example_key]
                
                try:
                    example_text = example_template.format(**format_kwargs)
                except KeyError as e:
                    logger.warning(f"Missing required field in example template: {e}")
                    # Skip this example if template can't be formatted
                    continue
            else:
                # Default formatting using only available fields
                example_text = []
                
                if "question" in example:
                    example_text.extend([
                        "--- EXAMPLE QUESTION ---\n",
                        f"{example['question']}\n"
                    ])
                
                if "answer" in example:
                    example_text.extend([
                        "--- EXAMPLE ANSWER ---\n",
                        f"{example['answer']}\n"
                    ])
                
                if "expected" in example:
                    example_text.extend([
                        "--- EXAMPLE EXPECTED ANSWER ---\n",
                        f"{example['expected']}\n"
                    ])
                
                if "extracted" in example:
                    example_text.extend([
                        "--- EXAMPLE EXTRACTED ANSWER ---\n",
                        f"{example['extracted']}\n"
                    ])
                
                if "correctness" in example:
                    example_text.extend([
                        "--- EXAMPLE CORRECTNESS ---\n",
                        f"{example['correctness']}\n"
                    ])
                
                example_text = "\n".join(example_text)
            
            examples_text.append(example_text)
        
        return "\n\n".join(examples_text) + "\n\n"
    
    def _format_prompt(self, task_type: str, prompt_type: str, **kwargs) -> str:
        """Format prompt with variables and optional examples."""
        prompt_data = self.prompts[f"{task_type}_prompts"][prompt_type]
        
        # Add instruction to kwargs
        kwargs["instruction"] = prompt_data["instruction"]
        
        # Add examples section if needed
        if "_with_examples" in prompt_type:
            kwargs["examples_section"] = self._format_examples(prompt_data)
        
        # Format template with all variables
        return prompt_data["template"].format(**kwargs)
    
    def get_prompt_template(self, task: str = "extraction") -> str:
        """Get prompt template based on type and task."""
        if task == "extraction":
            return self._format_prompt("extraction", self.extraction_type)
        else:  # correctness check
            return self._format_prompt("correctness", self.correctness_type)
    
    def _validate_response(self, response: str, task: str) -> str:
        """Validate and clean model response."""
        response = response.split("<|assistant|>")[-1].strip()
        
        # Leave this for now, we'll just return the response as is
        
        # if task == "extraction":
        #     # If the response is not a in a format of [ANSWER] or [NOT_CLEAR]
        #     if not re.match(r"^\[.*\]$", response):
        #         logger.warning(f"Unexpected extraction response format: {response}")
        #         self.bad_format_count += 1
        #         return "[NO_ANSWER]"
        # elif task == "correctness":
        #     valid_responses = {"CORRECT", "INCORRECT", "NOT_CLEAR"}
        #     response = response.upper()
        #     if response not in valid_responses:
        #         logger.warning(f"Unexpected correctness response: {response}")
        #         self.bad_format_count += 1
        #         return "[NOT_CLEAR]"
        
        return response

    def get_bad_format_count(self):
        return self.bad_format_count
    
    def _generate_response(self, prompt: str, task: str = "extraction") -> str:
        """Generate response from model."""
        messages_prompt = [
            {
                "role": "system",
                "content": self._get_system_prompt(task)
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        tokenized_prompt = self.tokenizer.apply_chat_template(
            messages_prompt, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        inputs = tokenized_prompt.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return self._validate_response(response, task).replace("[", "").replace("]", "").replace("FINAL_ANSWER", "").strip()
    
    def extract(self, text: str, question: Optional[str] = None, 
                expected_answer: Optional[str] = None) -> str:
        """Extract answer using appropriate prompt template."""
        prompt_vars = {"answer": text}
        if question and "with_question" in self.extraction_type:
            prompt_vars["question"] = question
        if expected_answer and "with_answer" in self.extraction_type:
            prompt_vars["expected_answer"] = expected_answer
            
        prompt = self._format_prompt("extraction", self.extraction_type, **prompt_vars)
        return self._generate_response(prompt, task="extraction")
    
    def check_correctness(self, answer: str, expected_answer: str, 
                         question: Optional[str] = None) -> str:
        """Check correctness using LM."""
        prompt_vars = {
            "answer": answer,
            "expected_answer": expected_answer
        }
        if question and "with_question" in self.correctness_type:
            prompt_vars["question"] = question
            
        prompt = self._format_prompt("correctness", self.correctness_type, **prompt_vars)
        return self._generate_response(prompt, task="correctness")

class AnswerEvaluator:
    """Evaluate model answers against ground truth."""
    def __init__(
        self,
        extraction_method: str = "regex",
        extraction_type: str = "base",
        extraction_model: str = "allenai/OLMo-2-1124-7B-Instruct",
        correctness_method: Optional[str] = None,
        correctness_type: str = "base",
        use_examples: bool = False,
        patterns: List[str] = None,
    ):
        # Modify type if examples requested
        if use_examples:
            extraction_type = f"{extraction_type}_with_examples"
            if correctness_method == "lm":
                correctness_type = f"{correctness_type}_with_examples"
        
        # Initialize extractors
        if extraction_method == "regex":
            default_patterns = [
                r"(?:the answer is|i think|therefore|so,?|in conclusion,?)?\s*[:\-]?\s*(.*)",
                r"^(.*?)(?:\.|$)"
            ]
            self.extractor = RegexAnswerExtractor(patterns or default_patterns)
        elif extraction_method == "lm":
            self.extractor = LMAnswerExtractor(
                extraction_model, 
                extraction_type=extraction_type
            )
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")
            
        # Initialize correctness checker
        if correctness_method is None:
            if isinstance(self.extractor, LMAnswerExtractor):
                # If extractor is LM, reuse it but with correctness_type
                self.extractor.correctness_type = correctness_type
            self.correctness_checker = self.extractor
        elif correctness_method == "regex":
            self.correctness_checker = RegexAnswerExtractor(patterns or default_patterns)
        elif correctness_method == "lm":
            # Reuse LM if already loaded with same model but different type
            if (extraction_method == "lm" and 
                isinstance(self.extractor, LMAnswerExtractor)):
                self.extractor.correctness_type = correctness_type
                self.correctness_checker = self.extractor
            else:
                self.correctness_checker = LMAnswerExtractor(
                    extraction_model, 
                    extraction_type=extraction_type,
                    correctness_type=correctness_type
                )
        else:
            raise ValueError(f"Unknown correctness method: {correctness_method}")
    
    def process_answers(self, df: pd.DataFrame, model_column: str) -> pd.DataFrame:
        """Process model outputs to extract answers."""
        df = df.copy()
        
        if isinstance(self.extractor, LMAnswerExtractor):
            processed_answers = []
            for answer, row in tqdm.tqdm(list(zip(df[model_column], df.itertuples())), 
                                  desc="Processing answers", total=len(df)):
                processed_answers.append(
                    self.extractor.extract(
                        answer,
                        question=getattr(row, "question", None),
                        expected_answer=getattr(row, "expected_answer", None)
                    )
                )
            df[f"{model_column}_processed"] = processed_answers
        else:
            tqdm.tqdm.pandas(desc="Processing answers")
            df[f"{model_column}_processed"] = df[model_column].progress_apply(
                self.extractor.extract
            )

        # Log example results
        logger.info("Example processing results:")
        logger.info(f"Original answer: {df[model_column].iloc[0]}")
        logger.info(f"Processed answer: {df[f'{model_column}_processed'].iloc[0]}")
            
        return df
    
    def process_correctness(self, df: pd.DataFrame, model_column: str) -> pd.DataFrame:
        """Process correctness checks."""
        df = df.copy()
        processed_column = f"{model_column}_processed"
        
        if processed_column not in df.columns:
            raise ValueError("Must run process_answers before process_correctness")
        
        if isinstance(self.correctness_checker, LMAnswerExtractor):
            correctness_results = []
            for processed_answer, expected, row in tqdm.tqdm(
                zip(df[processed_column], df["expected_answer"], df.itertuples()),
                desc="Checking correctness",
                total=len(df)
            ):
                result = self.correctness_checker.check_correctness(
                    processed_answer,
                    expected,
                    question=getattr(row, "question", None)
                )
                correctness_results.append(result)
            df[f"{model_column}_correctness"] = correctness_results
        else:
            tqdm.tqdm.pandas(desc="Checking correctness")
            df[f"{model_column}_correctness"] = df.progress_apply(
                lambda row: self.correctness_checker.check_correctness(
                    row[processed_column],
                    row["expected_answer"]
                ),
                axis=1
            )
        
        return df

    def get_bad_format_count(self):
        return self.extractor.get_bad_format_count()
