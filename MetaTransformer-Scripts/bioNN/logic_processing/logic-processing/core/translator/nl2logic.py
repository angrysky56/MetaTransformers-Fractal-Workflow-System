"""
Natural Language to Logic Translator
Handles the translation of natural language into logical formulations
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import yaml

class NL2LogicTranslator:
    """Translates natural language to logical formulations using Logic-LLM approach"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load translator configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            return full_config['logic_processing']['translator']
    
    def initialize(self) -> bool:
        """Initialize the translator model"""
        try:
            # We'll use GPT-4 for translation through the OpenAI API
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize translator: {str(e)}")
            return False
    
    async def translate(self, text: str) -> Dict[str, Union[str, float]]:
        """Translate natural language to logical formulation"""
        try:
            import openai
            from openai import OpenAI
            
            client = OpenAI()
            
            # Construct prompt for logical translation
            prompt = f"""Translate the following natural language text into a logical formulation using first-order logic:

Text: {text}

Convert this into a logical form that can be processed by a symbolic solver. Use the following guidelines:
1. Use standard logical operators (∀, ∃, ∧, ∨, ¬, →, ↔)
2. Define predicates clearly
3. Specify domain constraints
4. Handle quantifiers appropriately

Logical formulation:"""

            response = await client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": "You are a precise logical translator that converts natural language into formal logic."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens']
            )
            
            logical_form = response.choices[0].message.content
            
            return {
                "success": True,
                "logical_form": logical_form,
                "confidence": 1.0 - response.choices[0].finish_reason == "length"
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "logical_form": None,
                "confidence": 0.0
            }
    
    def validate_translation(self, logical_form: str) -> bool:
        """Validate the syntactic correctness of the logical formulation"""
        try:
            # Basic syntax validation
            required_elements = ['∀', '∃', '∧', '∨', '¬', '→', '↔']
            contains_operators = any(op in logical_form for op in required_elements)
            balanced_parentheses = logical_form.count('(') == logical_form.count(')')
            
            return contains_operators and balanced_parentheses
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False
    
    def get_predicates(self, logical_form: str) -> List[str]:
        """Extract predicates from the logical formulation"""
        import re
        try:
            # Extract predicate patterns (capital letter followed by parameters in parentheses)
            predicate_pattern = r'[A-Z][a-z]*\([^)]*\)'
            predicates = re.findall(predicate_pattern, logical_form)
            return list(set(predicates))
        except Exception as e:
            logger.error(f"Failed to extract predicates: {str(e)}")
            return []

if __name__ == "__main__":
    # Test the translator
    translator = NL2LogicTranslator()
    if translator.initialize():
        test_text = "All humans are mortal. Socrates is human. Therefore, Socrates is mortal."
        import asyncio
        result = asyncio.run(translator.translate(test_text))
        print(f"Translation result: {result}")
    else:
        print("Failed to initialize translator")