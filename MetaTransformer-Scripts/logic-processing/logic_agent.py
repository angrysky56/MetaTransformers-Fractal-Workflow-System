"""
Logic Agent Integration
Provides logical reasoning capabilities to agents using Logic-LLM
"""

import os
import json
from pathlib import Path
import openai
from loguru import logger
import sys

# Add Logic-LLM to path
LOGIC_LLM_PATH = Path("F:/Logic-LLM")
sys.path.append(str(LOGIC_LLM_PATH))

from models.logic_program import LogicProgramGenerator
from models.logic_inference import LogicInference

class LogicAgent:
    """Provides logical reasoning capabilities to agents"""
    
    def __init__(self, openai_api_key: str = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self._initialize_logic_llm()
    
    def _initialize_logic_llm(self):
        """Initialize Logic-LLM components"""
        # Create minimal args object expected by Logic-LLM
        from types import SimpleNamespace
        self.args = SimpleNamespace(
            api_key=self.api_key,
            model_name="gpt-4",
            data_path=str(LOGIC_LLM_PATH / "data"),
            save_path=str(LOGIC_LLM_PATH / "outputs/logic_programs"),
            stop_words="------",
            max_new_tokens=1024
        )
        
        self.generator = LogicProgramGenerator(self.args)
        
    async def reason(self, context: str, question: str) -> dict:
        """Apply logical reasoning to a question"""
        try:
            # Prepare input in Logic-LLM format
            input_data = {
                "id": "runtime_query",
                "context": context,
                "question": question,
                "answer": None,  # Will be determined
                "options": []    # For multiple choice
            }
            
            # Generate logical program
            prompt = self.generator.prompt_folio(input_data)
            logic_program = await self._generate_logic_program(prompt)
            
            # Run inference
            result = await self._run_inference(logic_program)
            
            return {
                "success": True,
                "reasoning": logic_program,
                "conclusion": result.get("answer"),
                "confidence": result.get("confidence", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Reasoning failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_logic_program(self, prompt: str) -> str:
        """Generate logical program using GPT-4"""
        try:
            completion = await openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a logical reasoning system that converts natural language into formal logic programs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Logic program generation failed: {str(e)}")
            raise
    
    async def _run_inference(self, logic_program: str) -> dict:
        """Run logical inference on the generated program"""
        try:
            # Initialize Logic-LLM inference
            inferencer = LogicInference()
            
            # Run inference
            result = inferencer.run_inference(logic_program)
            
            # Process result
            if result.get("success"):
                return {
                    "answer": result.get("conclusion"),
                    "confidence": result.get("confidence", 1.0)
                }
            else:
                raise ValueError(f"Inference failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise

    async def validate_reasoning(self, context: str, reasoning: str) -> dict:
        """Validate a piece of logical reasoning"""
        try:
            # Create validation prompt
            validation_prompt = f"""
            Validate the following logical reasoning:
            
            Context: {context}
            Reasoning: {reasoning}
            
            Check for:
            1. Logical consistency
            2. Valid inferences
            3. Sound conclusions
            
            Provide a detailed analysis."""
            
            # Use GPT-4 for validation
            validation = await openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a logical reasoning validator that checks for consistency and soundness."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            analysis = validation.choices[0].message.content
            
            # Extract key metrics (this is a simple heuristic)
            valid_markers = ["valid", "correct", "consistent", "sound"]
            confidence = sum(1 for marker in valid_markers if marker in analysis.lower()) / len(valid_markers)
            
            return {
                "valid": confidence > 0.7,
                "confidence": confidence,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                "valid": False,
                "confidence": 0.0,
                "error": str(e)
            }

if __name__ == "__main__":
    # Test the logic agent
    import asyncio
    
    async def test_agent():
        agent = LogicAgent()
        
        # Test context and question
        context = """
        All humans are mortal.
        Socrates is human.
        """
        question = "Is Socrates mortal?"
        
        result = await agent.reason(context, question)
        print(f"Reasoning result: {json.dumps(result, indent=2)}")
        
        if result["success"]:
            validation = await agent.validate_reasoning(
                context, 
                result["reasoning"]
            )
            print(f"Validation result: {json.dumps(validation, indent=2)}")
    
    asyncio.run(test_agent())