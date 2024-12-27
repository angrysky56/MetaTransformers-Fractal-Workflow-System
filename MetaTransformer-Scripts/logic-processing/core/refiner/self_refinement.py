"""
Self-Refinement Module
Handles error-guided revision of logical formulations
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import openai
from loguru import logger
import yaml
import z3
import os

class SelfRefinement:
    """Self-refinement module for improving logical formulations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.improvement_threshold = self.config.get('improvement_threshold', 0.1)
        self.max_rounds = self.config.get('max_rounds', 3)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load refiner configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            return full_config['logic_processing']['refiner']
    
    def initialize(self) -> bool:
        """Initialize the refiner"""
        try:
            # Check OpenAI API key
            if not os.getenv("OPENAI_API_KEY"):
                logger.error("OPENAI_API_KEY environment variable not set")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to initialize refiner: {str(e)}")
            return False

    async def refine(
        self, 
        logical_form: str, 
        error_message: str,
        context: str,
        previous_attempts: List[str] = None
    ) -> Dict[str, Union[str, float]]:
        """Refine logical formulation based on error feedback"""
        try:
            if previous_attempts is None:
                previous_attempts = []
                
            # Create history string
            history = "\n".join([
                f"Attempt {i+1}: {attempt}"
                for i, attempt in enumerate(previous_attempts)
            ])
            
            # Create prompt with focused error analysis
            prompt = f"""Given a logical formulation and its error, provide a refined version.

Context: {context}

Current Formulation:
{logical_form}

Error Message:
{error_message}

Previous Attempts:
{history}

Required Improvements:
1. Fix syntax errors
2. Ensure proper quantifier scoping
3. Validate predicate consistency
4. Maintain domain constraints
5. Simplify where possible without losing meaning

Please provide a corrected logical formulation that resolves the error."""

            client = openai.OpenAI()
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a precise logical refinement system that improves formal logic expressions based on error feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            refined_form = response.choices[0].message.content.strip()
            
            # Calculate improvement metrics
            improvement = self._calculate_improvement(
                original=logical_form,
                refined=refined_form,
                error_msg=error_message
            )
            
            return {
                "success": True,
                "refined_form": refined_form,
                "improvement_score": improvement,
                "significant_improvement": improvement > self.improvement_threshold,
                "error_addressed": self._check_error_addressed(error_message, refined_form)
            }
            
        except Exception as e:
            logger.error(f"Refinement failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "refined_form": None,
                "improvement_score": 0.0,
                "error_addressed": False
            }

    def _calculate_improvement(self, original: str, refined: str, error_msg: str) -> float:
        """Calculate improvement score between original and refined versions"""
        try:
            # Extract error components
            error_terms = self._extract_error_terms(error_msg)
            
            # Calculate component scores
            length_score = self._calculate_length_score(original, refined)
            error_reduction = self._calculate_error_reduction(original, refined, error_terms)
            structure_score = self._evaluate_structure(refined)
            simplicity_score = self._calculate_simplicity_score(original, refined)
            
            # Weighted combination
            weights = {
                'length': 0.15,
                'error_reduction': 0.40,
                'structure': 0.25,
                'simplicity': 0.20
            }
            
            total_score = (
                weights['length'] * length_score +
                weights['error_reduction'] * error_reduction +
                weights['structure'] * structure_score +
                weights['simplicity'] * simplicity_score
            )
            
            return max(0.0, min(1.0, total_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate improvement: {str(e)}")
            return 0.0

    def _calculate_length_score(self, original: str, refined: str) -> float:
        """Calculate length-based improvement score"""
        try:
            # Prefer refined versions that aren't too much longer than original
            length_ratio = len(refined) / len(original)
            if length_ratio > 2.0:
                return 0.0
            elif length_ratio > 1.5:
                return 0.5
            elif 0.8 <= length_ratio <= 1.2:
                return 1.0
            else:
                return 0.7
        except Exception:
            return 0.0

    def _calculate_error_reduction(
        self, 
        original: str, 
        refined: str, 
        error_terms: List[str]
    ) -> float:
        """Calculate error reduction score"""
        try:
            original_errors = sum(1 for term in error_terms if term.lower() in original.lower())
            refined_errors = sum(1 for term in error_terms if term.lower() in refined.lower())
            
            if original_errors == 0:
                return 1.0
            
            reduction = (original_errors - refined_errors) / original_errors
            return max(0.0, min(1.0, reduction))
        except Exception:
            return 0.0

    def _calculate_simplicity_score(self, original: str, refined: str) -> float:
        """Calculate simplicity score based on logical structure"""
        try:
            # Count logical operators
            operators = ['∀', '∃', '∧', '∨', '¬', '→', '↔']
            original_ops = sum(original.count(op) for op in operators)
            refined_ops = sum(refined.count(op) for op in operators)
            
            # Prefer similar or slightly reduced operator count
            if refined_ops > original_ops * 1.5:
                return 0.0
            elif refined_ops > original_ops * 1.2:
                return 0.5
            elif refined_ops <= original_ops:
                return 1.0
            else:
                return 0.8
        except Exception:
            return 0.0

    def _extract_error_terms(self, error_msg: str) -> List[str]:
        """Extract key error terms from error message"""
        import re
        
        # Common error patterns in logical formulations
        patterns = {
            'undefined': r'undefined\s+(\w+)',
            'type_error': r'type\s+error[:\s]+(\w+)',
            'invalid': r'invalid\s+(\w+)',
            'missing': r'missing\s+(\w+)',
            'unknown': r'unknown\s+(\w+)',
            'scope': r'scope\s+error[:\s]+(\w+)',
            'predicate': r'predicate\s+(\w+)',
            'quantifier': r'quantifier\s+(\w+)'
        }
        
        error_terms = []
        for category, pattern in patterns.items():
            matches = re.findall(pattern, error_msg, re.IGNORECASE)
            error_terms.extend(matches)
        
        return list(set(error_terms))

    def _evaluate_structure(self, logical_form: str) -> float:
        """Evaluate structural correctness of logical formulation"""
        scores = []
        
        # Check quantifiers
        has_quantifiers = any(q in logical_form for q in ['∀', '∃', 'ForAll', 'Exists'])
        scores.append(0.3 if has_quantifiers else 0.0)
        
        # Check logical operators
        has_operators = any(op in logical_form for op in ['∧', '∨', '¬', '→', '↔'])
        scores.append(0.2 if has_operators else 0.0)
        
        # Check predicates (capitalized terms)
        has_predicates = any(c.isupper() for c in logical_form if c.isalpha())
        scores.append(0.2 if has_predicates else 0.0)
        
        # Check parentheses balance
        balanced = logical_form.count('(') == logical_form.count(')')
        scores.append(0.3 if balanced else 0.0)
        
        return sum(scores)

    def _check_error_addressed(self, error_msg: str, refined_form: str) -> bool:
        """Check if the specific error has been addressed in refined form"""
        error_terms = self._extract_error_terms(error_msg)
        return not any(term.lower() in refined_form.lower() for term in error_terms)

    async def iterative_refinement(
        self,
        initial_form: str,
        context: str,
        error_message: str,
        max_iterations: Optional[int] = None
    ) -> Dict[str, Union[str, List[str], float]]:
        """Perform iterative refinement until success or max iterations reached"""
        if max_iterations is None:
            max_iterations = self.max_rounds
            
        current_form = initial_form
        refinement_history = []
        best_score = 0.0
        best_form = initial_form
        
        for iteration in range(max_iterations):
            logger.info(f"Starting refinement iteration {iteration + 1}")
            
            result = await self.refine(
                logical_form=current_form,
                error_message=error_message,
                context=context,
                previous_attempts=refinement_history
            )
            
            if not result['success']:
                logger.warning(f"Refinement failed at iteration {iteration + 1}")
                break
                
            refinement_history.append(result['refined_form'])
            current_form = result['refined_form']
            
            # Update best result if improvement is significant
            if result['improvement_score'] > best_score:
                best_score = result['improvement_score']
                best_form = current_form
                
            # Check if we've achieved sufficient improvement
            if result['error_addressed'] and result['improvement_score'] > self.improvement_threshold:
                logger.info("Sufficient improvement achieved")
                break
                
        return {
            "success": True,
            "final_form": best_form,
            "improvement_score": best_score,
            "refinement_history": refinement_history,
            "iterations_used": len(refinement_history)
        }

if __name__ == "__main__":
    # Test the refiner
    import asyncio
    
    async def test_refiner():
        refiner = SelfRefinement()
        if refiner.initialize():
            test_form = "∀x(Human(x) -> Mortal(x)"  # Missing closing parenthesis
            test_error = "Syntax error: missing closing parenthesis"
            test_context = "All humans are mortal"
            
            result = await refiner.iterative_refinement(
                initial_form=test_form,
                error_message=test_error,
                context=test_context
            )
            
            print(f"Refinement result: {result}")
        else:
            print("Failed to initialize refiner")
    
    asyncio.run(test_refiner())