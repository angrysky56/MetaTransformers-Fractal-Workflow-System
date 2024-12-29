"""
Validation Protocol Manager
Manages validation workflows and protocols
"""

from typing import Dict, List, Optional
from loguru import logger
import yaml
from pathlib import Path

class ValidationProtocol:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.stages = self.config['validation']['validation_stages']
        self._initialize_protocols()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_protocols(self):
        """Initialize validation protocols"""
        self.protocols = {
            'syntax': self._validate_syntax,
            'semantics': self._validate_semantics,
            'coherence': self._validate_coherence,
            'completeness': self._validate_completeness
        }
    
    async def validate(self, logical_form: str, context: Dict) -> Dict:
        """Run validation protocol"""
        try:
            results = {}
            passed = True
            
            for stage in self.stages:
                if stage not in self.protocols:
                    logger.warning(f"Unknown validation stage: {stage}")
                    continue
                    
                stage_result = await self.protocols[stage](logical_form, context)
                results[stage] = stage_result
                passed = passed and stage_result['passed']
            
            return {
                "success": True,
                "passed": passed,
                "results": results
            }
        except Exception as e:
            logger.error(f"Validation protocol failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _validate_syntax(self, logical_form: str, context: Dict) -> Dict:
        """Validate syntax"""
        try:
            # Basic syntax validation
            balanced = logical_form.count('(') == logical_form.count(')')
            valid_operators = all(op in '∀∃∧∨¬→↔' for op in logical_form if op not in '() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
            return {
                "passed": balanced and valid_operators,
                "score": float(balanced and valid_operators)
            }
        except Exception as e:
            logger.error(f"Syntax validation failed: {str(e)}")
            return {"passed": False, "score": 0.0}
    
    async def _validate_semantics(self, logical_form: str, context: Dict) -> Dict:
        """Validate semantics"""
        try:
            # Check predicate and variable usage
            predicates = set(c for c in logical_form if c.isupper())
            variables = set(c for c in logical_form if c.islower())
            
            valid_predicates = len(predicates) > 0
            valid_variables = len(variables) > 0
            
            # Check predicates against context
            if 'predicates' in context:
                valid_predicates = all(p in context['predicates'] for p in predicates)
            
            return {
                "passed": valid_predicates and valid_variables,
                "score": 0.5 * float(valid_predicates) + 0.5 * float(valid_variables)
            }
        except Exception as e:
            logger.error(f"Semantic validation failed: {str(e)}")
            return {"passed": False, "score": 0.0}
    
    async def _validate_coherence(self, logical_form: str, context: Dict) -> Dict:
        """Validate logical coherence"""
        try:
            # Check logical structure
            has_quantifiers = any(q in logical_form for q in ['∀', '∃'])
            has_implications = '→' in logical_form
            proper_structure = has_quantifiers and has_implications
            
            # Check quantifier-variable relationships
            valid_scoping = self._check_variable_scope(logical_form)
            
            score = 0.6 * float(proper_structure) + 0.4 * float(valid_scoping)
            
            return {
                "passed": score > 0.8,
                "score": score
            }
        except Exception as e:
            logger.error(f"Coherence validation failed: {str(e)}")
            return {"passed": False, "score": 0.0}
    
    async def _validate_completeness(self, logical_form: str, context: Dict) -> Dict:
        """Validate logical completeness"""
        try:
            # Check for required elements
            required_elements = {
                'quantifiers': any(q in logical_form for q in ['∀', '∃']),
                'implications': '→' in logical_form,
                'predicates': any(c.isupper() for c in logical_form if c.isalpha()),
                'variables': any(c.islower() for c in logical_form if c.isalpha())
            }
            
            # Check completeness against context
            if 'required_predicates' in context:
                required_elements['context_predicates'] = all(
                    p in logical_form for p in context['required_predicates']
                )
            
            score = sum(float(v) for v in required_elements.values()) / len(required_elements)
            
            return {
                "passed": score > 0.9,
                "score": score,
                "missing_elements": [k for k, v in required_elements.items() if not v]
            }
        except Exception as e:
            logger.error(f"Completeness validation failed: {str(e)}")
            return {"passed": False, "score": 0.0}
    
    def _check_variable_scope(self, logical_form: str) -> bool:
        """Check proper variable scoping in quantifiers"""
        try:
            # Find all quantifier-variable pairs
            import re
            quantifier_pairs = re.findall(r'[∀∃]([a-z])', logical_form)
            if not quantifier_pairs:
                return False
            
            # Check that variables are properly used within scope
            variables = set(v for v in quantifier_pairs)
            for var in variables:
                # Find quantifier position
                q_pos = logical_form.find(f'∀{var}' if f'∀{var}' in logical_form else f'∃{var}')
                if q_pos == -1:
                    return False
                    
                # Find variable usages
                var_positions = [i for i, c in enumerate(logical_form) if c == var]
                if not all(pos > q_pos for pos in var_positions):
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Variable scope check failed: {str(e)}")
            return False