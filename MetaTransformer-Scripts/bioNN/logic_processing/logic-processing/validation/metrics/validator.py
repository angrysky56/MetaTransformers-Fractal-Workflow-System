"""
Logic Validation Metrics
Defines validation metrics for logic processing
"""

from typing import Dict, List, Optional
from loguru import logger
import yaml

class LogicValidator:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.thresholds = self.config['validation']
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path is None:
            config_path = "config.yaml"
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config
    
    def validate_logic(self, logical_form: str) -> Dict:
        """Validate logical formulation"""
        try:
            # Run validation checks
            syntax_score = self._check_syntax(logical_form)
            semantics_score = self._check_semantics(logical_form)
            completeness_score = self._check_completeness(logical_form)
            
            # Calculate overall score
            total_score = (
                0.4 * syntax_score +
                0.3 * semantics_score +
                0.3 * completeness_score
            )
            
            passed = total_score >= self.thresholds['accuracy_threshold']
            
            return {
                "success": True,
                "valid": passed,
                "total_score": total_score,
                "metrics": {
                    "syntax": syntax_score,
                    "semantics": semantics_score,
                    "completeness": completeness_score
                }
            }
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _check_syntax(self, logical_form: str) -> float:
        """Check syntactic correctness"""
        try:
            # Basic syntax checks
            balanced_parens = logical_form.count('(') == logical_form.count(')')
            valid_operators = all(op in logical_form for op in ['∀', '∃', '→'])
            proper_spacing = not '  ' in logical_form
            
            score = (
                0.4 * float(balanced_parens) +
                0.4 * float(valid_operators) +
                0.2 * float(proper_spacing)
            )
            
            return score
        except Exception:
            return 0.0
    
    def _check_semantics(self, logical_form: str) -> float:
        """Check semantic validity"""
        try:
            # Semantic checks
            has_quantifiers = any(q in logical_form for q in ['∀', '∃'])
            has_predicates = any(c.isupper() for c in logical_form if c.isalpha())
            has_variables = any(c.islower() for c in logical_form if c.isalpha())
            
            score = (
                0.4 * float(has_quantifiers) +
                0.3 * float(has_predicates) +
                0.3 * float(has_variables)
            )
            
            return score
        except Exception:
            return 0.0
    
    def _check_completeness(self, logical_form: str) -> float:
        """Check logical completeness"""
        try:
            # Completeness checks
            has_implications = '→' in logical_form
            has_conclusion = logical_form.count('→') >= 1
            proper_scope = self._check_scope(logical_form)
            
            score = (
                0.3 * float(has_implications) +
                0.3 * float(has_conclusion) +
                0.4 * float(proper_scope)
            )
            
            return score
        except Exception:
            return 0.0
    
    def _check_scope(self, logical_form: str) -> bool:
        """Check proper variable scoping"""
        try:
            # Basic scope validation
            quantifier_pos = [i for i, c in enumerate(logical_form) if c in ['∀', '∃']]
            variable_pos = [i for i, c in enumerate(logical_form) if c.islower()]
            
            if not quantifier_pos or not variable_pos:
                return False
                
            # Check if variables appear after their quantifiers
            for var_pos in variable_pos:
                if not any(q_pos < var_pos for q_pos in quantifier_pos):
                    return False
                    
            return True
        except Exception:
            return False