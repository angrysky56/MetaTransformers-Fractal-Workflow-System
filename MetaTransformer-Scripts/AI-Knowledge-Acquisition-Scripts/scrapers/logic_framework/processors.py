"""
Advanced Logic Processing System
-----------------------------
Core processing module for logical concept extraction and analysis.

Processing Layers:
1. Pattern Recognition
2. Logical Structure Analysis
3. Inference Generation
4. Formal Verification
"""

import re
from typing import Dict, List, Optional
from datetime import datetime
import logging

class LogicProcessor:
    """
    Advanced processor for logical concept extraction and analysis.
    Implements pattern recognition and formal logic processing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize advanced pattern recognition templates."""
        self.patterns = {
            'definitional': [
                r'([A-Z][^.!?]*(?:is defined as|means|refers to|is called|denotes)[^.!?]*\.)',
                r'((?:A|An|The)\s+[^.!?]*(?:is|are|refers to|consists of)[^.!?]*\.)'
            ],
            'formal': [
                r'((?:In|By|Under)\s+[^.!?]*(?:we mean|we understand|is defined as)[^.!?]*\.)',
                r'([A-Z][^\n.!?]*(?:\bis\b|\bare\b)[^.!?]*(?:iff|if and only if)[^.!?]*\.)'
            ],
            'mathematical': [
                r'((?:Let|Given|For)\s+[^.!?]*(?:be|denote|represent)[^.!?]*\.)',
                r'([A-Z][^\n.!?]*(?:axiom|theorem|lemma|proposition)[^.!?]*\.)'
            ],
            'inference': [
                r'((?:If|When|Whenever)\s+[^,]+,\s+then[^.!?]*\.)',
                r'([^.!?]*implies[^.!?]*\.)'
            ],
            'modal': [
                r'((?:Necessarily|Possibly)\s+[^.!?]*\.)',
                r'([^.!?]*(?:modal|modality|modalities)[^.!?]*\.)'
            ],
            'temporal': [
                r'([^.!?]*(?:always|eventually|until|next)[^.!?]*\.)',
                r'([^.!?]*(?:temporal|time|sequence)[^.!?]*\.)'
            ]
        }

    def extract_logical_properties(self, text: str) -> Dict[str, bool]:
        """
        Extract logical properties from text using advanced pattern matching.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict of logical properties
        """
        return {
            'is_axiom': bool(re.search(r'axiom', text, re.I)),
            'is_theorem': bool(re.search(r'theorem', re.I)),
            'is_definition': bool(re.search(r'is defined as|means|refers to', text, re.I)),
            'has_condition': bool(re.search(r'if|when|whenever|where', text, re.I)),
            'has_equivalence': bool(re.search(r'iff|if and only if|equivalent', text, re.I)),
            'is_modal': bool(re.search(r'necessarily|possibly|modal', text, re.I)),
            'is_temporal': bool(re.search(r'always|eventually|until|next', text, re.I)),
            'is_first_order': bool(re.search(r'first.?order|predicate', text, re.I)),
            'is_higher_order': bool(re.search(r'higher.?order|second.?order', text, re.I))
        }

    def process_logical_content(self, content: Dict) -> List[Dict]:
        """
        Process content to extract logical concepts and relationships.
        
        Args:
            content: Dictionary containing content sections
            
        Returns:
            List of extracted logical concepts
        """
        concepts = []
        
        for section in content.get('sections', []):
            section_text = ' '.join(section['content'])
            
            # Process each pattern category
            for category, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, section_text)
                    for match in matches:
                        # Extract and clean term
                        term_match = re.search(r'^([^,.]+)', match)
                        if term_match:
                            term = term_match.group(1).strip()
                            term = re.sub(r'^(A|An|The|In|By|Under|Let|Given|For)\s+', 
                                        '', term, flags=re.IGNORECASE)
                            
                            # Extract logical properties
                            properties = self.extract_logical_properties(match)
                            
                            concept = {
                                'term': term,
                                'definition': match.strip(),
                                'context': section['heading'],
                                'source': content['title'],
                                'url': content['url'],
                                'category': category,
                                'properties': properties,
                                'extracted_at': datetime.now().isoformat()
                            }
                            concepts.append(concept)
        
        return concepts

    def analyze_logical_structure(self, concept: Dict) -> Dict:
        """
        Analyze logical structure and relationships of a concept.
        
        Args:
            concept: Extracted logical concept
            
        Returns:
            Enhanced concept with structural analysis
        """
        # Add structural analysis
        structure = {
            'complexity_level': self._determine_complexity(concept),
            'dependency_chain': self._extract_dependencies(concept),
            'logical_form': self._determine_logical_form(concept),
            'quantification': self._analyze_quantification(concept)
        }
        
        concept['structural_analysis'] = structure
        return concept

    def _determine_complexity(self, concept: Dict) -> str:
        """Determine logical complexity level."""
        text = concept['definition']
        if re.search(r'if and only if|iff', text, re.I):
            return 'high'
        elif re.search(r'if|implies|when', text, re.I):
            return 'medium'
        return 'basic'

    def _extract_dependencies(self, concept: Dict) -> List[str]:
        """Extract logical dependencies."""
        deps = []
        text = concept['definition']
        
        # Look for referenced concepts
        references = re.findall(r'(?:requires|depends on|uses|assumes)\s+([^,.]+)', text)
        deps.extend([ref.strip() for ref in references])
        
        return deps

    def _determine_logical_form(self, concept: Dict) -> str:
        """Determine the logical form of the concept."""
        text = concept['definition'].lower()
        
        if re.search(r'for all|∀|universal', text):
            return 'universal'
        elif re.search(r'exists|∃|existential', text):
            return 'existential'
        elif re.search(r'if|→|implies', text):
            return 'conditional'
        elif re.search(r'and|∧|conjunction', text):
            return 'conjunction'
        elif re.search(r'or|∨|disjunction', text):
            return 'disjunction'
        
        return 'atomic'

    def _analyze_quantification(self, concept: Dict) -> Dict:
        """Analyze quantification structure."""
        text = concept['definition']
        
        return {
            'has_universal': bool(re.search(r'for all|∀|universal', text, re.I)),
            'has_existential': bool(re.search(r'exists|∃|existential', text, re.I)),
            'has_unique': bool(re.search(r'unique|exactly one', text, re.I)),
            'quantifier_order': self._determine_quantifier_order(text)
        }

    def _determine_quantifier_order(self, text: str) -> List[str]:
        """Determine the order of quantifiers."""
        quantifiers = []
        text = text.lower()
        
        # Extract quantifiers in order
        pattern = r'(for all|exists|∀|∃)'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            if match.group(1) in ['for all', '∀']:
                quantifiers.append('universal')
            else:
                quantifiers.append('existential')
                
        return quantifiers