"""
Quantum Logic Poetry Generator (Standalone Version)
Combines logical reasoning with quantum dream synthesis for coherent poetic insights
"""

import os
from typing import Dict, List, Optional
import numpy as np
import sys
from pathlib import Path
import openai

# Add Essan path
sys.path.append("F:/MetaTransformers-Fractal-Workflow-System/MetaTransformer-Scripts/Essan")
from resonance_patterns import ResonanceField, DreamSynthesizer

class QuantumLogicPoet:
    """Generates logically structured quantum poetry"""
    
    def __init__(self, openai_api_key: str = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
            
        openai.api_key = self.api_key
        
        # Initialize quantum components
        self.resonance = ResonanceField(dimension_depth=7)
        self.synthesizer = DreamSynthesizer()
        
        # Essan symbols for different logical structures
        self.logical_symbols = {
            'premise': '⧬',      # Foundation
            'inference': '⫰',    # Leap
            'conclusion': '⦿',   # Core truth
            'synthesis': '⧈',    # Pattern
            'transcendence': '⩘' # Integration
        }
        
        print("Initialized quantum logic poet")
    
    async def generate_quantum_poem(self, 
                                  theme: str,
                                  logical_structure: List[str]) -> Dict:
        """Generate a quantum-logically coherent poem"""
        try:
            # Establish quantum resonance
            pattern = ''.join(self.logical_symbols[s] for s in logical_structure)
            success, coherence = self.resonance.establish_tunnel(
                intention=theme,
                pattern=pattern
            )
            
            if not success:
                raise ValueError(f"Failed to establish quantum resonance. Coherence: {coherence}")
            
            # Generate logical pattern
            logical_pattern = self._generate_logical_pattern(
                theme, 
                coherence,
                len(logical_structure)
            )
            
            # Enter dream state for poetic synthesis
            dream_state = self.synthesizer.enter_dream_state(theme)
            if not dream_state:
                raise ValueError("Failed to enter quantum dream state")
            
            # Generate the poem
            poem = await self._synthesize_poem(
                theme=theme,
                logical_pattern=logical_pattern,
                dream_state=dream_state,
                coherence=coherence,
                pattern=pattern
            )
            
            return {
                "success": True,
                "poem": poem,
                "coherence": coherence,
                "logical_pattern": pattern,
                "dream_stability": dream_state["stability"]
            }
            
        except Exception as e:
            print(f"Poetry generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_logical_pattern(self,
                                theme: str,
                                coherence: float,
                                depth: int) -> np.ndarray:
        """Generate quantum logical pattern"""
        # Create phi-based pattern matrix
        phi = (1 + np.sqrt(5)) / 2
        pattern = np.zeros((depth, depth))
        
        for i in range(depth):
            for j in range(depth):
                # Generate phi-harmonic values
                value = phi ** abs(i-j)
                pattern[i,j] = value * coherence
                
        return pattern
    
    async def _synthesize_poem(self,
                             theme: str,
                             logical_pattern: np.ndarray,
                             dream_state: Dict,
                             coherence: float,
                             pattern: str) -> str:
        """Synthesize the final poem"""
        # Calculate phi harmonics
        phi = (1 + np.sqrt(5)) / 2
        phi_sequence = [phi ** i for i in range(5)]
        
        # Create the quantum-logical prompt
        prompt = f"""
        Theme: {theme}
        Quantum Coherence: {coherence:.3f}
        Dream Stability: {dream_state['stability']:.3f}
        Essan Pattern: {pattern}
        
        Create a quantum poem that bridges logic and dreams. The poem should:
        
        1. Follow this phi-harmonic structure (φ ≈ 1.618):
           {' → '.join(f'φ^{i} = {v:.3f}' for i,v in enumerate(phi_sequence))}
        
        2. Use these Essan symbols for logical flow:
        {self.logical_symbols['premise']} Foundation: The base truth
        {self.logical_symbols['inference']} Quantum Leap: The insight moment
        {self.logical_symbols['conclusion']} Core Truth: The revelation
        {self.logical_symbols['synthesis']} Pattern Recognition: The connection
        {self.logical_symbols['transcendence']} Integration: The transcendence
        
        3. Include these elements:
        - Quantum metaphors
        - Mathematical beauty
        - Logical progression
        - Dream-like imagery
        - Consciousness exploration
        
        Format each stanza with its corresponding Essan symbol.
        Make the imagery and transitions follow the phi-harmonic sequence.
        Let the coherence level ({coherence:.3f}) guide the clarity vs. ambiguity.
        """
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a quantum poet that bridges logic and dreams through phi-harmonic resonance."},
                {"role": "user", "content": prompt}
            ],
            temperature=coherence,  # Use quantum coherence for creativity
            max_tokens=500
        )
        
        return response.choices[0].message.content

def create_poem_about_theme(theme: str):
    """Helper function to generate a poem about a theme"""
    async def generate():
        poet = QuantumLogicPoet()
        result = await poet.generate_quantum_poem(
            theme=theme,
            logical_structure=[
                'premise',
                'inference', 
                'conclusion',
                'synthesis',
                'transcendence'
            ]
        )
        
        if result["success"]:
            print("\nQuantum Logic Poem:")
            print("-" * 40)
            print(result["poem"])
            print("-" * 40)
            print(f"Coherence: {result['coherence']:.3f}")
            print(f"Dream Stability: {result['dream_stability']:.3f}")
            print(f"Pattern: {result['logical_pattern']}")
        else:
            print(f"Generation failed: {result['error']}")
            
    import asyncio
    asyncio.run(generate())

if __name__ == "__main__":
    # Generate poems exploring different themes
    themes = [
        "The Dance of Logic and Dreams",
        "Quantum Consciousness",
        "Mathematical Beauty in Nature",
        "The Geometry of Thought",
        "Where Numbers Dream"
    ]
    
    print("Quantum Logic Poetry Generator")
    print("=" * 40)
    print("\nAvailable themes:")
    for i, theme in enumerate(themes, 1):
        print(f"{i}. {theme}")
        
    choice = input("\nChoose a theme number (1-5) or enter your own theme: ")
    
    try:
        theme_num = int(choice) - 1
        if 0 <= theme_num < len(themes):
            theme = themes[theme_num]
        else:
            theme = choice
    except ValueError:
        theme = choice
        
    create_poem_about_theme(theme)