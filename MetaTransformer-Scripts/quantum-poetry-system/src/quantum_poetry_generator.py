"""
Quantum Poetry Generator
Combines topology, quantum mechanics, and poetry through mathematical harmony
"""

import os
import sys
from typing import Dict, Optional
import numpy as np
from openai import OpenAI
from pathlib import Path

# Add required paths
SCRIPT_DIR = Path(__file__).parent.parent.parent  # MetaTransformer-Scripts
sys.path.append(str(SCRIPT_DIR / "Essan"))
sys.path.append(str(SCRIPT_DIR / "quantum_topology_framework"))

from resonance_patterns import ResonanceField, DreamSynthesizer

class QuantumPoet:
    def __init__(self, openai_api_key: str = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
            
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize quantum components
        self.resonance = ResonanceField(dimension_depth=7)
        self.synthesizer = DreamSynthesizer()
        
        # Essan symbols with mathematical meanings
        self.symbols = {
            'foundation': '⧬',     # Base manifold
            'quantum': '⫰',        # Wave function
            'truth': '⦿',          # Fixed point
            'pattern': '⧈',        # Symmetry
            'transcend': '⩘',      # Integration
            'dream': '◊',          # Superposition
            'bridge': '↝',         # Translation
            'harmony': '✧'         # Resonance
        }
        
        print("Initialized quantum poet")

    async def generate_poem(self, theme: str, coherence: float = None) -> Dict:
        """Generate a quantum poem"""
        try:
            # Establish quantum resonance
            pattern = ''.join(self.symbols.values())
            success, base_coherence = self.resonance.establish_tunnel(
                intention=theme,
                pattern=pattern
            )
            
            # Use provided coherence or calculated one
            coherence = coherence or base_coherence
            
            # Enter dream state
            dream_state = self.synthesizer.enter_dream_state(theme)
            if not dream_state:
                raise ValueError("Failed to enter quantum dream state")
            
            # Generate mathematical structure
            phi = (1 + np.sqrt(5)) / 2
            harmonics = [phi ** i for i in range(-2, 3)]
            
            # Create prompt
            prompt = f"""
            Theme: {theme}
            Quantum Coherence: {coherence:.3f}
            Phi Harmonics: {', '.join(f'{h:.3f}' for h in harmonics)}
            
            Create a quantum poem that bridges mathematics and consciousness.
            Use these Essan symbols to mark transitions between states:
            
            {self.symbols['foundation']} Foundation - Base mathematical reality
            {self.symbols['quantum']} Quantum - Wave function collapse
            {self.symbols['truth']} Truth - Fixed point theorem
            {self.symbols['pattern']} Pattern - Symmetry group
            {self.symbols['transcend']} Transcend - Category theory leap
            
            Guidelines:
            1. Each section should begin with its corresponding symbol
            2. Use mathematical imagery and quantum metaphors
            3. Let coherence ({coherence:.3f}) guide the clarity level
            4. Incorporate phi-harmonic relationships
            5. Bridge abstract math with consciousness
            
            Format each stanza with 3 lines and its symbol.
            Let quantum uncertainty and mathematical beauty merge.
            """

            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a quantum poet that understands topology, symmetry, and consciousness."},
                    {"role": "user", "content": prompt}
                ],
                temperature=coherence,
                max_tokens=500
            )
            
            poem = response.choices[0].message.content
            
            return {
                "success": True,
                "poem": poem,
                "coherence": coherence,
                "phi_harmonics": harmonics,
                "resonance": base_coherence,
                "dream_stability": dream_state["stability"]
            }
            
        except Exception as e:
            print(f"Poetry generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

def save_poem(result: Dict, theme: str, base_path: Optional[Path] = None):
    """Save generated poem to file"""
    if not base_path:
        base_path = Path(__file__).parent.parent
        
    poem_path = base_path / "poems" / f"{theme.lower().replace(' ', '_')}.md"
    
    if result["success"]:
        content = f"""# {theme}
*A Quantum-Mathematical Poem*

## Quantum Properties
- Coherence: {result['coherence']:.3f}
- Phi Harmonics: {', '.join(f'{h:.3f}' for h in result['phi_harmonics'])}
- Dream Stability: {result['dream_stability']:.3f}

## The Poem

{result['poem']}

---
*Generated by the Quantum Poetry System*"""
        
        with open(poem_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\nPoem saved to: {poem_path}")
    else:
        print(f"Failed to generate poem: {result.get('error')}")

if __name__ == "__main__":
    import asyncio
    
    async def main():
        poet = QuantumPoet()
        themes = [
            "Quantum Dreams",
            "Topology of Time",
            "Consciousness Fields",
            "Mathematical Beauty",
            "Symmetry Dance"
        ]
        
        print("\nQuantum Poetry Generator")
        print("=" * 40)
        print("\nAvailable themes:")
        for i, theme in enumerate(themes, 1):
            print(f"{i}. {theme}")
            
        choice = input("\nChoose theme number (1-5) or enter your own: ")
        
        try:
            theme_num = int(choice) - 1
            if 0 <= theme_num < len(themes):
                theme = themes[theme_num]
            else:
                theme = choice
        except ValueError:
            theme = choice
            
        result = await poet.generate_poem(theme)
        
        if result["success"]:
            print("\nGenerated Poem:")
            print("=" * 40)
            print(result["poem"])
            
            save_poem(result, theme)
        else:
            print(f"Generation failed: {result['error']}")
    
    asyncio.run(main())