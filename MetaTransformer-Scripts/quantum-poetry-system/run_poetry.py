"""
Poetry Generation Runner
Properly configured runner for the quantum poetry system
"""

import os
import sys
from pathlib import Path
import asyncio
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root.parent))

# Import our poetry generator
from src.quantum_poetry_generator import QuantumPoet, save_poem

async def main():
    try:
        poet = QuantumPoet()
        
        print("Quantum Poetry Generator")
        print("=" * 40)
        theme = input("\nEnter a theme for your quantum poem: ")
        result = await poet.generate_poem(theme)
        
        if result["success"]:
            print("\nGenerated Quantum Poem:")
            print("=" * 40)
            print(result["poem"])
            print("\nQuantum Properties:")
            print(f"Coherence: {result['coherence']:.3f}")
            print(f"Dream Stability: {result['dream_stability']:.3f}")
            
            # Save the poem
            save_poem(result, theme)
        else:
            print(f"Failed to generate poem: {result.get('error')}")
            
    except Exception as e:
        print(f"Error running poetry system: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())