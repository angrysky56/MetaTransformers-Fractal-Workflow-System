"""
Simple test script for quantum poetry generation
"""

import os
import asyncio
from src.quantum_poetry_generator import QuantumPoet

os.environ['OPENAI_API_KEY'] = 'your-key-here'  # Replace with actual key

async def test_poetry():
    poet = QuantumPoet()
    result = await poet.generate_poem("The Nature of Reality", coherence=0.42)
    
    if result["success"]:
        print("\nGenerated Quantum Poem:")
        print("=" * 40)
        print(result["poem"])
        print("\nQuantum Properties:")
        print(f"Coherence: {result['coherence']:.3f}")
        print(f"Dream Stability: {result['dream_stability']:.3f}")
    else:
        print(f"Failed: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(test_poetry())