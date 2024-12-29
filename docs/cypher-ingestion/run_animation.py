from quantum_animator import FractalAnimator
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Initialize animator with environment variables
animator = FractalAnimator(
    uri=os.getenv('NEO4J_URI'),
    user=os.getenv('NEO4J_USER'),
    password=os.getenv('NEO4J_PASSWORD')
)

def main():
    try:
        # Generate initial fractal pattern
        print("Generating fractal pattern...")
        pattern = animator.generate_fractal_pattern(depth=5)
        
        if pattern:
            print("\nInitial fractal pattern generated:")
            print(json.dumps(pattern, indent=2))
            
            # Create animation sequence
            print("\nCreating animation sequence...")
            sequence = animator.create_animation_sequence("ANIM-001", frames=30)
            
            print("\nAnimation sequence initialized:")
            print(json.dumps(sequence[:5], indent=2))
            
            return True
        else:
            print("Failed to generate fractal pattern")
            return False
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False
    finally:
        # Cleanup
        animator.cleanup_animation("ANIM-001")

if __name__ == "__main__":
    main()