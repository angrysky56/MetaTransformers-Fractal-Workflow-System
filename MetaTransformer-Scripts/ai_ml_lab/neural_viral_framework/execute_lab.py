from lab_experiment import run_experiment
import cupy as cp

def main():
    try:
        print("\n=== Quantum Viral Learning Laboratory ===")
        print("Initializing GPU-accelerated experimentation...")
        
        # Execute experiment
        analysis = run_experiment()
        
        # Record results in Neo4j
        update_experiment_status(analysis)
        
        return analysis
        
    except Exception as e:
        print(f"\nExperiment Error: {str(e)}")
        return None

if __name__ == "__main__":
    main()