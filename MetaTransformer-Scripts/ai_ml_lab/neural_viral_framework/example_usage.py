import cupy as cp
from compute_core import ComputeCore, ComputeConfig

def demonstrate_gpu_computing():
    """
    Demonstrate pure GPU computing with CuPy
    """
    # Initialize compute system
    config = ComputeConfig(enable_profiling=True)
    compute = ComputeCore(config)
    
    # Create test data directly on GPU
    gpu_data = cp.random.randn(1000, 1000)
    
    # FFT Analysis
    print("\n=== FFT Analysis ===")
    fft_results = compute.fft_analysis(gpu_data)
    print(f"FFT Shape: {fft_results['fft'].shape}")
    
    # Matrix Operations
    print("\n=== Matrix Operations ===")
    matrix_b = cp.random.randn(1000, 1000)
    matrix_results = compute.matrix_operations(gpu_data, matrix_b)
    print(f"Matrix Product Shape: {matrix_results['product'].shape}")
    
    # Scientific Computing
    print("\n=== Scientific Computing ===")
    science_results = compute.scientific_computing(gpu_data)
    print(f"Mean: {science_results['mean']}")
    print(f"Standard Deviation: {science_results['std']}")
    
    # Memory Status
    print("\n=== Memory Status ===")
    memory = compute.memory_status()
    print(f"GPU Memory Used: {memory['total_used'] / 1e9:.2f} GB")

if __name__ == "__main__":
    demonstrate_gpu_computing()