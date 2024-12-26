import cupy as cp
import numpy as np
from time import perf_counter
from typing import Union, Optional, Tuple, List
import logging
from pathlib import Path

class GPUAccelerator:
    """
    GPU Acceleration Framework using CuPy
    Implements efficient NumPy/SciPy operations on GPU
    """
    def __init__(self):
        self.setup_logging()
        self.initialize_gpu()
        
    def setup_logging(self):
        """Configure logging system"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=log_dir / "gpu_operations.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def initialize_gpu(self):
        """Initialize GPU and memory pool"""
        try:
            # Get GPU device information
            self.device = cp.cuda.runtime.getDeviceCount()
            self.logger.info(f"Found {self.device} CUDA device(s)")
            
            # Enable memory pool
            with cp.cuda.Device(0):
                self.pool = cp.cuda.MemoryPool()
                cp.cuda.set_allocator(self.pool.malloc)
                self.logger.info("Memory pool initialized")
                
            # Log device properties
            props = cp.cuda.runtime.getDeviceProperties(0)
            self.logger.info(f"Using device: {props['name'].decode()}")
            self.logger.info(f"Compute capability: {props['computeCapability']}")
            
        except Exception as e:
            self.logger.error(f"GPU initialization error: {str(e)}")
            raise

    def to_gpu(self, data: np.ndarray) -> cp.ndarray:
        """Transfer NumPy array to GPU memory"""
        try:
            start = perf_counter()
            gpu_array = cp.asarray(data)
            transfer_time = perf_counter() - start
            
            self.logger.debug(f"Data transferred to GPU in {transfer_time:.4f}s")
            return gpu_array
            
        except Exception as e:
            self.logger.error(f"GPU transfer error: {str(e)}")
            raise

    def to_cpu(self, data: cp.ndarray) -> np.ndarray:
        """Transfer CuPy array to CPU memory"""
        try:
            start = perf_counter()
            cpu_array = cp.asnumpy(data)
            transfer_time = perf_counter() - start
            
            self.logger.debug(f"Data transferred to CPU in {transfer_time:.4f}s")
            return cpu_array
            
        except Exception as e:
            self.logger.error(f"CPU transfer error: {str(e)}")
            raise

    def accelerated_computation(self, 
                              func_name: str, 
                              *args, 
                              **kwargs) -> np.ndarray:
        """
        Execute NumPy/SciPy function on GPU with automatic memory management
        
        Args:
            func_name: Name of NumPy function to accelerate
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result as NumPy array
        """
        try:
            # Get CuPy equivalent function
            cp_func = getattr(cp, func_name)
            
            # Transfer inputs to GPU
            gpu_args = [self.to_gpu(arg) if isinstance(arg, np.ndarray) 
                       else arg for arg in args]
            
            # Execute on GPU
            start = perf_counter()
            gpu_result = cp_func(*gpu_args, **kwargs)
            compute_time = perf_counter() - start
            
            # Transfer result back to CPU
            cpu_result = self.to_cpu(gpu_result)
            
            self.logger.info(f"{func_name} computed on GPU in {compute_time:.4f}s")
            return cpu_result
            
        except Exception as e:
            self.logger.error(f"GPU computation error: {str(e)}")
            raise

    def benchmark_operation(self, 
                          func_name: str,
                          sample_data: np.ndarray,
                          iterations: int = 100) -> dict:
        """
        Benchmark CPU vs GPU performance for a given operation
        
        Args:
            func_name: NumPy function name to benchmark
            sample_data: Sample input data
            iterations: Number of iterations for timing
            
        Returns:
            Dictionary with timing results
        """
        results = {"cpu_time": [], "gpu_time": [], "speedup": None}
        
        # Get both CPU and GPU functions
        np_func = getattr(np, func_name)
        cp_func = getattr(cp, func_name)
        
        # CPU timing
        start = perf_counter()
        for _ in range(iterations):
            _ = np_func(sample_data)
        cpu_time = (perf_counter() - start) / iterations
        results["cpu_time"] = cpu_time
        
        # GPU timing
        gpu_data = self.to_gpu(sample_data)
        start = perf_counter()
        for _ in range(iterations):
            _ = cp_func(gpu_data)
        gpu_time = (perf_counter() - start) / iterations
        results["gpu_time"] = gpu_time
        
        # Calculate speedup
        results["speedup"] = cpu_time / gpu_time
        
        self.logger.info(
            f"Benchmark results for {func_name}:\n"
            f"CPU time: {cpu_time:.4f}s\n"
            f"GPU time: {gpu_time:.4f}s\n"
            f"Speedup: {results['speedup']:.2f}x"
        )
        
        return results

    def get_memory_info(self) -> dict:
        """Get current GPU memory usage information"""
        try:
            with cp.cuda.Device(0) as device:
                info = device.mem_info
                total = info[1]
                used = info[1] - info[0]
                
                return {
                    "total": total,
                    "used": used,
                    "free": info[0],
                    "utilization": used / total * 100
                }
                
        except Exception as e:
            self.logger.error(f"Memory info error: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize accelerator
    accelerator = GPUAccelerator()
    
    # Create sample data
    data = np.random.rand(1000, 1000)
    
    # Perform accelerated computation
    result = accelerator.accelerated_computation(
        "matmul",
        data,
        data.T
    )
    
    # Run benchmark
    benchmark = accelerator.benchmark_operation(
        "matmul",
        data
    )
    
    # Print memory usage
    memory = accelerator.get_memory_info()
    print(f"GPU Memory Usage: {memory['utilization']:.2f}%")