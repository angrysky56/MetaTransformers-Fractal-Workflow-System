import cupy as cp
import cupyx.scipy as sp
from typing import Union, Optional, Dict, List
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class ComputeConfig:
    """Configuration for compute operations"""
    enable_profiling: bool = False
    memory_tracking: bool = True
    debug_mode: bool = False

class ComputeCore:
    """
    Pure GPU computation system leveraging CuPy's native capabilities
    """
    def __init__(self, config: Optional[ComputeConfig] = None):
        self.config = config or ComputeConfig()
        self.setup_logging()
        
        if self.config.enable_profiling:
            self._setup_profiling()

    def setup_logging(self):
        """Initialize logging system"""
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=log_path / "compute.log",
            level=logging.DEBUG if self.config.debug_mode else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_profiling(self):
        """Configure performance profiling"""
        self.profiler = cp.cuda.profiler
        self.logger.info("CUDA profiling enabled")

    def process_array(self, data: cp.ndarray) -> cp.ndarray:
        """Process array data"""
        try:
            if not isinstance(data, cp.ndarray):
                return cp.asarray(data)
            return data
        except Exception as e:
            self.logger.error(f"Array processing error: {str(e)}")
            raise

    def fft_analysis(self, signal: cp.ndarray) -> Dict:
        """FFT analysis on GPU"""
        try:
            gpu_signal = self.process_array(signal)
            
            return {
                'fft': cp.fft.fft(gpu_signal),
                'power': cp.abs(cp.fft.fft(gpu_signal)) ** 2,
                'freq': cp.fft.fftfreq(len(gpu_signal))
            }
        except Exception as e:
            self.logger.error(f"FFT analysis error: {str(e)}")
            raise

    def matrix_operations(self, 
                         matrix_a: cp.ndarray,
                         matrix_b: cp.ndarray) -> Dict:
        """Matrix operations on GPU"""
        try:
            a_gpu = self.process_array(matrix_a)
            b_gpu = self.process_array(matrix_b)
            
            return {
                'product': cp.dot(a_gpu, b_gpu),
                'eigenvalues': cp.linalg.eigvals(a_gpu),
                'svd': cp.linalg.svd(a_gpu)
            }
        except Exception as e:
            self.logger.error(f"Matrix operation error: {str(e)}")
            raise

    def scientific_computing(self, 
                           data: cp.ndarray,
                           operation: str = 'all') -> Dict:
        """Scientific computing on GPU"""
        try:
            gpu_data = self.process_array(data)
            results = {}
            
            if operation in ['all', 'stats']:
                results.update({
                    'mean': cp.mean(gpu_data),
                    'std': cp.std(gpu_data),
                    'max': cp.max(gpu_data),
                    'min': cp.min(gpu_data)
                })
                
            if operation in ['all', 'signal']:
                results.update({
                    'spectrogram': sp.signal.spectrogram(gpu_data)[0],
                    'wavelets': sp.signal.cwt(gpu_data, sp.signal.ricker)
                })
                
            if operation in ['all', 'special']:
                results.update({
                    'bessel': sp.special.jv(0, gpu_data),
                    'gamma': sp.special.gamma(cp.abs(gpu_data))
                })
                
            return results
            
        except Exception as e:
            self.logger.error(f"Scientific computing error: {str(e)}")
            raise

    def memory_status(self) -> Dict:
        """GPU memory usage information"""
        if not self.config.memory_tracking:
            return {}
            
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            return {
                'total_allocated': mempool.total_bytes(),
                'total_used': mempool.used_bytes(),
                'total_pinned': pinned_mempool.n_free_blocks(),
            }
        except Exception as e:
            self.logger.error(f"Memory status error: {str(e)}")
            return {}