"""
Quantum Logic Bridge
Integrates Logic-LLM with BioNN's quantum processing
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from loguru import logger

# Update Logic-LLM path to new location
LOGIC_LLM_PATH = Path(__file__).parent / "Logic-LLM"
if str(LOGIC_LLM_PATH) not in sys.path:
    sys.path.append(str(LOGIC_LLM_PATH))

try:
    sys.path.append(str(LOGIC_LLM_PATH / "models"))
    from logic_program import LogicProgramGenerator
    from logic_inference import LogicInference
    LOGIC_LLM_AVAILABLE = True
except ImportError:
    logger.warning(f"Logic-LLM models not found at {LOGIC_LLM_PATH}/models - some features will be disabled")
    LOGIC_LLM_AVAILABLE = False

class QuantumLogicBridge:
    """Bridge between quantum states and logical processing"""
    
    def __init__(self, openai_key: str = None):
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.coherence_threshold = 0.85
        self.logic_llm_available = LOGIC_LLM_AVAILABLE
        
        if self.logic_llm_available:
            self._initialize_logic_llm()
            logger.info("Logic-LLM initialized successfully")
        else:
            logger.warning(f"Running without Logic-LLM integration. Please ensure models exist at: {LOGIC_LLM_PATH}/models")
    
    def _initialize_logic_llm(self):
        """Initialize Logic-LLM components"""
        from types import SimpleNamespace
        args = SimpleNamespace(
            api_key=self.openai_key,
            model="gpt-4",
            data_path=str(LOGIC_LLM_PATH / "data"),
            save_path=str(LOGIC_LLM_PATH / "outputs"),
            model_path=str(LOGIC_LLM_PATH / "models")
        )
        try:
            self.program_generator = LogicProgramGenerator(args)
            self.inferencer = LogicInference()
            logger.info("Logic-LLM components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Logic-LLM components: {str(e)}")
            self.logic_llm_available = False
    
    def process_quantum_state(self, quantum_state: torch.Tensor, pattern_data: dict = None):
        """Process quantum state through logical reasoning"""
        if not self.logic_llm_available:
            return {
                "success": False,
                "error": "Logic-LLM not available",
                "paths_checked": {
                    "logic_llm": str(LOGIC_LLM_PATH),
                    "models": str(LOGIC_LLM_PATH / "models")
                }
            }
            
        try:
            # Convert quantum state to logical pattern
            logical_pattern = self._quantum_to_logical(quantum_state)
            
            # Generate logical program
            program = self.program_generator.generate(
                context=pattern_data.get('context', ''),
                pattern=logical_pattern
            )
            
            # Run inference
            inference = self.inferencer.infer(program)
            
            # Update quantum state
            new_state = self._logical_to_quantum(inference['pattern'])
            
            return {
                "success": True,
                "new_state": new_state,
                "coherence": self._calculate_coherence(new_state),
                "inference": inference
            }
            
        except Exception as e:
            logger.error(f"Quantum state processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _quantum_to_logical(self, quantum_state: torch.Tensor) -> dict:
        """Convert quantum state to logical pattern"""
        # Extract key quantum properties
        state_matrix = quantum_state.detach().cpu().numpy()
        phase = np.angle(state_matrix)
        magnitude = np.abs(state_matrix)
        
        # Map to logical constructs
        return {
            "pattern_type": "quantum_logical",
            "structure": {
                "phase": phase.tolist(),
                "magnitude": magnitude.tolist()
            },
            "properties": {
                "coherence": float(np.mean(magnitude)),
                "complexity": float(np.std(phase))
            }
        }
    
    def _logical_to_quantum(self, logical_pattern: dict) -> torch.Tensor:
        """Convert logical pattern back to quantum state"""
        phase = np.array(logical_pattern['structure']['phase'])
        magnitude = np.array(logical_pattern['structure']['magnitude'])
        
        # Reconstruct complex quantum state
        state = magnitude * np.exp(1j * phase)
        return torch.tensor(state, dtype=torch.complex64)
    
    def _calculate_coherence(self, quantum_state: torch.Tensor) -> float:
        """Calculate quantum coherence of state"""
        try:
            state_matrix = quantum_state.detach().cpu().numpy()
            magnitude = np.abs(state_matrix)
            phase_coherence = np.abs(np.mean(np.exp(1j * np.angle(state_matrix))))
            
            coherence = float(np.mean(magnitude) * phase_coherence)
            return min(coherence, 1.0)
            
        except Exception as e:
            logger.error(f"Coherence calculation failed: {str(e)}")
            return 0.0

if __name__ == "__main__":
    # Test the bridge
    bridge = QuantumLogicBridge()
    
    print("\nChecking Logic-LLM availability...")
    print(f"LOGIC_LLM_PATH: {LOGIC_LLM_PATH}")
    print(f"Models directory exists: {(LOGIC_LLM_PATH / 'models').exists()}")
    print(f"Logic-LLM available: {bridge.logic_llm_available}")
    
    # Create test quantum state
    test_state = torch.randn(3, 3, dtype=torch.complex64)
    test_data = {
        "context": "Quantum state represents logical pattern structure"
    }
    
    result = bridge.process_quantum_state(test_state, test_data)
    print(f"\nProcessing result: {result}")
