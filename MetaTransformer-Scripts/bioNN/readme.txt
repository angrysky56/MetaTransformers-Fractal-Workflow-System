conda create -n bionn python=3.11

# check here for current torch https://pytorch.org/

# I got torch-2.5.1-cp311-cp311-win_amd64.whl from:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# To get wheels built
pip install torch_geometric

# You also need Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html


Results from test:

Using device: cuda

Initializing test data with 10 nodes and 16 features...
Created edge index tensor with shape: torch.Size([2, 90])

Initializing hybrid processor...

Processing data through hybrid system...

Processing completed successfully!

Metrics:
uncertainty: 0.000000
bio_state_norm: 105.717255
quantum_state_norm: 3.162278

Final quantum state uncertainty: 0.000000

Tensor Information:
Output shape: torch.Size([10, 32])
Output device: cuda:0
Quantum state shape: torch.Size([10, 32])

(bionn) PS F:\MetaTransformers-Fractal-Workflow-System> & F:/miniconda3/envs/bionn/python.exe f:/MetaTransformers-Fractal-Workflow-System/MetaTransformer-Scripts/bioNN/test_quantum_stdp.py
Testing Quantum-Enhanced STDP Layer...
Using device: cuda

Created test network with:
Nodes: 10
Features: 16
Edge index shape: torch.Size([2, 90])

Processing through STDP layer...

Timestep 0:
Average spike rate: 0.2844
Quantum entanglement: 0.0000

Timestep 1:
Average spike rate: 0.4031
Quantum entanglement: 1.0000

Timestep 2:
Average spike rate: 0.3844
Quantum entanglement: 1.0000

Timestep 3:
Average spike rate: 0.4125
Quantum entanglement: 1.0000

Timestep 4:
Average spike rate: 0.3875
Quantum entanglement: 1.0000

Timestep 5:
Average spike rate: 0.4344
Quantum entanglement: 1.0000

Timestep 6:
Average spike rate: 0.4188
Quantum entanglement: 1.0000

Timestep 7:
Average spike rate: 0.4344
Quantum entanglement: 1.0000

Timestep 8:
Average spike rate: 0.4188
Quantum entanglement: 1.0000

Timestep 9:
Average spike rate: 0.4375
Quantum entanglement: 1.0000

Final Statistics:
Average spike rate across time: 0.4016
Average quantum entanglement: 0.9000

Weight Statistics:
Mean weight magnitude: 0.1871
Mean weight phase: 0.0663
