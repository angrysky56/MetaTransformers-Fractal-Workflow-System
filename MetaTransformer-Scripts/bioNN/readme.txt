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
