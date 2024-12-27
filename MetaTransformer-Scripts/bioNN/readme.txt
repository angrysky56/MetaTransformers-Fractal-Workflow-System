conda create -n bionn python=3.11

# check here for current torch https://pytorch.org/

# I got torch-2.5.1-cp311-cp311-win_amd64.whl from:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# To get wheels built
pip install torch_geometric

# You also need Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
