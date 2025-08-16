# If running in Colab, uncomment the next line:
# !pip -q install torch torchvision torchaudio torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

import os, random, math
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.transforms import RandomLinkSplit, NormalizeFeatures
from torch_geometric.nn import GCNConv, GAE, VGAE

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
