import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch as t
from torch import nn, Tensor
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
import einops
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
from functools import partial
from tqdm.notebook import tqdm
from dataclasses import dataclass
from rich import print as rprint
from rich.table import Table
from IPython.display import display, HTML
from pathlib import Path

import jaxtyping
import sys
# # Make sure exercises are in the path
# exercises_dir = Path("../exercises").resolve()
# section_dir = (exercises_dir / "part4_superposition_and_saes").resolve()
# if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, hist
from utils import (
    plot_features_in_2d,
    plot_features_in_Nd,
    plot_features_in_Nd_discrete,
    plot_correlated_features,
    plot_feature_geometry,
    frac_active_line_plot,
    plot_features_in_2d_hierarchy
)
# import part4_superposition_and_saes.tests as tests
# import part4_superposition_and_saes.solutions as solutions

if t.backends.mps.is_available():
    print("current PyTorch install was "
              "built with MPS enabled.")
    if t.backends.mps.is_built():
        print("MPS is available")
        device = t.device("mps")
else:
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

## Class and Function for Trees
class Node:
    def __init__(self, val = 0):
        self.children = []
        self.val = val
    # def __str__(self):
    #     return str(index)

    def add_child(self, child = None):
        if child is None:
            self.children.append(Node())
        else:
            self.children.append(child)

    def __repr__(self):
        return f"Node {self.val}: {self.children}"

class Tree:
    def __init__(self, node = None):
        if node is not None:
            self.root = node
        else:
            self.root = Node()
        # self.tree_list = self.to_list()
        
    def to_list(self):
        return self.returnInorder(self.root, result = [])
    
    def returnInorder(self, node, string="", result = []):
            
        if len(node.children) == 0:
            return result
            
        for i in range(len(node.children)):
            if string == "":
                new_string = f"{i}"
            else:
                new_string = f"{string}.{i}"
            result.append(new_string)
            self.returnInorder(node.children[i], new_string, result)
            # result.append(new_string)

        return result

## For constructing a tree with same branching factor over depth d
def construct_tree(branching_factor, depth):
    # Create a new Node
    a = Node()
    # Base case: if depth is 0, return the leaf node
    if depth == 0:
        return a
    # Recursive case: create children for the current node
    for branching_factor_i in range(branching_factor):
        # Recursively construct subtrees and add them as children
        a.add_child(construct_tree(branching_factor, depth - 1))
    # Return the root of the constructed tree
    return a   


## For Toy Model of Hierarchy
def linear_lr(step, steps):
    return (1 - (step / steps))

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))

def get_leaf_indices(tree):
    all_paths = tree.to_list()
    max_dots = max(path.count('.') for path in all_paths)
    leaf_paths = [path for path in all_paths if path.count('.') == max_dots]
    leaf_paths_index = [all_paths.index(path) for path in leaf_paths]
    return leaf_paths_index

@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    tree: Tree
    n_features: int = 6
    # tree_depth: int = 2
    # branching_factor: int = 2
    n_hidden: int = 2
    partial_paths: bool = True

class Model(nn.Module):
    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]
    # Our linear map is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Union[float, Tensor]] = None,
        importance: Optional[Union[float, Tensor]] = None,
        device = device,
    ):
        super().__init__()
        self.cfg = cfg

        if feature_probability is None: feature_probability = t.ones(())
        if isinstance(feature_probability, float): feature_probability = t.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to((cfg.n_instances, cfg.n_features))
        if importance is None: importance = t.ones(())
        if isinstance(importance, float): importance = t.tensor(importance)
        self.importance = importance.to(device).broadcast_to((cfg.n_instances, cfg.n_features))

        self.W = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))))
        self.b_final = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_features)))
        self.to(device)

        self.partial_paths = self.cfg.partial_paths

        if self.cfg.partial_paths:
            self.valid_indices = list(range(self.cfg.n_features))
        else:
            self.valid_indices = get_leaf_indices(self.cfg.tree)

        print(self.valid_indices)

        self.device = device

        all_paths = tree.to_list()
        self.input_tensor = t.stack([t.tensor([1 if any(p == '.'.join(path.split('.')[:i+1]) 
                                                        for i in range(len(path.split('.')))) else 0 for p in all_paths], 
                               device=device).to(t.int) for path in all_paths], dim = 0)


    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        hidden = einops.einsum(
           features, self.W,
           "... instances features, instances hidden features -> ... instances hidden"
        )
        out = einops.einsum(
            hidden, self.W,
            "... instances hidden, instances hidden features -> ... instances features"
        )
        return F.relu(out + self.b_final)

    def sample(self, size):
        return t.tensor(np.random.choice(self.valid_indices, size=size), dtype = t.int)
    
    def generate_batch(self, batch_size) -> Float[Tensor, "batch_size instances features"]:
        '''
        Generates a batch of data using the tree structure and sample_equal_probability function.
        '''
        indices = self.sample(size = (batch_size, self.cfg.n_instances))
        feat = self.input_tensor[indices]
        return feat.to(t.float)

    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        '''
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Remember, `self.importance` will always have shape (n_instances, n_features).
        '''
        error = self.importance * ((batch - out) ** 2)
        loss = einops.reduce(error, 'batch instances features -> instances', 'mean').sum()
        return loss

    def calculate_loss_per_instance(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        '''
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Remember, `self.importance` will always have shape (n_instances, n_features).
        '''
        error = self.importance * ((batch - out) ** 2)
        losses = einops.reduce(error, 'batch instances features -> instances', 'mean')
        return losses

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        '''
        Optimizes the model using the given hyperparameters.
        '''
        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):

                progress_bar.set_postfix(loss=loss.item()/self.cfg.n_instances, lr=step_lr)