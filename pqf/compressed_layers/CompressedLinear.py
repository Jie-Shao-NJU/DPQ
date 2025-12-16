# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pqf.compression.coding import decode, decode_all, get_num_centroids
from pqf.compression.kmeans import kmeans
from pqf.compression.kmeans_sr import src
from pqf.compressed_layers.AbstractCompressedLayer import AbstractCompressedLayer


class CompressedLinear(AbstractCompressedLayer):
    """Compressed representation of a linear layer"""

    def __init__(self, codes_matrix: torch.Tensor, codebook: torch.Tensor, bias: Optional[torch.Tensor] = None, type: str = 'new'):
        super(CompressedLinear, self).__init__()

        self.initialize_codes(codes_matrix, codebook)

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

        self.codebook = nn.Parameter(codebook)
        self.type = type

    def _get_uncompressed_weight(self):
        if self.type == 'new':
           return decode_all(self.codes_matrix, self.codebook).float()
        else:
            return decode(self.codes_matrix, self.codebook).float()

    def forward(self, x):
        return F.linear(input=x, weight=self._get_uncompressed_weight(), bias=self.bias)

    @staticmethod
    def from_uncompressed(
        uncompressed_layer,
        k,
        k_means_n_iters,
        kmeans_fn,
        subvector_size,
        name = "",
        logger = None,
        type = 'new'
    ):
        """Given an uncompressed layer, initialize the compressed equivalent according to the specified parameters

        Parameters:
            uncompressed_layer: Linear layer to compress
            k: Size of the codebook
            k_means_n_iters: Number of iterations of k means
            subvector_size: Subvector size for the layer
            name : Name of the layer to print alongside mean-squared error
        Returns:
            compressed_layer: Initialized compressed layer
        """
        if type == 'new':
            assert kmeans_fn in [kmeans, src]   
            weight = uncompressed_layer.weight.detach()
            c_out, c_in = weight.size()
            num_blocks_per_row = c_in // subvector_size
            training_set = weight.reshape(weight.shape[0], -1, subvector_size)
            num_centroids = get_num_centroids(num_blocks_per_row, c_out, k)
            
            if k_means_n_iters > 0:
                all_codebooks = []
                all_codes = []
                for i in range(training_set.shape[1]):
                    training_set_block = training_set[:, i]
                    codebook, codes = kmeans_fn(training_set_block, k=num_centroids, n_iters=1, slow_cb_update=True)
                    # codebook, codes = kmeans_fn(training_set_block, k=num_centroids, n_iters=k_means_n_iters, slow_cb_update=True)
                    codes_matrix = codes.reshape(-1)
                    all_codebooks.append(codebook.unsqueeze(0))
                    all_codes.append(codes_matrix.unsqueeze(0))
                    if i % 100 == 0:
                        logger.info(f"{name}: {i}")
                all_codebooks = torch.cat(all_codebooks, dim=0)
                all_codes = torch.cat(all_codes, dim=0)
            else:
                all_codebooks = torch.zeros((num_blocks_per_row, k, subvector_size), device=weight.device)
                all_codes = torch.zeros((num_blocks_per_row, c_out), device=weight.device, dtype=torch.int64)
            
            # Log quantization error
            decoded_weights = decode_all(all_codes, all_codebooks)
            error = (decoded_weights - weight).pow(2).sum() / (num_blocks_per_row * weight.size(0))
            AbstractCompressedLayer.log_quantization_error(name, k_means_n_iters, error, all_codebooks, all_codes, logger=logger)

            return CompressedLinear(all_codes, all_codebooks, uncompressed_layer.bias, type=type)
        
        else:
            assert kmeans_fn in [kmeans, src]   
            weight = uncompressed_layer.weight.detach()
            c_out, c_in = weight.size()
            num_blocks_per_row = c_in // subvector_size
            training_set = weight.reshape(-1, subvector_size)

            if k_means_n_iters > 0:
                num_centroids = get_num_centroids(num_blocks_per_row, c_out, k)
                codebook, codes = kmeans_fn(training_set, k=num_centroids, n_iters=k_means_n_iters, slow_cb_update=True)
            else:
                codebook = torch.zeros((k, subvector_size), device=weight.device)
                codes = torch.zeros((num_blocks_per_row, c_out), device=weight.device, dtype=torch.int64)

            codes_matrix = codes.reshape(-1, num_blocks_per_row)
            decoded_weights = decode(codes_matrix, codebook)
            error = (decoded_weights - weight).pow(2).sum() / (num_blocks_per_row * weight.size(0))
            AbstractCompressedLayer.log_quantization_error(name, k_means_n_iters, error, codebook, codes_matrix, logger)

            return CompressedLinear(codes_matrix, codebook, uncompressed_layer.bias, type=type)
    

    def reset_code(self, uncompressed_weight: torch.Tensor):
        weight = uncompressed_weight.detach()
        training_set = weight.reshape(-1, self.codebook.shape[1])
        d = torch.sum(training_set**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', training_set, self.codebook.T)
        self.codes_matrix = nn.Parameter(torch.argmin(d, dim=1).view(weight.shape[0], -1).byte(), requires_grad=False).to(uncompressed_weight.device)
    
    def get_loss(self, uncompressed_weight: torch.Tensor):
        weight = uncompressed_weight.detach()
        training_set = weight.reshape(-1, self.codebook.shape[1])
        d = torch.sum(training_set**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', training_set, self.codebook.T)
        self.codes_matrix = nn.Parameter(torch.argmin(d, dim=1).view(weight.shape[0], -1).byte(), requires_grad=False)

        weight_quantized = decode(self.codes_matrix, self.codebook)
        codebook_loss = torch.mean((weight_quantized - weight) **2)

        return codebook_loss
    
    def get_loss_act(self, uncompressed_weight: torch.Tensor, act: torch.Tensor):
        act = act.reshape(act.shape[0]*act.shape[1], -1, 4)  # [2048, 288, 4]
        weight = uncompressed_weight.detach()   # [6912, 1152]
        training_set = weight.reshape(weight.shape[0], -1, self.codebook.shape[1])  # [6912, 288, 4]
        wx = torch.einsum('bnc,dc->bnd', act, training_set)  # xW, shape: [2048, 6912, 288]
        # self.codebook.shape: [256, 4]
        wx_codebook = torch.einsum('bnc,dc->bnd', act, self.codebook)  # xW', shape: [2048, , D]
        # Compute L2 distance ||xW - xW'||^2
        d = torch.sum(wx**2, dim=2, keepdim=True) + \
            torch.sum(wx_codebook**2, dim=2) - 2 * torch.einsum('bnd,bnd->bnd', wx, wx_codebook)
        self.codes_matrix = nn.Parameter(torch.argmin(d, dim=2).byte(), requires_grad=False).to(uncompressed_weight.device)
        codebook_loss = torch.mean((wx - wx_codebook) ** 2)
        return codebook_loss
