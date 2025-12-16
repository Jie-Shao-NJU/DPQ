# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for compressing neural networks implementing the bit allocation from [0].

[0]: Stock, P., Joulin, A., Gribonval, R., Graham, B., & Jégou, H. (2020).
And the bit goes down: Revisiting the quantization of neural networks.
International Conference on Learning Representations (ICLR) 2020.
"""

import math
from typing import Callable

import torch

from pqf.compression.kmeans import kmeans
from pqf.compression.kmeans_sr import src

KMEANS_STR_TO_FN = {"kmeans": kmeans, "src": src}


def get_kmeans_fn(kmeans_name: str) -> Callable:
    """Select the appropriate kmeans function given a string

    Parameters:
        kmeans_name: Either `kmeans` or `src`, depending on what method we want
    Returns:
        kmeans_fn: A plain or src callable function
    Raises:
        ValueError: If the name is not either `kmeans` or `src`
    """
    if kmeans_name not in KMEANS_STR_TO_FN:
        raise ValueError("kmeans function must be one of {}, not {}".format(list(KMEANS_STR_TO_FN.keys()), kmeans_name))
    return KMEANS_STR_TO_FN[kmeans_name]


def nearest_smaller_power_of_two(x):
    """Return the number closest to x which is also a power of two"""
    return int(2 ** math.floor(math.log(x, 2)))


def get_num_centroids(n_blocks_per_row: int, n_output_channels: int, k: int) -> int:
    """ Clamp the number of centroids. From Bit Goes Down paper:
    ''Note that we clamp the number of centroids to $min(k, C_{out} x m/4)$ for stability.''.
    We also clamp the number of stability centroids to the smallest number that is also a power of two.

    Parameters:
        n_blocks_per_row: Number of vector blocks per column used to divide the weight matrix. Called $m$ in the paper
        n_output_channels: Number of output channels. Called $C_{out}$ in the paper
        k: Default number of centroids used throughout the network
    Returns:
        stability_centroids: Number of centroids used to match the bitrated used in the Bit Goes Down paper
    """
    stability_centroids = n_output_channels * float(n_blocks_per_row) / 4
    stability_centroids = nearest_smaller_power_of_two(stability_centroids)
    return min(k, stability_centroids)


def decode(codes_matrix: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Given the codes and codebook, get the uncompressed weight matrix

    Parameters:
        codes_matrix: n-by-m matrix with codes for the compressed matrix
        codebook: k-by-d codebook to use for decoding
    Returns:
        weight: n-by-md decoded matrix
    """
    num_output_rows = codes_matrix.size(0)

    one_dimensional_codes = codes_matrix.reshape(1, -1).squeeze().long()
    one_dimensional_output = torch.index_select(codebook, dim=0, index=one_dimensional_codes)

    return one_dimensional_output.reshape(num_output_rows, -1)

# def decode_all_old(codes_matrixs: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
#     weights = []
#     for codes_matrix, codebook in zip(codes_matrixs, codebooks):
#         weight = decode(codes_matrix, codebook)
#         weights.append(weight)
#     return torch.cat(weights, dim=1)


def decode_all(codes_matrixs: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    num_codebooks, num_output_rows = codes_matrixs.size()   # codes_matrixs shape: [288, 4608]
    codebook_size, codebook_dim = codebooks.size(1), codebooks.size(2)  # codebooks shape: [288, 256, 4]

    offsets = torch.arange(num_codebooks, device=codes_matrixs.device) * codebook_size
    offsets = offsets.view(-1, 1)
    codes_matrixs_offset = codes_matrixs + offsets

    one_dimensional_codes = codes_matrixs_offset.transpose(0,1).contiguous().view(-1)
    one_dimensional_output = codebooks.view(-1, codebook_dim)[one_dimensional_codes]

    return one_dimensional_output.view(num_output_rows, -1)

