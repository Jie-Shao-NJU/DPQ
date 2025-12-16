import torch
import numpy as np
from matplotlib import pyplot as plt
from pq.utils_model import calculate_outliers, plot_3d_surface
from timm.layers.helpers import to_2tuple

## stage1: calculate the u, s, v of activation tensor
def cal_cov(tensor_i, avg_tensor_i, ratio):
    u, s, vh = torch.linalg.svd(tensor_i, full_matrices=False)
    return  u, vh, avg_tensor_i

def cal_output_sum_square(fc_output_list, fc_square_list, fc_sum_list, fc_count_list):
    for i in range(len(fc_output_list)):
        if len(fc_output_list[i].shape) == 3:
            tensor_i = fc_output_list[i].flatten(0, 1)
        elif len(fc_output_list[i].shape) == 2:
            tensor_i = fc_output_list[i]      
        fc_sum = torch.sum(tensor_i, dim=0)
        fc_square = tensor_i.T @ tensor_i
        fc_count_list[i] += tensor_i.shape[0]
        if fc_square_list[i] is None:
            fc_sum_list[i] = fc_sum
            fc_square_list[i] = fc_square
        else:
            fc_sum_list[i] += fc_sum
            fc_square_list[i] += fc_square

def cal_save_uvb(fc_square_list, fc_sum_list, fc_count_list, fc_ratio, save_name = "fc1", save_path= "test", logger=None):
    for i in range(len(fc_square_list)):
        logger.info(f"Save name: {save_name}, index: {i}")
        fc_square = fc_square_list[i]
        fc_sum = fc_sum_list[i]
        if fc_sum is not None:
            avg_tensor_i = fc_sum.reshape(-1, 1) / fc_count_list[i]
            cov_tensor_i = fc_square / fc_count_list[i] - avg_tensor_i @ avg_tensor_i.T
            u, v, b  = cal_cov(cov_tensor_i, avg_tensor_i, fc_ratio[i])
            u_name = save_path + save_name  + "_"  + str(i) + "_u.pt"
            torch.save(u, u_name)
            v_name = save_path + save_name  + "_" + str(i) + "_v.pt"
            torch.save(v, v_name)
            b_name = save_path + save_name  + "_" + str(i) + "_avg.pt"
            torch.save(b, b_name)


## stage2: merge the u, v, b to the model and calculate the score (loss)
def merge_model(checkpoint, block_i, fc_i, dim_i, load_path, logger, visual=False):
    fc_params = {
        1: ("fc1", ".mlp.", "fc1", "fc1_u", "fc1_v"),
        2: ("fc2", ".mlp.", "fc2", "fc2_u", "fc2_v"),
        3: ("qkv", ".attn.", "qkv", "qkv_u", "qkv_v"),
        4: ("proj", ".attn.", "proj", "proj_u", "proj_v"),
        5: ("adaln", ".adaLN_modulation.", "1", "1", "2")
    }
    
    file_name, layer_name, org_name, w_u_name, w_v_name = fc_params[fc_i]

    u = torch.load(f"{load_path}{file_name}_{block_i}_u.pt", map_location='cpu').cpu()
    v = torch.load(f"{load_path}{file_name}_{block_i}_v.pt", map_location='cpu').cpu()
    avg = torch.load(f"{load_path}{file_name}_{block_i}_avg.pt", map_location='cpu').cpu()

    new_u = u[:, :dim_i]
    new_v = v[:dim_i, :]
    b = (torch.eye(new_u.shape[0]).cpu() - new_u @ new_v) @ avg

    original_w_name = f"blocks.{block_i}{layer_name}{org_name}.weight"
    original_b_name = f"blocks.{block_i}{layer_name}{org_name}.bias"
    new_w_name = f"blocks.{block_i}{layer_name}{w_u_name}.weight"
    new_b_name = f"blocks.{block_i}{layer_name}{w_u_name}.bias"
    
    original_w = checkpoint.pop(original_w_name)
    original_b = checkpoint.pop(original_b_name)

    new_w = new_v @ original_w
    new_b = original_b @ new_v.transpose(0, 1)

    if visual:
        min_w, max_w, _ = calculate_outliers(original_w, logger, "Original W")
        min_new_w, max_new_w, _ = calculate_outliers(new_w, logger, "New W")
        min_new_u, max_new_u, _ = calculate_outliers(new_u, logger, "New U")

        fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})  # Add projection='3d' here
        global_min = min(min_w, min_new_w, min_new_u)
        global_max = max(max_w, max_new_w, max_new_u)

        plot_3d_surface(original_w, axs[0], 'Original W', vmin=global_min, vmax=global_max)
        plot_3d_surface(new_w, axs[1], 'New W', vmin=global_min, vmax=global_max)
        plot_3d_surface(new_u, axs[2], 'New U', vmin=global_min, vmax=global_max)

    checkpoint[new_w_name] = new_w
    checkpoint[new_b_name] = new_b

    k_split = new_w_name.split(".")
    k_split[-2] = w_v_name
    final_key = ".".join(k_split)
    
    logger.info(final_key)
    checkpoint[final_key] = new_u
    checkpoint[final_key.replace("weight", "bias")] = b.reshape(-1)

    return checkpoint


def reset_param(num_layers):
    fc1_len = [128] * num_layers
    fc2_len = [128] * num_layers
    qkv_len = [128] * num_layers
    proj_len = [128] * num_layers
    adaln_len = [128] * num_layers

    fc1_use_uv = [False] * num_layers
    fc2_use_uv = [False] * num_layers
    qkv_use_uv = [False] * num_layers
    proj_use_uv = [False] * num_layers
    adaln_use_uv = [False] * num_layers

    return fc1_len, fc2_len, qkv_len, proj_len, adaln_len, fc1_use_uv, fc2_use_uv, qkv_use_uv, proj_use_uv, adaln_use_uv


## stage3: get the optimal model according to the loss
def dp(value, speed):
    """
    Args:
        value (np.ndarray): A 2D array of shape [num_dims, num_blocks], where value[i, j] represents 
                            the loss for the i-th dimension in the j-th block.
        speed (list or np.ndarray): A 1D array where each element represents the parameter cost for 
                                    each dimension.

    Returns:
        dict: A dictionary where the keys are the total number of parameters (sum of selected dimensions' 
              parameter costs), and the values are lists of the selected dimensions for each block and 
              the total accumulated loss.
              Format: {total_param_count: [dim1_idx, dim2_idx, ..., total_loss]}
    """

    count = 0 
    results = {}

    for bi in range(value.shape[1]):
        if results == {}:
            for bj in range(value.shape[0]):
                results[speed[bj]] = [bj, value[bj, bi]]  # results: {param_cost: [dim_idx, loss]}
        else:
            single = {}
            for k, v in results.items():  # k: param_cost, v: [dim_idx_list, loss]
                for bj in range(value.shape[0]):
                    sum_val = v[-1] + value[bj, bi]  # Calculate the new loss (previous loss + current loss)
                    count = float(k) + speed[bj]  # Calculate the new parameter cost (previous + current)
                    if count not in single.keys():
                        single[count] = v[:-1] + [bj, sum_val]  # New entry with the new dimension and loss
                    elif sum_val < single[count][-1]:
                        single[count] = v[:-1] + [bj, sum_val]  # Update with the better path
            results = single

    return results


def remove_unreason_item(results):
    """
    Args:
        results (dict): A dictionary where keys are parameter counts and values are lists 
                        containing the path of selected dimensions and the associated loss.
                        Format: {param_count: [dim1_idx, dim2_idx, ..., total_loss]}

    Returns:
        dict: A filtered dictionary where each key is a parameter count, and the value is the 
              path and total loss for that param count. Only paths with a decreasing loss for increasing 
              parameter costs are retained.
    """

    f = {}
    keys = list(results.keys())
    keys.sort(reverse=False)  # Sort parameter counts in ascending order
    b = {key: results[key] for key in keys}

    for k, v in b.items():  # k: param_count, v: [path, total_loss]
        if not f:
            f[k] = v
        else:
            flag = True
            for fk, fv in f.items():
                if fv[-1] < v[-1]:  # If any existing path has a lower loss, don't add this one
                    flag = False
            if flag:
                f[k] = v  # Keep only paths with better (lower) loss

    return f


def get_compress_results(params, dim, value, f):
    """
    Args:
        params (list or np.ndarray): A 1D array of parameter costs for each dimension.
        dim (list or np.ndarray): A list of dimension indices to be checked against the loaded file data.
        value (np.ndarray): A 2D array to store the losses. Shape: [num_dims, num_blocks]. 
                            This will be populated with loss values from the files.
        f (int): A specific file identifier used in constructing file paths for loading the loss data.

    Returns:
        dict: A dictionary containing the optimal paths after dynamic programming and filtering.
              Format: {param_count: [dim1_idx, dim2_idx, ..., total_loss]}
    """

    for b in range(28):
        ind_value = b
        file_name = "results/low_rank/008-DiT-XL-2/dim_loss/" + str(b) + "_" + str(f) + ".txt"
        files = np.loadtxt(file_name, delimiter=',')
        param_kl_dict = dict(zip(files[:, 0].astype(np.int32), files[:, 1]))

        for i in range(len(dim)):
            if dim[i] in param_kl_dict.keys():
                value[i, ind_value] = param_kl_dict[dim[i]]  # Store loss for [dimension, block]

    results = dp(value, params)
    results = remove_unreason_item(results)

    return results


def get_ind(percent, results, total_param):
    """
    Args:
        percent (float): The percentage (0 to 1) of the total parameters to be used.
        results (dict): A dictionary where the keys are parameter counts and the values are lists 
                        of selected dimensions and their associated loss.
                        Format: {param_count: [dim1_idx, dim2_idx, ..., total_loss]}
        total_param (float): The total available parameter count.

    Returns:
        list: The optimal path (list of dimension indices) corresponding to the closest 
              parameter count to the target (total_param * percent).
    """

    params = [key for key in results]
    target_param_count = total_param * percent
    closest_key = min(params, key=lambda x: abs(x - target_param_count))

    return results[closest_key]  # Return the optimal path corresponding to the closest parameter count


def get_dims(ind, dim, compress_all=False):
    """
    Args:
        ind (list): A list of indices representing the selected dimensions for each block.
        dim (list or np.ndarray): A list or array containing all available dimension indices.

    Returns:
        tuple:
            uv_dim (list): A list of selected dimensions for each block.
            use_uv (list): A list of boolean flags indicating whether a UV operation is used for each block.
                           If a dimension index is 0, the corresponding flag is set to False.
    """

    uv_dim = []
    use_uv = [True] * 28

    for i in range(28):
        uv_dim.append(dim[ind[i]])  # Append the dimension corresponding to the current index
        if ind[i] == 0 and (not compress_all) :  # If the selected index is 0, set the flag to False for that block
            use_uv[i] = False

    return uv_dim, use_uv


def get_blocks(percent=0.9, f=1, compress_all=False):
    bias = to_2tuple(True)
    dim_ranges = {
        1: ([921.6] + np.arange(896, 32, -32).tolist(), 1152, 4608),
        2: ([921.6] + np.arange(896, 32, -32).tolist(), 4608, 1152),
        3: ([864] + np.arange(832, 32, -32).tolist(), 1152, 3456),
        4: ([576] + np.arange(544, 32, -32).tolist(), 1152, 1152),
        5: ([987.4] + np.arange(960, 32, -32).tolist(), 1152, 6912),
    }

    dim, in_features, hidden_features = dim_ranges[f]
    params = []
    if not compress_all:
        param_count_fc = in_features * hidden_features + (hidden_features if bias[0] else 0)
        params.append(param_count_fc)
    for dim_size in dim[1:]:
        param_count_u = in_features * dim_size + (dim_size if bias[0] else 0)
        param_count_v = dim_size * hidden_features + (hidden_features if bias[0] else 0)
        params.append(param_count_u + param_count_v)
    if compress_all:
        dim = dim[1:]

    total_param = (in_features * hidden_features + (hidden_features if bias[0] else 0)) * 28
    value = np.zeros([len(dim), 28])
    results = get_compress_results(params, dim, value, f=f)
    ind = get_ind(percent=percent, results=results, total_param=total_param)
    
    return get_dims(ind, dim, compress_all)