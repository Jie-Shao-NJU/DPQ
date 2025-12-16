import numpy as np
import copy
from tqdm import tqdm
import json
import torch

def dp(value, speed):
    count = 0
    results = {}
    for bi in range(value.shape[1]):
        if results == {}:
            for bj in range(value.shape[0]):
                results[speed[bj]] = [bj, value[bj, bi]]
        else:
            single = {}
            for k,v in results.items():
                for bj in range(value.shape[0]):

                    sum_val =  v[-1] + value[bj, bi] 
                    count = float(k) + speed[bj]

                    if count not in single.keys():
                        single[count] = v[:-1] + [bj, sum_val]
                    elif sum_val < single[count][-1]:
                        single[count] = v[:-1] + [bj, sum_val]
            results = single
    return results

def remove_unreason_item(results):
    f = {}
    keys = list(results.keys())
    keys.sort(reverse=False)
    b = {key:results[key] for key in keys}
    # print(b)
    for k,v in b.items():
        if list(f.keys()) == []:
            f[k] = v 
            # print(f)
        else:
            flag = True
            for fk,fv in f.items():
                if fv[-1] < v[-1]:
                    flag = False
            if flag:
                f[k] = v
    return f


def gen_final_results(results_qkv, results_ffn):
    f = {}
    for k1,v1 in results_qkv.items():
        for k2,v2 in results_ffn.items():
            sum_block = k1 +  k2
            count = v1[-1] + v2[-1]
            if sum_block not in f.keys():
                f[sum_block] = v1[:-1] + v2[:-1] + [count]
            elif count < f[sum_block][-1]:
                f[sum_block] = v1[:-1] + v2[:-1] + [count]
    # print(f)
    return f


def get_qkv_results(params, dim, value):
    for b in range(12):
        f = 3
        ind_value = b
        file_name = "deit/"+ str(b) + "_" +str(f) + ".txt"
        files = np.loadtxt(file_name,delimiter=',')
        param_kl_dict = dict(zip(files[:,0].astype(np.int32),files[:,1]))
        # print(param_kl_dict)
        for i in range(len(dim)):
            if dim[i] in param_kl_dict.keys():
                value[i, ind_value] = param_kl_dict[dim[i]] 
        value[0] = 0
    print(value)
    results = dp(value, params)
    results = remove_unreason_item(results)
    return results

def get_ffn_results(params, dim, value):
    for b in range(12):
        for f in range(1,3):
            ind_value = 2*b +f -1
            file_name = "deit/"+ str(b) + "_" +str(f) + ".txt"
            files = np.loadtxt(file_name,delimiter=',')
            param_kl_dict = dict(zip(files[:,0].astype(np.int32),files[:,1]))
            # print(param_kl_dict)
            for i in range(len(dim)):
                if dim[i] in param_kl_dict.keys():
                    value[i, ind_value] = param_kl_dict[dim[i]] 
    print(value)
    results = dp(value, params)
    results = remove_unreason_item(results)
    return results


## Deit B
# qkv_params = [1769472, 1671168, 1572864, 1474560, 1376256, 1277952, 1179648, 1081344, 983040, 884736, 786432, 688128, 589824, 491520, 393216]
# qkv_dim =[576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128]
# qkv_value = np.zeros([len(qkv_dim),12])
# results_qkv = get_qkv_results(qkv_params, qkv_dim, qkv_value)

# ffn_params = [2359296, 2334720, 2211840, 2088960, 1966080, 1843200, 1720320, 1597440, 1474560, 1351680, 1228800, 1105920, 983040, 860160, 737280, 614400, 491520]
# ffn_dim =[614.4, 608, 576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128]
# ffn_value = np.zeros([len(ffn_dim),24])
# results_ffn = get_ffn_results(ffn_params, ffn_dim, ffn_value)

# results = gen_final_results(results_qkv, results_ffn)
# results = remove_unreason_item(results)
# print(results)

## Deit S
# qkv_params = [442368, 393216, 344064, 294912, 245760, 196608, 147456, 98304]
# qkv_dim =[ 288, 256, 224, 192, 160, 128, 96, 64]
# qkv_value = np.zeros([len(qkv_dim),12])
# results_qkv = get_qkv_results(qkv_params, qkv_dim, qkv_value)

# ffn_params = [589824, 552960, 491520, 430080, 368640, 307200, 245760, 184320, 122880]
# ffn_dim =[307.2, 288, 256, 224, 192, 160, 128, 96, 64]
# ffn_value = np.zeros([len(ffn_dim),24])
# results_ffn = get_ffn_results(ffn_params, ffn_dim, ffn_value)

# results = gen_final_results(results_qkv, results_ffn)
# results = remove_unreason_item(results)
# print(results)

## Deit T
# qkv_params = [110592, 98304, 73728, 49152, 24576]
qkv_dim =[144, 128, 96, 64, 32]
# qkv_value = np.zeros([len(qkv_dim),12])
# results_qkv = get_qkv_results(qkv_params, qkv_dim, qkv_value)

# ffn_params = [147456, 122880, 92160, 61440, 30720]
ffn_dim =[153.6, 128, 96, 64, 32]
# ffn_value = np.zeros([len(ffn_dim),24])
# results_ffn = get_ffn_results(ffn_params, ffn_dim, ffn_value)

# results = gen_final_results(results_qkv, results_ffn)
# results = remove_unreason_item(results)
# print(results)

def get_dims(ind, blocks, qkv_dim, ffn_dim):
    # dim_qkv = [576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128]
    # dim_ffn = [614.4, 608, 576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128]

    qkv = []
    fc1 = []
    fc2 = []

    qkv_use_uv = [True] * blocks
    fc1_use_uv = [True] * blocks
    fc2_use_uv = [True] * blocks

    for i in range(blocks):
        qkv.append(qkv_dim[ind[i]])
        if ind[i] == 0:
            qkv_use_uv[i] = False
    for j in range(blocks, len(ind)):
        if j % 2 == 0:
            fc1.append(ffn_dim[ind[j]])
            if ind[j] == 0:
                fc1_use_uv[(j-blocks)//2] = False
        else:
            fc2.append(ffn_dim[ind[j]])
            if ind[j] == 0:
                fc2_use_uv[(j-blocks)//2] = False
    print(qkv)
    print(fc1)
    print(fc2)

    print(qkv_use_uv)
    print(fc1_use_uv)
    print(fc2_use_uv)

# ind = [14, 14, 12, 11, 8, 8, 7, 7, 4, 0, 0, 0, 15, 11, 15, 13, 13, 12, 10, 11, 10, 11, 9, 11, 9, 8, 8, 5, 0, 0, 0, 0, 0, 0, 0, 5]
# get_dims(ind, 12)

# ind = [5, 6, 5, 5, 5, 5, 4, 3, 2, 3, 3, 0, 5, 4, 5, 5, 4, 5, 5, 6, 5, 6, 5, 5, 5, 5, 0, 3, 0, 4, 0, 3, 0, 4, 0, 2]
# get_dims(ind, 12, qkv_dim, ffn_dim)

# ind = [13, 13, 10, 7, 4, 4, 3, 0, 0, 0, 0, 0, 14, 9, 14, 11, 11, 8, 7, 6, 5, 5, 5, 4, 6, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ind = [14, 14, 14, 13, 11, 11, 10, 9, 7, 6, 0, 0, 15, 12, 16, 14, 14, 13, 12, 12, 11, 12, 11, 12, 12, 11, 10, 9, 6, 0, 0, 0, 0, 0, 0, 7]
# get_dims(ind, 12, qkv_dim, ffn_dim)


# 0.4
# ind = [5, 6, 5, 5, 6, 5, 5, 4, 3, 3, 4, 7, 5, 4, 5, 5, 5, 6, 5, 7, 6, 7, 6, 7, 5, 5, 4, 5, 0, 4, 0, 5, 0, 5, 0, 4]
# 0.33
# [5, 6, 5, 5, 5, 5, 5, 3, 2, 3, 3, 7, 5, 4, 5, 5, 4, 5, 5, 6, 5, 6, 5, 6, 5, 5, 0, 3, 0, 4, 0, 0, 0, 4, 0, 0]
# 0.25
# [5, 6, 5, 5, 5, 4, 3, 3, 0, 0, 3, 0, 5, 3, 5, 4, 4, 4, 5, 6, 5, 5, 5, 5, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#0.2
# [5, 6, 4, 4, 4, 4, 3, 0, 0, 0, 0, 0, 5, 3, 4, 4, 3, 4, 4, 5, 5, 5, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 0.15
# ind = [4, 6, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 5, 3, 4, 3, 2, 3, 4, 4, 4, 4, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# get_dims(ind, 12, qkv_dim, ffn_dim)


# DeiT T
ind = [4, 3, 3, 2, 2, 2, 2, 2, 1, 0, 0, 4, 1, 0, 1, 2, 0, 2, 2, 3, 2, 3, 2, 3, 2, 3, 1, 3, 0, 2, 3, 3, 4, 4, 0, 4]
get_dims(ind, 12, qkv_dim, ffn_dim)

def load_model_torch_deit_ada():
    model_dict = torch.load("pretrainmodel/deit_tiny_patch16_224-a1311bcf.pth", map_location='cpu')
    # model_dict = torch.load("pretrainmodel/deit_base_patch16_224-b5f2ef4d.pth",map_location='cpu')
    # model_dict = torch.load("pretrainmodel/deit_small_patch16_224-cd65a155.pth",map_location='cpu')
    model_dict = model_dict['model']
    u_list = []
    v_list = []


    # root_path = "deit/ffn_output/all_fc_ada/deit_b/"
    # qkv_len = [128, 128, 192, 224, 320, 320, 352, 352, 448, 576, 576, 576]
    # fc1_len = [160, 160, 224, 320, 320, 352, 352, 384, 614.4, 614.4, 614.4, 614.4]
    # fc2_len = [288, 224, 256, 288, 288, 288, 384, 480, 614.4, 614.4, 614.4, 480]

    # qkv_len = [160, 160, 256, 352, 448, 448, 480, 576, 576, 576, 576, 576]
    # fc1_len = [192, 192, 288, 416, 480, 480, 448, 512, 614.4, 614.4, 614.4, 614.4]
    # fc2_len = [352, 288, 384, 448, 480, 512, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4]
    # [True, True, True, True, True, True, True, False, False, False, False, False]
    # [True, True, True, True, True, True, True, True, False, False, False, False]
    # [True, True, True, True, True, True, False, False, False, False, False, False]


    # qkv_len = [128, 128, 128, 160, 224, 224, 256, 288, 352, 384, 576, 576]
    # fc1_len = [160, 128, 192, 256, 288, 288, 256, 320, 448, 614.4, 614.4, 614.4]
    # fc2_len = [256, 192, 224, 256, 256, 256, 288, 352, 614.4, 614.4, 614.4, 416]
    # [True, True, True, True, True, True, True, True, True, True, False, False]
    # [True, True, True, True, True, True, True, True, True, False, False, False]
    # [True, True, True, True, True, True, True, True, False, False, False, True]

    # root_path = "deit/ffn_output/all_fc_ada/deit_s/"
    # qkv_len = [128, 96, 128, 128, 96, 128, 128, 160, 192, 192, 160, 64]
    # fc1_len = [160, 160, 160, 160, 128, 128, 160, 192, 307.2, 307.2, 307.2, 307.2]
    # fc2_len = [192, 160, 128, 96, 96, 96, 160, 160, 192, 160, 160, 192]
    # [True, True, True, True, True, True, True, True, True, True, True, True]
    # [True, True, True, True, True, True, True, True, False, False, False, False]
    # [True, True, True, True, True, True, True, True, True, True, True, True]

    # qkv_len = [128, 96, 128, 128, 128, 128, 128, 192, 224, 192, 192, 64]
    # fc1_len = [160, 160, 192, 160, 160, 160, 160, 307.2, 307.2, 307.2, 307.2, 307.2]
    # fc2_len = [192, 160, 160, 128, 128, 128, 160, 224, 192, 307.2, 192, 307.2]
    # [True, True, True, True, True, True, True, True, True, True, True, True]
    # [True, True, True, True, True, True, True, False, False, False, False, False]
    # [True, True, True, True, True, True, True, True, True, False, True, False]

    # qkv_len = [128, 96, 128, 128, 128, 160, 192, 192, 288, 288, 192, 288]
    # fc1_len = [160, 160, 192, 160, 160, 160, 160, 307.2, 307.2, 307.2, 307.2, 307.2]
    # fc2_len = [224, 192, 192, 128, 160, 160, 224, 307.2, 307.2, 307.2, 307.2, 307.2]
    # [True, True, True, True, True, True, True, True, False, False, True, False]
    # [True, True, True, True, True, True, True, False, False, False, False, False]
    # [True, True, True, True, True, True, True, False, False, False, False, False]

    # qkv_len = [128, 96, 160, 160, 160, 160, 192, 288, 288, 288, 288, 288]
    # fc1_len = [160, 192, 224, 192, 160, 192, 224, 307.2, 307.2, 307.2, 307.2, 307.2]
    # fc2_len = [224, 192, 192, 160, 160, 192, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2]
    # [True, True, True, True, True, True, True, False, False, False, False, False]
    # [True, True, True, True, True, True, True, False, False, False, False, False]
    # [True, True, True, True, True, True, False, False, False, False, False, False]


    # qkv_len = [160, 96, 192, 192, 192, 192, 288, 288, 288, 288, 288, 288]
    # fc1_len = [160, 192, 256, 192, 192, 224, 256, 307.2, 307.2, 307.2, 307.2, 307.2]
    # fc2_len = [224, 224, 224, 192, 192, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2]
    # [True, True, True, True, True, True, False, False, False, False, False, False]
    # [True, True, True, True, True, True, True, False, False, False, False, False]
    # [True, True, True, True, True, False, False, False, False, False, False, False]

    root_path = "deit/ffn_output/all_fc_ada/deit_t/"
    qkv_len = [32, 64, 64, 96, 96, 96, 96, 96, 128, 144, 144, 32]
    fc1_len = [128, 128, 153.6, 96, 96, 96, 96, 128, 153.6, 64, 32, 153.6]
    fc2_len = [153.6, 96, 96, 64, 64, 64, 64, 64, 96, 64, 32, 32]
    [True, True, True, True, True, True, True, True, True, False, False, True]
    [True, True, False, True, True, True, True, True, False, True, True, False]
    [False, True, True, True, True, True, True, True, True, True, True, True]

    new_dict = {}
    count = 0
    for k,v in model_dict.items():
        if "fc1.weight" in k or "fc2.weight" in k or "qkv.weight" in k:
            k_split = k.split(".")
            fc_ind =  k_split[-2]

            if fc_ind == "fc1":
                dim = fc1_len[count]
            if fc_ind == "fc2":
                dim = fc2_len[count]
            if fc_ind == "qkv":
                dim = qkv_len[count]

            print(k,count)

            # if dim == 307.2 and fc_ind == "fc1":
            #     continue
            # if dim== 307.2 and fc_ind == "fc2":
            #     count +=1
            #     continue
            # if dim == 288 and fc_ind[:3] == "qkv":
            #     continue
            if dim == 153.6 and fc_ind == "fc1":
                continue
            if dim== 153.6 and fc_ind == "fc2":
                count +=1
                continue
            if dim == 144 and fc_ind[:3] == "qkv":
                continue

            if fc_ind == "fc1" or fc_ind == "fc2":
                k_split[-2] = "uv" + fc_ind[-1]
            elif fc_ind[:3] == "qkv":
                k_split[-2] = "uv1"

            v = torch.load(root_path +  "deit_t_fewshot2_" + fc_ind + "_" + str(count) + "_v.pt",map_location='cpu')
            new_v = v[:dim,:]
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = new_v.transpose(0,1)

            u = torch.load(root_path +  "deit_t_fewshot2_" + fc_ind + "_" + str(count) + "_u.pt",map_location='cpu')
            new_u = u[:,:dim]
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = new_u.transpose(0,1)

            avg = torch.load(root_path +  "deit_t_fewshot2_" + fc_ind + "_" + str(count) + "_avg.pt",map_location='cpu')
            b = (torch.eye(new_u.shape[0]).cpu()- new_u @ new_v) @ avg

            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name, new_v.shape ,u_name, new_u.shape,b_name, b.shape)

            if "fc2.weight" in k:
                count +=1

    model_dict.update(new_dict)
    torch.save(model_dict, root_path +  "deit_t_pca_all_uvb_fewshot2_0.33.pth")
load_model_torch_deit_ada()

def merge_model_torch_deit_ada():
    root_path = "deit/ffn_output/all_fc_ada/deit_t/"
    model_dict = torch.load(root_path +  "deit_t_pca_all_uvb_fewshot2_0.33.pth", map_location='cpu')
    count = 0
    l = list(model_dict.keys())
    for k in l:
        if "fc1.weight" in k or "fc2.weight" in k or "qkv.weight" in k: 
            print(k)
            w = model_dict[k]
            b = model_dict[k[:-6] + "bias"]

            k_split = k.split(".")
            fc_ind = k_split[-2]
            if fc_ind == "fc1" or fc_ind == "fc2":
                k_split[-2] = "uv" + fc_ind[-1]
            elif fc_ind[:3] == "qkv":
                k_split[-2] = "uv1"


            # k_split[-2] = "uv" + fc_ind[-1]
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            if v_name not in model_dict.keys():
                continue
            v = model_dict[v_name]

            new_w = v.transpose(0,1) @ w
            new_b = b @ v
            # print(new_w.shape)
            # print(new_b.shape)
            model_dict[k] = new_w
            model_dict[k[:-6] + "bias"] = new_b

    l = list(model_dict.keys())
    for k in l:
        if ("uv1.u" in k or "uv2.u" in k): 
            print(k)

            w = model_dict[k]

            k_split = k.split(".")
            k_split[-1] = "weight"
            print(".".join(k_split))
            model_dict[".".join(k_split)] = w.T

            k_split[-1] = "b"
            b_name = ".".join(k_split)
            b = model_dict[b_name]
            k_split[-1] = "bias"
            model_dict[".".join(k_split)] = b
    l = list(model_dict.keys())
    for k in l:
        if k[-5:-2] == "uv1"  or k[-5:-2] == "uv2":
            model_dict.pop(k)
    

    torch.save(model_dict, root_path +  "deit_t_pca_all_uvb_fewshot2_0.33_merge_ada.pth")
merge_model_torch_deit_ada()