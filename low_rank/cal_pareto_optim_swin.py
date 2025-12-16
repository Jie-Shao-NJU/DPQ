import numpy as np
import copy
from tqdm import tqdm
import json
import torch
# speed =  [0,24,39,48,78]
# dim = [192, 136, 128, 96, 64]
# params = [0, 115,167, 375, 417]


def dp(value, dim, speed):
    count = 0
    results = {}
    for bi in range(value.shape[1]):
        if results == {}:
            for bj in range(value.shape[0]):
                results[speed[bj]] = [dim[bj], value[bj, bi]]
        else:
            single = {}
            for k,v in results.items():
                for bj in range(value.shape[0]):

                    sum_val =  v[-1] + value[bj, bi] 
                    count = float(k) + speed[bj]

                    if count not in single.keys():
                        single[count] = v[:-1] + [dim[bj], sum_val]
                    elif sum_val < single[count][-1]:
                        single[count] = v[:-1] + [dim[bj], sum_val]
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


def gen_final_results(results_qkv_l0, results_qkv_l1, results_qkv_l2, results_qkv_l3, results_ffn_l0, results_ffn_l1, results_ffn_l2, results_ffn_l3):
    f = {}
    results = [results_qkv_l0, results_qkv_l1, results_qkv_l2, results_qkv_l3, results_ffn_l0, results_ffn_l1, results_ffn_l2, results_ffn_l3]

    for results_i in tqdm(results):
        if f == {}:
            f = results_i
        else:
            single = {}
            for k1,v1 in results_i.items():
                for k2, v2 in f.items():
                    sum_block = k1 + k2
                    count = v1[-1] + v2[-1]
                    if sum_block not in single.keys():
                        single[sum_block] =  [v2] + [v1] + [count]
                    elif count < single[sum_block][-1]:
                        single[sum_block] = [v2] + [v1] + [count]
            f = single
    # print(f)
    return f


def get_qkv_results(params, dim, value, layer,  block):
    for b in range( block):
        f = 3
        ind_value = b
        file_name = "swin/checkpoint_pca_all_mimicking_t/"+ str(layer) + "_" + str(b) + "_" +str(f) + ".txt"
        files = np.loadtxt(file_name,delimiter=',')
        param_kl_dict = dict(zip(files[:,0].astype(np.int32),files[:,1]))
        # print(param_kl_dict)
        for i in range(len(dim)):
            if dim[i] in param_kl_dict.keys():
                value[i, ind_value] = param_kl_dict[dim[i]] 
        value[0] = 0
    print(value)
    results = dp(value, dim, params)
    results = remove_unreason_item(results)
    return results

def get_ffn_results(params, dim, value, layer,  block):
    for b in range(block):
        for f in range(1,3):
            ind_value = 2*b +f -1
            file_name = "swin/checkpoint_pca_all_mimicking_t/"+ str(layer) + "_" +  str(b) + "_" +str(f) + ".txt"
            files = np.loadtxt(file_name,delimiter=',')
            param_kl_dict = dict(zip(files[:,0].astype(np.int32),files[:,1]))
            # print(param_kl_dict)
            for i in range(len(dim)):
                if dim[i] in param_kl_dict.keys():
                    value[i, ind_value] = param_kl_dict[dim[i]] 
    print(value)
    results = dp(value, dim, params)
    results = remove_unreason_item(results)
    return results

# qkv_params_l0 = [110592, 98304, 73728, 49152, 24576]
# qkv_dim_l0 = [144, 128, 96, 64, 32]
# layer_l0,  block_l0 = 0,  2
# qkv_value_l0 = np.zeros([len(qkv_dim_l0), block_l0])
# results_qkv_l0 = get_qkv_results(qkv_params_l0, qkv_dim_l0, qkv_value_l0, layer_l0,  block_l0)
# print(results_qkv_l0)

# qkv_params_l1 = [442368, 393216, 344064, 294912, 245760, 196608, 147456, 98304, 49152]
# qkv_dim_l1 = [288, 256, 224, 192, 160, 128, 96, 64, 32]
# layer_l1,  block_l1 = 1,  2
# qkv_value_l1 = np.zeros([len(qkv_dim_l1), block_l1])
# results_qkv_l1 = get_qkv_results(qkv_params_l1, qkv_dim_l1, qkv_value_l1, layer_l1,  block_l1)
# print(results_qkv_l1)

# qkv_params_l2 = [1769472, 1671168, 1572864, 1474560, 1376256, 1277952, 1179648, 1081344, 983040, 884736, 786432, 688128, 589824, 491520, 393216, 294912, 196608, 98304]
# qkv_dim_l2 = [576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192,160,128, 96, 64, 32]
# layer_l2,  block_l2 = 2,  18
# qkv_value_l2 = np.zeros([len(qkv_dim_l2), block_l2])
# results_qkv_l2 = get_qkv_results(qkv_params_l2, qkv_dim_l2, qkv_value_l2, layer_l2,  block_l2)
# print(results_qkv_l2)

# qkv_params_l3 = [7077888, 6881280, 6684672, 6488064, 6291456, 6094848, 5898240, 5701632, 5505024, 5308416, 5111808, 4915200, 4718592, 4521984, 4325376, 4128768, 3932160, 3735552, 3538944, 3342336, 3145728, 2949120, 2752512, 2555904, 2359296, 2162688, 1966080, 1769472, 1572864, 1376256, 1179648, 983040, 786432, 589824, 393216, 196608]
# qkv_dim_l3 = [1152, 1120, 1088, 1056, 1024, 992, 960, 928, 896, 864, 832, 800, 768, 736, 704, 672, 640, 608, 576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32]
# layer_l3,  block_l3 = 3,  2
# qkv_value_l3 = np.zeros([len(qkv_dim_l3), block_l3])
# results_qkv_l3 = get_qkv_results(qkv_params_l3, qkv_dim_l3, qkv_value_l3, layer_l3,  block_l3)
# print(results_qkv_l3)

# ffn_params_l0 = [147456, 122880, 92160, 61440, 30720]
# ffn_dim_l0 =  [153.6, 128, 96, 64, 32]
# layer_l0,  block_l0 = 0,  2
# ffn_value_l0 = np.zeros([len(ffn_dim_l0), block_l0 * 2])
# results_ffn_l0 = get_ffn_results(ffn_params_l0, ffn_dim_l0, ffn_value_l0, layer_l0,  block_l0)
# print(results_ffn_l0)

# ffn_params_l1 = [589824, 552960, 491520, 430080, 368640, 307200, 245760, 184320, 122880, 61440]
# ffn_dim_l1 = [307.2, 288, 256, 224, 192, 160, 128, 96, 64, 32]
# layer_l1,  block_l1 = 1,  2
# ffn_value_l1 = np.zeros([len(ffn_dim_l1), block_l1* 2])
# results_ffn_l1 = get_ffn_results(ffn_params_l1, ffn_dim_l1, ffn_value_l1, layer_l1,  block_l1)
# print(results_ffn_l1)

# ffn_params_l2 = [2359296, 2334720, 2211840, 2088960, 1966080, 1843200, 1720320, 1597440, 1474560, 1351680, 1228800, 1105920, 983040, 860160, 737280, 614400, 491520, 368640, 245760, 122880]
# ffn_dim_l2 = [614.4, 608, 576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32]
# layer_l2,  block_l2 = 2,  18
# ffn_value_l2 = np.zeros([len(ffn_dim_l2), block_l2* 2])
# results_ffn_l2 = get_ffn_results(ffn_params_l2, ffn_dim_l2, ffn_value_l2, layer_l2,  block_l2)
# print(results_ffn_l2)

# ffn_params_l3 = [9437184, 9338880, 9093120, 8847360, 8601600, 8355840, 8110080, 7864320, 7618560, 7372800, 7127040, 6881280, 6635520, 6389760, 6144000, 5898240, 5652480, 5406720, 5160960, 4915200, 4669440, 4423680, 4177920, 3932160, 3686400, 3440640, 3194880, 2949120, 2703360, 2457600, 2211840, 1966080, 1720320, 1474560, 1228800, 983040, 737280, 491520, 245760]
# ffn_dim_l3 = [1228.8, 1216, 1184, 1152, 1120, 1088, 1056, 1024, 992, 960, 928, 896, 864, 832, 800, 768, 736, 704, 672, 640, 608, 576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32]
# layer_l3,  block_l3 = 3,  2
# ffn_value_l3 = np.zeros([len(ffn_dim_l3), block_l3* 2])
# results_ffn_l3 = get_ffn_results(ffn_params_l3, ffn_dim_l3, ffn_value_l3, layer_l3,  block_l3)
# print(results_ffn_l3)

# results = gen_final_results(results_qkv_l0, results_qkv_l1, results_qkv_l2, results_qkv_l3, results_ffn_l0, results_ffn_l1, results_ffn_l2, results_ffn_l3)
# results = remove_unreason_item(results)
# print(results)


# qkv_params_l0 = [49152, 32768, 16384]
# qkv_dim_l0 = [96, 64, 32]
# layer_l0,  block_l0 = 0,  2
# qkv_value_l0 = np.zeros([len(qkv_dim_l0), block_l0])
# results_qkv_l0 = get_qkv_results(qkv_params_l0, qkv_dim_l0, qkv_value_l0, layer_l0,  block_l0)
# print(results_qkv_l0)

# qkv_params_l1 = [196608, 163840, 131072, 98304, 65536, 32768]
# qkv_dim_l1 = [192,160,128, 96, 64, 32]
# layer_l1,  block_l1 = 1,  2
# qkv_value_l1 = np.zeros([len(qkv_dim_l1), block_l1])
# results_qkv_l1 = get_qkv_results(qkv_params_l1, qkv_dim_l1, qkv_value_l1, layer_l1,  block_l1)
# print(results_qkv_l1)

# qkv_params_l2 = [786432, 720896, 655360, 589824, 524288, 458752, 393216, 327680, 262144, 196608, 131072, 65536]
# qkv_dim_l2 = [384, 352, 320, 288, 256, 224, 192,160,128, 96, 64, 32]
# layer_l2,  block_l2 = 2,  18
# qkv_value_l2 = np.zeros([len(qkv_dim_l2), block_l2])
# results_qkv_l2 = get_qkv_results(qkv_params_l2, qkv_dim_l2, qkv_value_l2, layer_l2,  block_l2)
# print(results_qkv_l2)

# qkv_params_l3 = [3145728, 3014656, 2883584, 2752512, 2621440, 2490368, 2359296, 2228224, 2097152, 1966080, 1835008, 1703936, 1572864, 1441792, 1310720, 1179648, 1048576, 917504, 786432, 655360, 524288, 393216, 262144, 131072]
# qkv_dim_l3 = [768, 736, 704, 672, 640, 608, 576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32]
# layer_l3,  block_l3 = 3,  2
# qkv_value_l3 = np.zeros([len(qkv_dim_l3), block_l3])
# results_qkv_l3 = get_qkv_results(qkv_params_l3, qkv_dim_l3, qkv_value_l3, layer_l3,  block_l3)
# print(results_qkv_l3)

# ffn_params_l0 = [65536, 61440, 40960, 20480]
# ffn_dim_l0 = [102.4, 96, 64, 32]
# layer_l0,  block_l0 = 0,  2
# ffn_value_l0 = np.zeros([len(ffn_dim_l0), block_l0 * 2])
# results_ffn_l0 = get_ffn_results(ffn_params_l0, ffn_dim_l0, ffn_value_l0, layer_l0,  block_l0)
# print(results_ffn_l0)

# ffn_params_l1 = [262144, 245760, 204800, 163840, 122880, 81920, 40960]
# ffn_dim_l1 = [204.8, 192,160,128, 96, 64, 32]
# layer_l1,  block_l1 = 1,  2
# ffn_value_l1 = np.zeros([len(ffn_dim_l1), block_l1* 2])
# results_ffn_l1 = get_ffn_results(ffn_params_l1, ffn_dim_l1, ffn_value_l1, layer_l1,  block_l1)
# print(results_ffn_l1)

# ffn_params_l2 = [1048576, 983040, 901120, 819200, 737280, 655360, 573440, 491520, 409600, 327680, 245760, 163840, 81920]
# ffn_dim_l2 = [409.6, 384, 352, 320, 288, 256, 224, 192,160,128, 96, 64, 32]
# layer_l2,  block_l2 = 2,  18
# ffn_value_l2 = np.zeros([len(ffn_dim_l2), block_l2* 2])
# results_ffn_l2 = get_ffn_results(ffn_params_l2, ffn_dim_l2, ffn_value_l2, layer_l2,  block_l2)
# print(results_ffn_l2)

# ffn_params_l3 = [4194304, 4096000, 3932160, 3768320, 3604480, 3440640, 3276800, 3112960, 2949120, 2785280, 2621440, 2457600, 2293760, 2129920, 1966080, 1802240, 1638400, 1474560, 1310720, 1146880, 983040, 819200, 655360, 491520, 327680, 163840]
# ffn_dim_l3 = [819.2, 800, 768, 736, 704, 672, 640, 608, 576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32]
# layer_l3,  block_l3 = 3,  2
# ffn_value_l3 = np.zeros([len(ffn_dim_l3), block_l3* 2])
# results_ffn_l3 = get_ffn_results(ffn_params_l3, ffn_dim_l3, ffn_value_l3, layer_l3,  block_l3)
# print(results_ffn_l3)


# results = gen_final_results(results_qkv_l0, results_qkv_l1, results_qkv_l2, results_qkv_l3, results_ffn_l0, results_ffn_l1, results_ffn_l2, results_ffn_l3)
# results = remove_unreason_item(results)
# print(results)


# qkv_params_l0 = [27648, 24576, 12288]
# qkv_dim_l0 = [72, 64, 32]
# layer_l0,  block_l0 = 0,  2
# qkv_value_l0 = np.zeros([len(qkv_dim_l0), block_l0])
# results_qkv_l0 = get_qkv_results(qkv_params_l0, qkv_dim_l0, qkv_value_l0, layer_l0,  block_l0)
# print(results_qkv_l0)

# qkv_params_l1 = [110592, 98304, 73728, 49152, 24576]
# qkv_dim_l1 = [144,128, 96, 64, 32]
# layer_l1,  block_l1 = 1,  2
# qkv_value_l1 = np.zeros([len(qkv_dim_l1), block_l1])
# results_qkv_l1 = get_qkv_results(qkv_params_l1, qkv_dim_l1, qkv_value_l1, layer_l1,  block_l1)
# print(results_qkv_l1)

# qkv_params_l2 = [442368, 393216, 344064, 294912, 245760, 196608, 147456, 98304, 49152]
# qkv_dim_l2 = [288, 256, 224, 192,160,128, 96, 64, 32]
# layer_l2,  block_l2 = 2,  18
# qkv_value_l2 = np.zeros([len(qkv_dim_l2), block_l2])
# results_qkv_l2 = get_qkv_results(qkv_params_l2, qkv_dim_l2, qkv_value_l2, layer_l2,  block_l2)
# print(results_qkv_l2)

# qkv_params_l3 = [1769472, 1671168, 1572864, 1474560, 1376256, 1277952, 1179648, 1081344, 983040, 884736, 786432, 688128, 589824, 491520, 393216, 294912, 196608, 98304]
# qkv_dim_l3 = [576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32]
# layer_l3,  block_l3 = 3,  2
# qkv_value_l3 = np.zeros([len(qkv_dim_l3), block_l3])
# results_qkv_l3 = get_qkv_results(qkv_params_l3, qkv_dim_l3, qkv_value_l3, layer_l3,  block_l3)
# print(results_qkv_l3)

# ffn_params_l0 = [36864.0, 30720, 15360]
# ffn_dim_l0 = [76.8, 64, 32]
# layer_l0,  block_l0 = 0,  2
# ffn_value_l0 = np.zeros([len(ffn_dim_l0), block_l0 * 2])
# results_ffn_l0 = get_ffn_results(ffn_params_l0, ffn_dim_l0, ffn_value_l0, layer_l0,  block_l0)
# print(results_ffn_l0)

# ffn_params_l1 = [147456.0, 122880, 92160, 61440, 30720]
# ffn_dim_l1 = [153.6, 128, 96, 64, 32]
# layer_l1,  block_l1 = 1,  2
# ffn_value_l1 = np.zeros([len(ffn_dim_l1), block_l1* 2])
# results_ffn_l1 = get_ffn_results(ffn_params_l1, ffn_dim_l1, ffn_value_l1, layer_l1,  block_l1)
# print(results_ffn_l1)

# ffn_params_l2 = [589824.0, 552960, 491520, 430080, 368640, 307200, 245760, 184320, 122880, 61440]
# ffn_dim_l2 = [307.2, 288, 256, 224, 192,160,128, 96, 64, 32]
# layer_l2,  block_l2 = 2,  18
# ffn_value_l2 = np.zeros([len(ffn_dim_l2), block_l2* 2])
# results_ffn_l2 = get_ffn_results(ffn_params_l2, ffn_dim_l2, ffn_value_l2, layer_l2,  block_l2)
# print(results_ffn_l2)

# ffn_params_l3 = [2359296.0, 2334720, 2211840, 2088960, 1966080, 1843200, 1720320, 1597440, 1474560, 1351680, 1228800, 1105920, 983040, 860160, 737280, 614400, 491520, 368640, 245760, 122880]
# ffn_dim_l3 = [614.4, 608, 576, 544, 512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32]
# layer_l3,  block_l3 = 3,  2
# ffn_value_l3 = np.zeros([len(ffn_dim_l3), block_l3* 2])
# results_ffn_l3 = get_ffn_results(ffn_params_l3, ffn_dim_l3, ffn_value_l3, layer_l3,  block_l3)
# print(results_ffn_l3)


# results = gen_final_results(results_qkv_l0, results_qkv_l1, results_qkv_l2, results_qkv_l3, results_ffn_l0, results_ffn_l1, results_ffn_l2, results_ffn_l3)
# results = remove_unreason_item(results)
# print(results)

def get_dims(attn_dim, ffn_dim):
    # attn_no_use_uv_dim = [72, 144, 288, 576]
    # ffn_no_use_uv_dim = [76.8, 153.6, 307.2, 614.4]


    attn_no_use_uv_dim = [96, 192, 384, 768]
    ffn_no_use_uv_dim = [102.4, 204.8, 409.6, 819.2]

    # attn_no_use_uv_dim = [144, 288, 576, 1152]
    # ffn_no_use_uv_dim = [153.6, 307.2, 614.4, 1228.8]

    def gen_use_uv_mark(dim, no_use_uv_dim):
        use_uv_mark = []

        for i in range(len(dim)):
            layer_i = dim[i]
            layer_use_uv_mark = []
            for dims in layer_i:
                if dims != no_use_uv_dim[i]:
                    layer_use_uv_mark.append(True)
                else:
                    layer_use_uv_mark.append(False)
            use_uv_mark.append(layer_use_uv_mark)
        return use_uv_mark
    
    attn_no_use_uv_mark = gen_use_uv_mark(attn_dim, attn_no_use_uv_dim)

    fc1_dim = []
    fc2_dim = []
    for i in range(len(ffn_dim)):
        ffn_layer_i = ffn_dim[i]
        fc1_layer_i =[]
        fc2_layer_i =[]
        for j in range(len(ffn_layer_i)):
            if j % 2 == 0:
                fc1_layer_i.append(ffn_layer_i[j])
            else:
                fc2_layer_i.append(ffn_layer_i[j])
        fc1_dim.append(fc1_layer_i)
        fc2_dim.append(fc2_layer_i)
    fc1_no_use_uv_mark = gen_use_uv_mark(fc1_dim, ffn_no_use_uv_dim)
    fc2_no_use_uv_mark = gen_use_uv_mark(fc2_dim, ffn_no_use_uv_dim)

    print(attn_dim)
    print(fc1_dim)
    print(fc2_dim)

    print(attn_no_use_uv_mark)
    print(fc1_no_use_uv_mark)
    print(fc2_no_use_uv_mark)
## S
# 33%

# attn_dim  = [[32, 72], [144, 144], [160, 160, 128, 160, 128, 128, 192, 160, 160, 128, 160, 128, 160, 160, 160, 128, 160, 128], [160, 224,]]
# ffn_dim =[ [64, 76.8, 76.8, 76.8],  [153.6, 153.6, 153.6, 153.6], [192, 224, 192, 192, 192, 160, 192, 128, 224, 160, 192, 160, 160, 128, 192, 160, 160, 160, 192, 192, 224, 192, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2,], [288, 416, 96, 384]]

# 25%
# attn_dim  = [[32, 72], [144, 144], [160, 160, 160, 160, 288, 160, 288, 160, 288, 192, 288, 160, 288, 288, 288, 128, 160, 160], [160, 224]]
# ffn_dim = [[64, 76.8, 76.8, 76.8], [153.6, 153.6, 153.6, 153.6] , [192, 224, 192, 192, 192, 160, 192, 128, 224, 192, 224, 192, 192, 307.2, 192, 307.2, 224, 307.2, 224, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [320, 448, 96, 416]]

# 20%
# attn_dim  =[ [32, 72], [144, 144],   [160, 160, 160, 160, 288, 192, 288, 192, 288, 288, 288, 288, 288, 288, 288, 160, 192, 160], [192, 256]]
# ffn_dim = [[64, 76.8, 76.8, 76.8], [153.6, 153.6, 153.6, 153.6],[192, 224, 224, 224, 224, 192, 224, 160, 224, 307.2, 307.2, 307.2, 307.2, 307.2, 192, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [320, 480, 96, 448]]


# attn_dim  = [[96, 96], [192, 192], [192, 192, 192, 160, 192, 160, 384, 192, 224, 224, 256, 224, 192, 224, 256, 224, 192, 128], [32, 64]]

# ffn_dim =  [[102.4, 102.4, 102.4, 102.4], [204.8, 204.8, 204.8, 204.8], [224, 352, 256, 256, 256, 288, 288, 256, 224, 288, 256, 256, 256, 256, 256, 288, 409.6, 409.6, 288, 288, 320, 224, 409.6, 224, 409.6, 224, 409.6, 256, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 288, 409.6],  [288, 704, 544, 672]]

# get_dims(attn_dim, ffn_dim)

## b
# 33%
# attn_dim = [ [64, 96], [160, 192], [192, 192, 192, 160, 256, 192, 256, 192, 224, 224, 224, 256, 256, 256, 256, 224, 224, 160], [192, 256]  ]
# ffn_dim = [ [102.4, 102.4, 102.4, 102.4], [204.8, 204.8, 204.8, 204.8], [224, 256, 224, 224, 224, 224, 256, 224, 256, 224, 256, 224, 224, 288, 224, 288, 256, 288, 224, 256, 288, 288, 288, 320, 409.6, 320, 409.6, 320, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 320, 409.6], [352, 544, 544, 480]]

# attn_dim = [ [96, 96], [192, 192],  [192, 192, 256, 288, 384, 384, 384, 384, 384, 384, 384, 256, 384, 384, 384, 256, 224, 128], [32, 64]  ]
# ffn_dim = [[102.4, 102.4, 102.4, 102.4],  [204.8, 204.8, 204.8, 204.8], [224, 409.6, 288, 320, 288, 320, 320, 288, 320, 320, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 288, 409.6, 320, 409.6, 288, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6], [352, 736, 608, 672]]
# get_dims(attn_dim, ffn_dim)

# 25%

# attn_dim = [[64, 96], [160, 192], [192, 192, 224, 192, 256, 256, 288, 224, 288, 256, 384, 256, 384, 384, 384, 256, 256, 192], [224, 288]]
# ffn_dim = [[96, 102.4, 102.4, 102.4], [204.8, 204.8, 204.8, 204.8], [224, 288, 224, 256, 288, 224, 288, 256, 288, 320, 288, 288, 288, 320, 288, 352, 288, 320, 288, 320, 320, 320, 352, 320, 409.6, 320, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6], [384, 576, 576, 544]]

# 20% 
# attn_dim = [[64, 96], [160, 192], [192, 224, 256, 288, 288, 256, 384, 256, 288, 288, 384, 384, 384, 384, 384, 288, 288, 224], [224, 288]]
# ffn_dim = [[96, 102.4, 102.4, 102.4], [204.8, 204.8, 204.8, 204.8], [256, 288, 256, 288, 288, 256, 288, 288, 288, 320, 320, 320, 320, 409.6, 320, 409.6, 320, 409.6, 320, 409.6, 320, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6], [416, 640, 576, 576]]

## l
# 33%
# attn_dim = [[64, 144], [192, 288],[160, 192, 256, 224, 288, 256, 288, 288, 352, 352, 384, 320, 384, 320, 384, 320, 576, 320], [480, 608] ]
# ffn_dim = [[96, 153.6, 153.6, 153.6], [288, 307.2, 307.2, 256], [224, 256, 256, 224, 256, 192, 288, 256, 320, 256, 352, 256, 384, 320, 352, 320, 352, 320, 384, 352, 384, 352, 352, 384, 384, 416, 448, 448, 480, 480, 480, 416, 614.4, 480, 614.4, 448], [704, 1228.8, 1228.8, 992] ]

# 25%
# attn_dim = [[64, 144],[192, 288],[160, 256, 256, 256, 288, 288, 288, 320, 384, 352, 384, 352, 416, 320, 576, 416, 576, 416],[544, 672]]
# ffn_dim = [[96, 153.6, 153.6, 153.6],[307.2, 307.2, 307.2, 307.2], [224, 256, 288, 288, 256, 256, 320, 288, 320, 288, 384, 352, 416, 352, 352, 384, 416, 416, 384, 384, 448, 416, 448, 480, 448, 480, 480, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 480],[800, 1228.8, 1228.8, 1228.8]]

# 20%
# attn_dim = [[64, 144], [224, 288], [160, 256, 352, 288, 352, 352, 352, 384, 416, 384, 576, 384, 576, 416, 576, 576, 576, 576], [576, 736]]
# ffn_dim = [[96, 153.6, 153.6, 153.6],[307.2, 307.2, 307.2, 307.2], [256, 256, 288, 320, 288, 320, 352, 320, 352, 320, 416, 384, 416, 384, 448, 416, 448, 448, 480, 416, 448, 480, 480, 614.4, 512, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4],  [800, 1228.8, 1228.8, 1228.8]]

# 12.5%
# attn_dim = [[64, 144],[224, 288],[192, 256, 352, 352, 416, 448, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576],[640, 864]]
# ffn_dim = [[128, 153.6, 153.6, 153.6],[ 307.2, 307.2, 307.2, 307.2], [288, 320, 320, 384, 288, 352, 384, 384, 448, 416, 512, 448, 512, 480, 614.4, 480, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4], [864, 1228.8, 1228.8, 1228.8] ]


def load_model_torch_deit_ada():
    # model_dict = torch.load("pretrainmodel/swin_small_patch4_window7_224.pth",map_location='cpu')
    model_dict = torch.load("pretrainmodel/swin_base_patch4_window7_224.pth",map_location='cpu')
    # model_dict = torch.load("pretrainmodel/swin_large_patch4_window7_224_22kto1k.pth",map_location='cpu')
    model_dict = model_dict['model']
    u_list = []
    v_list = []


    # root_path = "swin/ffn_output/all_fc_ada/swin_s/"
    # qkv_len =[[32, 72], [144, 144], [160, 160, 128, 160, 128, 128, 192, 160, 160, 128, 160, 128, 160, 160, 160, 128, 160, 128], [160, 224]]
    # fc1_len =[[64, 76.8], [153.6, 153.6], [192, 192, 192, 192, 224, 192, 160, 192, 160, 192, 224, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [288, 96]]
    # fc2_len = [[76.8, 76.8], [153.6, 153.6], [224, 192, 160, 128, 160, 160, 128, 160, 160, 192, 192, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [416, 384]]
    # qkv_use_uv =[[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv =[[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv =[[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False], [True, True]]

    proj_len = [[64,64],[128,128],[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],[512,512]]
    proj_use_uv = [[False, False], [False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False]]

    # qkv_len = [[32, 72], [144, 144], [160, 160, 160, 160, 288, 160, 288, 160, 288, 192, 288, 160, 288, 288, 288, 128, 160, 160], [160, 224]]
    # fc1_len = [[64, 76.8], [153.6, 153.6], [192, 192, 192, 192, 224, 224, 192, 192, 224, 224, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [320, 96]]
    # fc2_len = [[76.8, 76.8], [153.6, 153.6], [224, 192, 160, 128, 192, 192, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [448, 416]]
    # qkv_use_uv = [[True, False], [False, False], [True, True, True, True, False, True, False, True, False, True, False, True, False, False, False, True, True, True], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False], [True, True]]


    # qkv_len = [[32, 72], [144, 144], [160, 160, 160, 160, 288, 192, 288, 192, 288, 288, 288, 288, 288, 288, 288, 160, 192, 160], [192, 256]]
    # fc1_len = [[64, 76.8], [153.6, 153.6], [192, 224, 224, 224, 224, 307.2, 307.2, 192, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [320, 96]]
    # fc2_len = [[76.8, 76.8], [153.6, 153.6], [224, 224, 192, 160, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [480, 448]]
    # qkv_use_uv = [[True, False], [False, False], [True, True, True, True, False, True, False, True, False, False, False, False, False, False, False, True, True, True], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, False, False, True, False, False, False, False, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [True, True]]

    # qkv_len = [[32, 72], [144, 144], [160, 160, 160, 160, 288, 288, 288, 192, 288, 288, 288, 288, 288, 288, 288, 288, 288, 160], [192, 256]]
    # fc1_len = [[76.8, 76.8], [153.6, 153.6], [192, 224, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [352, 96]]
    # fc2_len = [[76.8, 76.8], [153.6, 153.6], [307.2, 307.2, 224, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [614.4, 448]]
    # qkv_use_uv = [[True, False], [False, False], [True, True, True, True, False, False, False, True, False, False, False, False, False, False, False, False, False, True], [True, True]]
    # fc1_use_uv = [[False, False], [False, False], [True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, True]]

    root_path = "swin/ffn_output/all_fc_ada/swin_b/"

    # qkv_len =  [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # fc1_len = [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # fc2_len = [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # proj_len = [[64,64],[128,128],[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],[512,512]]
    # proj_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # qkv_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc2_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]

    qkv_len =  [[64, 96], [160, 192], [192, 192, 192, 160, 256, 192, 256, 192, 224, 224, 224, 256, 256, 256, 256, 224, 224, 160], [192, 256]]
    fc1_len = [[768, 768], [768, 768], [224, 224, 224, 256, 256, 256, 224, 224, 256, 224, 288, 288, 768, 768, 768, 768, 768, 320], [352, 544]]
    fc2_len = [[768, 768], [768, 768], [256, 224, 224, 224, 224, 224, 288, 288, 288, 256, 288, 320, 320, 320, 768, 768, 768, 768], [544, 480]]
    qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    fc1_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, True], [True, True]]
    fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False], [True, True]]

    # qkv_len = [[64, 96], [160, 192], [192, 192, 224, 192, 256, 256, 288, 224, 288, 256, 384, 256, 384, 384, 384, 256, 256, 192], [224, 288]]
    # fc1_len = [[96, 102.4], [204.8, 204.8], [224, 224, 288, 288, 288, 288, 288, 288, 288, 288, 320, 352, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6], [384, 576]]
    # fc2_len = [[102.4, 102.4], [204.8, 204.8], [288, 256, 224, 256, 320, 288, 320, 352, 320, 320, 320, 320, 320, 409.6, 409.6, 409.6, 409.6, 409.6], [576, 544]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, False, True, False, False, False, True, True, True], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False], [True, True]]

    # qkv_len = [[64, 96], [160, 192], [192, 224, 256, 288, 288, 256, 384, 256, 288, 288, 384, 384, 384, 384, 384, 288, 288, 224], [224, 288]]
    # fc1_len = [[96, 102.4], [204.8, 204.8], [256, 256, 288, 288, 288, 320, 320, 320, 320, 320, 320, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6], [416, 576]]
    # fc2_len = [[102.4, 102.4], [204.8, 204.8], [288, 288, 256, 288, 320, 320, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6], [640, 576]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, False, True, True, True, False, False, False, False, False, True, True, True], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False], [True, True]]

    # root_path = "swin/ffn_output/all_fc_ada/swin_l/"

    # qkv_len = [[64, 144], [192, 288], [160, 192, 256, 224, 288, 256, 288, 288, 352, 352, 384, 320, 384, 320, 384, 320, 576, 320], [480, 608]]
    # fc1_len = [[96, 153.6], [288, 307.2], [224, 256, 256, 288, 320, 352, 384, 352, 352, 384, 384, 352, 384, 448, 480, 480, 614.4, 614.4], [704, 1228.8]]
    # fc2_len = [[153.6, 153.6], [307.2, 256], [256, 224, 192, 256, 256, 256, 320, 320, 320, 352, 352, 384, 416, 448, 480, 416, 480, 448], [1228.8, 992]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True], [True, True]]
    # fc1_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False], [True, False]]
    # fc2_use_uv = [[False, False], [False, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [False, True]]

    # qkv_len = [[64, 144], [192, 288], [160, 256, 256, 256, 288, 288, 288, 320, 384, 352, 384, 352, 416, 320, 576, 416, 576, 416], [544, 672]]
    # fc1_len = [[96, 153.6], [307.2, 307.2], [224, 288, 256, 320, 320, 384, 416, 352, 416, 384, 448, 448, 448, 480, 614.4, 614.4, 614.4, 614.4], [800, 1228.8]]
    # fc2_len = [[153.6, 153.6], [307.2, 307.2], [256, 288, 256, 288, 288, 352, 352, 384, 416, 384, 416, 480, 480, 614.4, 614.4, 614.4, 614.4, 480], [1228.8, 1228.8]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False], [True, False]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, True], [False, False]]


    # qkv_len = [[64, 144], [224, 288], [160, 256, 352, 288, 352, 352, 352, 384, 416, 384, 576, 384, 576, 416, 576, 576, 576, 576], [576, 736]]
    # fc1_len = [[96, 153.6], [307.2, 307.2], [256, 288, 288, 352, 352, 416, 416, 448, 448, 480, 448, 480, 512, 614.4, 614.4, 614.4, 614.4, 614.4], [800, 1228.8]]
    # fc2_len = [[153.6, 153.6], [307.2, 307.2], [256, 320, 320, 320, 320, 384, 384, 416, 448, 416, 480, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4], [1228.8, 1228.8]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, False, True, False, True, False, False, False, False], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False], [True, False]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False], [False, False]]

    # qkv_len = [[64, 144], [224, 288], [192, 256, 352, 352, 416, 448, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576], [640, 864]]
    # fc1_len = [[128, 153.6], [307.2, 307.2], [288, 320, 288, 384, 448, 512, 512, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4], [864, 1228.8]]
    # fc2_len = [[153.6, 153.6], [307.2, 307.2], [320, 384, 352, 384, 416, 448, 480, 480, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4], [1228.8, 1228.8]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False], [True, False]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False], [False, False]]

    new_dict = {}
    count = 0
    for k,v in model_dict.items():
        if "fc1.weight" in k or "fc2.weight" in k or "qkv.weight" in k  or "attn.proj.weight" in k:
        # if "fc1.weight" in k or "fc2.weight" in k or "qkv.weight" in k:
            k_split = k.split(".")

            fc_ind =  k_split[-2]
            layer_ind = int(k_split[1])
            block_ind = int(k_split[3])

            if fc_ind == "qkv":
                dim = qkv_len[layer_ind][block_ind]
                if not qkv_use_uv[layer_ind][block_ind]:
                    continue
            if fc_ind == "fc1":
                dim = fc1_len[layer_ind][block_ind]
                if not fc1_use_uv[layer_ind][block_ind]:
                    continue

            if fc_ind == "fc2":
                dim = fc2_len[layer_ind][block_ind]
                if not fc2_use_uv[layer_ind][block_ind]:
                    count +=1
                    continue
            if fc_ind == "proj":
                dim = proj_len[layer_ind][block_ind]
                if not proj_use_uv[layer_ind][block_ind]:
                    continue

            print(k,count)

            if fc_ind == "fc1" or fc_ind == "fc2":
                k_split[-2] = "uv" + fc_ind[-1]
            elif fc_ind[:3] == "qkv":
                k_split[-2] = "uv1"
            elif fc_ind == "proj":
                k_split[-2] = "uv2"

            v = torch.load(root_path +  "swin_b_fewshot2_" + fc_ind + "_" + str(count) + "_v.pt",map_location='cpu')
            new_v = v[:dim,:]
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = new_v.transpose(0,1)

            u = torch.load(root_path +  "swin_b_fewshot2_" + fc_ind + "_" + str(count) + "_u.pt",map_location='cpu')
            new_u = u[:,:dim]
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = new_u.transpose(0,1)

            avg = torch.load(root_path +  "swin_b_fewshot2_" + fc_ind + "_" + str(count) + "_avg.pt",map_location='cpu')
            b = (torch.eye(new_u.shape[0]).cpu()- new_u @ new_v) @ avg

            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name, new_v.shape ,u_name, new_u.shape,b_name, b.shape)

            if "fc2.weight" in k:
                count +=1

    model_dict.update(new_dict)
    torch.save(model_dict, root_path +  "swin_b_pca_all_uvb_fewshot2_0.33.pth")
load_model_torch_deit_ada()

def merge_model_torch_deit_ada():
    root_path = "swin/ffn_output/all_fc_ada/swin_b/"
    model_dict = torch.load(root_path +  "swin_b_pca_all_uvb_fewshot2_0.33.pth", map_location='cpu')
    count = 0
    l = list(model_dict.keys())
    for k in l:
        if "fc1.weight" in k or "fc2.weight" in k or "qkv.weight" in k or "attn.proj.weight" in k: 
            print(k)
            w = model_dict[k]
            b = model_dict[k[:-6] + "bias"]

            k_split = k.split(".")
            fc_ind = k_split[-2]
            if fc_ind == "fc1" or fc_ind == "fc2":
                k_split[-2] = "uv" + fc_ind[-1]
            elif fc_ind == "qkv":
                k_split[-2] = "uv1"
            elif fc_ind == "proj":
                k_split[-2] = "uv2"


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
    

    torch.save(model_dict, root_path +  "swin_b_pca_all_uvb_fewshot2_0.33_merge_ada.pth")
merge_model_torch_deit_ada()



def divide_svd():
    root_path = "swin/ffn_output/all_fc_ada/swin_b/"
    model_dict = torch.load("pretrainmodel/swin_base_patch4_window7_224.pth",map_location='cpu')
    # root_path = "swin/ffn_output/all_fc_ada/swin_l/"
    # model_dict = torch.load("pretrainmodel/swin_large_patch4_window7_224_22kto1k.pth",map_location='cpu')

    # root_path = "swin/ffn_output/all_fc_ada/swin_s/"
    # model_dict = torch.load("pretrainmodel/swin_small_patch4_window7_224.pth",map_location='cpu')

    model_dict = model_dict['model']
    # qkv_len =  [[64, 64], [128, 128], [224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224], [416, 416]]
    # fc1_len = [[64, 64], [128, 128], [224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224], [416, 416]]
    # fc2_len = [[64, 64], [128, 128], [224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224], [416, 416]]
    # proj_len = [[64, 64], [128, 128], [224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224], [416, 416]]
    # qkv_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc2_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # proj_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]


    qkv_len =  [[96, 96], [192, 192], [192, 192, 192, 160, 192, 160, 384, 192, 224, 224, 256, 224, 192, 224, 256, 224, 192, 128], [32, 64]]
    fc1_len = [[102.4, 102.4], [204.8, 204.8], [224, 256, 256, 288, 224, 256, 256, 256, 409.6, 288, 320, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 288], [288, 544]]
    fc2_len = [[102.4, 102.4], [204.8, 204.8], [352, 256, 288, 256, 288, 256, 256, 288, 409.6, 288, 224, 224, 224, 256, 409.6, 409.6, 409.6, 409.6], [704, 672]]
    qkv_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    fc1_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, False, True, True, False, False, False, False, False, False, True], [True, True]]
    fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, False, False, False], [True, True]]

    # qkv_len =  [[64, 96], [160, 192], [192, 192, 192, 160, 256, 192, 256, 192, 224, 224, 224, 256, 256, 256, 256, 224, 224, 160], [192, 256]]
    # fc1_len = [[768, 768], [768, 768], [224, 224, 224, 256, 256, 256, 224, 224, 256, 224, 288, 288, 768, 768, 768, 768, 768, 320], [352, 544]]
    # fc2_len = [[768, 768], [768, 768], [256, 224, 224, 224, 224, 224, 288, 288, 288, 256, 288, 320, 320, 320, 768, 768, 768, 768], [544, 480]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, True], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False], [True, True]]

    proj_len = [[64,64],[128,128],[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],[512,512]]
    proj_use_uv = [[False, False], [False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False]]

    # qkv_len =  [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # fc1_len = [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # fc2_len = [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # proj_len = [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # qkv_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc2_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # proj_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]


    # qkv_len =  [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # fc1_len = [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # fc2_len = [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # proj_len = [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # qkv_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc2_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # proj_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]

    new_dict = {}
    l = list(model_dict.keys())
    for k in l:
        if "fc1.weight" in k or "fc2.weight" in k or "qkv.weight" in k or "attn.proj.weight" in k:
            k_split = k.split(".")
            k_split[-1] = "bias"
            bias_name = ".".join(k_split)

            fc_ind =  k_split[-2]
            layer_ind = int(k_split[1])
            block_ind = int(k_split[3])

            if fc_ind == "qkv":
                dim = qkv_len[layer_ind][block_ind]
                if not qkv_use_uv[layer_ind][block_ind]:
                    continue
            if fc_ind == "fc1":
                dim = fc1_len[layer_ind][block_ind]
                if not fc1_use_uv[layer_ind][block_ind]:
                    continue

            if fc_ind == "fc2":
                dim = fc2_len[layer_ind][block_ind]
                if not fc2_use_uv[layer_ind][block_ind]:
                    continue

            if fc_ind == "proj":
                dim = proj_len[layer_ind][block_ind]
                if not proj_use_uv[layer_ind][block_ind]:
                    continue

            print(k)

            if fc_ind == "fc1" or fc_ind == "fc2":
                k_split[-2] = "uv" + fc_ind[-1]
            elif fc_ind[:3] == "qkv":
                k_split[-2] = "uv1"
            elif fc_ind == "proj":
                k_split[-2] = "uv2"

            u,s,vh = torch.linalg.svd(model_dict[k], full_matrices=False)
            new_uv_weight = u[:,:dim] @ torch.diag(torch.sqrt(s[:dim]))
            new_weight = torch.diag(torch.sqrt(s[:dim])) @ vh[:dim,:]

            new_dict[".".join(k_split)] = model_dict[bias_name]

            new_dict[k] = new_weight

            k_split[-1] = "weight"
            new_dict[".".join(k_split)] = new_uv_weight

            model_dict.pop(bias_name)

    model_dict.update(new_dict)
    torch.save(model_dict, root_path +  "swin_b_ada_svd_0.33.pth")


# divide_svd()