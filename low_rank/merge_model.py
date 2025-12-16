import torch
import torch.nn as nn
import os
import time
import numpy as np
from functools import partial
import random
import cupy
import argparse
import math
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--block-i', type=int)
parser.add_argument('--idx-len', type=int, default=50)

args, unparsed = parser.parse_known_args()
block_i = args.block_i
idx_len = args.idx_len

def load_model_torch_resmlp():
    model_dict = torch.load("pretrainmodel/resmlpB_24_no_dist.pth")
    u_list = []
    v_list = []
    for i in range(24):
        v = torch.load("resmlp/ffn_output/all_fc_0.5/resmlpB_24/resmlpB_24_fewshot2_train_fc1_" + str(i) + "_v.pt")
        v_name = "blocks." + str(i) + ".mlp.uv1.v"
        model_dict[v_name] = v.transpose(0,1)

        u = torch.load("resmlp/ffn_output/all_fc_0.5/resmlpB_24/resmlpB_24_fewshot2_train_fc1_" + str(i) + "_u.pt")

        u_name = "blocks." + str(i) + ".mlp.uv1.u"
        model_dict[u_name] = u.transpose(0,1)

        b = torch.load("resmlp/ffn_output/all_fc_0.5/resmlpB_24/resmlpB_24_fewshot2_train_fc1_" + str(i) + "_b.pt")
        b_name = "blocks." + str(i) + ".mlp.uv1.b"
        # u_list.append(u)
        model_dict[b_name] = b.reshape(-1)

        v = torch.load("resmlp/ffn_output/all_fc_0.5/resmlpB_24/resmlpB_24_fewshot2_train_fc2_" + str(i) + "_v.pt")
        v_name = "blocks." + str(i) + ".mlp.uv2.v"
        model_dict[v_name] = v.transpose(0,1)

        u = torch.load("resmlp/ffn_output/all_fc_0.5/resmlpB_24/resmlpB_24_fewshot2_train_fc2_" + str(i) + "_u.pt")

        u_name = "blocks." + str(i) + ".mlp.uv2.u"
        model_dict[u_name] = u.transpose(0,1)

        b = torch.load("resmlp/ffn_output/all_fc_0.5/resmlpB_24/resmlpB_24_fewshot2_train_fc2_" + str(i) + "_b.pt")
        b_name = "blocks." + str(i) + ".mlp.uv2.b"
        # u_list.append(u)
        model_dict[b_name] = b.reshape(-1)

    torch.save(model_dict, "resmlp/ffn_output/all_fc_0.5/resmlpB_24/resmlpB_24_no_dist_pca_fewshot2_ffn_uvb.pt")
load_model_torch_resmlp()
def merge_model_torch_resmlp():
    model_dict = torch.load("resmlp/ffn_output/all_fc_0.5/resmlpB_24/resmlpB_24_no_dist_pca_fewshot2_ffn_uvb.pt", map_location='cpu')
    u_list = []
    v_list = []
    for i in range(24):
        w_name = "blocks."+ str(i) +".mlp.fc1.weight"
        b_name = "blocks."+ str(i) +".mlp.fc1.bias"
        v_name = "blocks." + str(i) + ".mlp.uv1.v"

        w = model_dict[w_name]
        b = model_dict[b_name]
        v = model_dict[v_name]
        new_w = v.transpose(0,1).float() @ w.float()
        new_b = b.float() @ v.float()
        print(new_w.shape)
        print(new_b.shape)
        model_dict[w_name] = new_w
        model_dict[b_name] = new_b
        model_dict.pop(v_name)

        w_name = "blocks."+ str(i) +".mlp.fc2.weight"
        b_name = "blocks."+ str(i) +".mlp.fc2.bias"
        v_name = "blocks." + str(i) + ".mlp.uv2.v"

        w = model_dict[w_name]
        b = model_dict[b_name]
        v = model_dict[v_name]
        new_w = v.transpose(0,1).float() @ w.float()
        new_b = b.float() @ v.float()
        print(new_w.shape)
        print(new_b.shape)
        model_dict[w_name] = new_w
        model_dict[b_name] = new_b
        model_dict.pop(v_name)
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

    torch.save(model_dict, "resmlp/ffn_output/all_fc_0.5/resmlpB_24/resmlpB_24_no_dist_pca_fewshot2_ffn_uvb_merge.pth")
merge_model_torch_resmlp()


def load_model_torch_swin():
    model_dict = torch.load("pretrainmodel/swin_base_patch4_window7_224.pth")
    # model_dict = torch.load("pretrainmodel/swin_small_patch4_window7_224.pth")
    # model_dict = torch.load("pretrainmodel/swin_tiny_patch4_window7_224.pth")

    model_dict = model_dict['model']
    u_list = []
    v_list = []

    count = 0
    # for i in range(24):
    new_dict = {}
    for k,v in model_dict.items():
        if "attn.qkv.weight" in k:
            print(k,count)
            k_split = k.split(".")
            k_split[-2] = "uv1"

            v = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_qkv_" + str(count) + "_v.pt")
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = v.transpose(0,1)

            u = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_qkv_" + str(count) + "_u.pt")
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = u.transpose(0,1)

            b = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_qkv_" + str(count) + "_b.pt")
            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name,u_name,b_name)

        elif "attn.proj.weight" in k:
            print(k,count)
            k_split = k.split(".")
            k_split[-2] = "uv2"

            v = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_proj_" + str(count) + "_v.pt")
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = v.transpose(0,1)

            u = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_proj_" + str(count) + "_u.pt")
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = u.transpose(0,1)

            b = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_proj_" + str(count) + "_b.pt")
            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name,u_name,b_name)

        elif "fc1.weight" in k:
            print(k,count)
            k_split = k.split(".")
            k_split[-2] = "uv1"

            v = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_fc1_" + str(count) + "_v.pt")
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = v.transpose(0,1)

            u = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_fc1_" + str(count) + "_u.pt")
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = u.transpose(0,1)

            b = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_fc1_" + str(count) + "_b.pt")
            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name,u_name,b_name)

        elif "fc2.weight" in k:
            print(k,count)
            k_split = k.split(".")
            k_split[-2] = "uv2"

            v = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_fc2_" + str(count) + "_v.pt")
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = v.transpose(0,1)

            u = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_fc2_" + str(count) + "_u.pt")
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = u.transpose(0,1)

            b = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_b_gen_fewshot2_train_fc2_" + str(count) + "_b.pt")
            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name,u_name,b_name)

            count += 1
    model_dict.update(new_dict)
    torch.save(model_dict, "swin/ffn_output/all_fc_0.5/swin_b/swin_gen_pca_all_uvb_fewshot2.pth")
# load_model_torch_swin()

def merge_model_torch_swin():
    # model_dict = torch.load("swin_pac_model/swin_pac_16_all_uvb.pth", map_location='cpu')
    model_dict = torch.load("swin/ffn_output/all_fc_0.5/swin_b/swin_gen_pca_all_uvb_fewshot2.pth", map_location='cpu')
    count = 0
    l = list(model_dict.keys())
    for k in l:
        if ("fc1.weight" in k or "attn.qkv.weight" in k): 
        # and "layers.3.blocks.1" not in k:
            print(k)
            w = model_dict[k]
            b = model_dict[k[:-6] + "bias"]

            k_split = k.split(".")
            k_split[-2] = "uv1"
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            v = model_dict[v_name]

            new_w = v.transpose(0,1) @ w
            new_b = b @ v
            # print(new_w.shape)
            # print(new_b.shape)
            model_dict[k] = new_w
            model_dict[k[:-6] + "bias"] = new_b

        if "fc2.weight" in k or "attn.proj.weight" in k: 
        # if "fc2.weight" in k: 
        # and "layers.3.blocks.1" not in k:
            print(k)
            w = model_dict[k]
            b = model_dict[k[:-6] + "bias"]

            k_split = k.split(".")
            k_split[-2] = "uv2"
            k_split[-1] = "v"
            v_name = ".".join(k_split)
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

    torch.save(model_dict, "swin/ffn_output/all_fc_0.5/swin_b/swin_gen_pca_fewshot2_merge_attn_ffn.pth")
# merge_model_torch_swin()

def load_model_torch_levit():
    model_dict = torch.load("pretrainmodel/LeViT-256-nobn.pth")

    u_list = []
    v_list = []

    attn_count = 0
    fc_count = 0
    # for i in range(24):
    new_dict = {}
    for k,v in model_dict.items():
        if "m.qkv.weight" in k:
            print(k,attn_count)
            k_split = k.split(".")
            k_split[-2] = "uv1"

            v = torch.load("levit/ffn_output/all_fc_0.5/levit_256/levit_256_fewshot2_train_qkv_" + str(attn_count) + "_v.pt")
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = v.transpose(0,1)

            u = torch.load("levit/ffn_output/all_fc_0.5/levit_256/levit_256_fewshot2_train_qkv_" + str(attn_count) + "_u.pt")
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = u.transpose(0,1)

            b = torch.load("levit/ffn_output/all_fc_0.5/levit_256/levit_256_fewshot2_train_qkv_" + str(attn_count) + "_b.pt")
            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name,u_name,b_name)
            attn_count += 1

        elif "m.0.weight" in k:
            print(k,fc_count)
            k_split = k.split(".")
            k_split[-2] = "uv1"

            v = torch.load("levit/ffn_output/all_fc_0.5/levit_256/levit_256_fewshot2_train_fc1_" + str(fc_count) + "_v.pt")
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = v.transpose(0,1)

            u = torch.load("levit/ffn_output/all_fc_0.5/levit_256/levit_256_fewshot2_train_fc1_" + str(fc_count) + "_u.pt")
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = u.transpose(0,1)

            b = torch.load("levit/ffn_output/all_fc_0.5/levit_256/levit_256_fewshot2_train_fc1_" + str(fc_count) + "_b.pt")
            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name,u_name,b_name)

        elif "m.2.weight" in k:
            print(k,fc_count)
            k_split = k.split(".")
            k_split[-2] = "uv2"

            v = torch.load("levit/ffn_output/all_fc_0.5/levit_256/levit_256_fewshot2_train_fc2_" + str(fc_count) + "_v.pt")
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = v.transpose(0,1)

            u = torch.load("levit/ffn_output/all_fc_0.5/levit_256/levit_256_fewshot2_train_fc2_" + str(fc_count) + "_u.pt")
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = u.transpose(0,1)

            b = torch.load("levit/ffn_output/all_fc_0.5/levit_256/levit_256_fewshot2_train_fc2_" + str(fc_count) + "_b.pt")
            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name,u_name,b_name)

            fc_count += 1
    model_dict.update(new_dict)
    torch.save(model_dict, "levit/ffn_output/all_fc_0.5/levit_256/levit_256_pca_all_uvb_fewshot2.pth")
# load_model_torch_levit()

def merge_model_torch_levit():
    model_dict = torch.load("levit/ffn_output/all_fc_0.5/levit_256/levit_256_pca_all_uvb_fewshot2.pth", map_location='cpu')
    count = 0
    l = list(model_dict.keys())
    for k in l:
        if ("m.0.weight" in k or "m.qkv.weight" in k): 
        # and "layers.3.blocks.1" not in k:
            print(k)
            w = model_dict[k]
            b = model_dict[k[:-6] + "bias"]

            k_split = k.split(".")
            k_split[-2] = "uv1"
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            v = model_dict[v_name]

            new_w = v.transpose(0,1) @ w
            new_b = b @ v
            # print(new_w.shape)
            # print(new_b.shape)
            model_dict[k] = new_w
            model_dict[k[:-6] + "bias"] = new_b

        # if "fc2.weight" in k or "attn.proj.weight" in k: 
        if "m.2.weight" in k: 
        # and "layers.3.blocks.1" not in k:
            print(k)
            w = model_dict[k]
            b = model_dict[k[:-6] + "bias"]

            k_split = k.split(".")
            k_split[-2] = "uv2"
            k_split[-1] = "v"
            v_name = ".".join(k_split)
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
    
    l = list(model_dict.keys())
    for k in l:
        if (".m.0.weight" in k): 
            print(k)
            k_split = k.split(".")

            k_split[-2] = "uv1"
            w1 = model_dict[".".join(k_split)]
            k_split[-1] = "bias"
            b1 = model_dict[".".join(k_split)]

            k_split[-2] = "2"
            b3 = model_dict[".".join(k_split)]
            k_split[-1] = "weight"
            w3 = model_dict[".".join(k_split)]

            k_split[-2] = "uv2"
            w4 = model_dict[".".join(k_split)]
            k_split[-1] = "bias"
            b4 = model_dict[".".join(k_split)]

            k_split[-2] = "1"
            model_dict[".".join(k_split)] = b1
            k_split[-1] = "weight"
            model_dict[".".join(k_split)] = w1

            k_split[-2] = "3"
            model_dict[".".join(k_split)] = w3
            k_split[-1] = "bias"
            model_dict[".".join(k_split)] = b3

            k_split[-2] = "4"
            model_dict[".".join(k_split)] = b4
            k_split[-1] = "weight"
            model_dict[".".join(k_split)] = w4

    # l = list(model_dict.keys())
    # for k in l:
    #     if k[-5:-2] == "uv1"  or k[-5:-2] == "uv2":
    #         model_dict.pop(k)

    torch.save(model_dict, "levit/ffn_output/all_fc_0.5/levit_256/levit_256_pca_all_uvb_fewshot2_merge.pth")
# merge_model_torch_levit()

def load_model_torch_deit():
    # model_dict = torch.load("pretrainmodel/deit_tiny_patch16_224-a1311bcf.pth")
    model_dict = torch.load("pretrainmodel/deit_base_patch16_224-b5f2ef4d.pth")
    model_dict = model_dict['model']
    u_list = []
    v_list = []

    attn_count = 0
    fc_count = 0

    # root_path = "deit/ffn_output/all_fc_ada/deit_t/"
    root_path = "deit/ffn_output/all_fc_0.5/deit_b/"
    new_dict = {}
    for k,v in model_dict.items():
        if "qkv.weight" in k:
            print(k,attn_count)
            k_split = k.split(".")
            k_split[-2] = "uv1"

            v = torch.load(root_path +  "deit_b_fewshot2_qkv_" + str(attn_count) + "_v.pt")
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = v.transpose(0,1)

            u = torch.load(root_path +  "deit_b_fewshot2_qkv_" + str(attn_count) + "_u.pt")
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = u.transpose(0,1)

            b = torch.load(root_path +  "deit_b_fewshot2_qkv_" + str(attn_count) + "_b.pt")
            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name,u_name,b_name)
            attn_count += 1

        elif "fc1.weight" in k:
            print(k,fc_count)
            k_split = k.split(".")
            k_split[-2] = "uv1"

            v = torch.load(root_path +  "deit_b_fewshot2_fc1_" + str(fc_count) + "_v.pt")
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = v.transpose(0,1)

            u = torch.load(root_path +  "deit_b_fewshot2_fc1_" + str(fc_count) + "_u.pt")
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = u.transpose(0,1)

            b = torch.load(root_path +  "deit_b_fewshot2_fc1_" + str(fc_count) + "_b.pt")
            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name,u_name,b_name)

        elif "fc2.weight" in k:
            print(k,fc_count)
            k_split = k.split(".")
            k_split[-2] = "uv2"

            v = torch.load(root_path +  "deit_b_fewshot2_fc2_" + str(fc_count) + "_v.pt")
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            new_dict[v_name] = v.transpose(0,1)

            u = torch.load(root_path +  "deit_b_fewshot2_fc2_" + str(fc_count) + "_u.pt")
            k_split[-1] = "u"
            u_name = ".".join(k_split)
            new_dict[u_name] = u.transpose(0,1)

            b = torch.load(root_path +  "deit_b_fewshot2_fc2_" + str(fc_count) + "_b.pt")
            k_split[-1] = "b"
            b_name = ".".join(k_split)
            new_dict[b_name] = b.reshape(-1)
            print(v_name,u_name,b_name)

            fc_count += 1
    model_dict.update(new_dict)
    torch.save(model_dict, root_path +  "deit_b_pca_all_uvb_fewshot2.pth")
# load_model_torch_deit()

def merge_model_torch_deit():
    root_path = "deit/ffn_output/all_fc_0.5/deit_b/"
    model_dict = torch.load(root_path +  "deit_b_pca_all_uvb_fewshot2.pth", map_location='cpu')
    count = 0
    l = list(model_dict.keys())
    for k in l:
        # if ("fc1.weight" in k or "attn.qkv.weight" in k) and "blocks." + str(args.block_i) + "." not in k and False: 
        # and "layers.3.blocks.1" not in k:
        if "fc1.weight" in k: 
            print(k)
            w = model_dict[k]
            b = model_dict[k[:-6] + "bias"]

            k_split = k.split(".")
            k_split[-2] = "uv1"
            k_split[-1] = "v"
            v_name = ".".join(k_split)
            v = model_dict[v_name]

            new_w = v.transpose(0,1) @ w
            new_b = b @ v
            # print(new_w.shape)
            # print(new_b.shape)
            model_dict[k] = new_w
            model_dict[k[:-6] + "bias"] = new_b

        # if "fc2.weight" in k or : 
        if "fc2.weight" in k and "blocks." + str(args.block_i) + "." not in k: 
        # and "layers.3.blocks.1" not in k:
            print(k)
            w = model_dict[k]
            b = model_dict[k[:-6] + "bias"]

            k_split = k.split(".")
            k_split[-2] = "uv2"
            k_split[-1] = "v"
            v_name = ".".join(k_split)
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
    

    torch.save(model_dict, root_path +  "deit_b_pca_all_uvb_fewshot2_merge_wob" + str(args.block_i) + ".pth")
# merge_model_torch_deit()