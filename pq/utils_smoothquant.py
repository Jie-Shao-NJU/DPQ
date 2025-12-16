import torch
import torch.nn as nn
from pq.low_rank_models import DiTBlock_uv

@torch.no_grad()
def smooth_fcs(fcs_u, fcs_v, alpha=0.5):
    if not isinstance(fcs_u, list):
        fcs_u = [fcs_u]
    if not isinstance(fcs_v, list):
        fcs_v = [fcs_v]

    device, dtype = fcs_u[0].weight.device, fcs_u[0].weight.dtype
    weight_scales_u = torch.cat(
        [fc_u.weight.T.abs().max(dim=0, keepdim=True)[0] for fc_u in fcs_u], dim=0
    )
    weight_scales_v = torch.cat(
        [fc_v.weight.abs().max(dim=0, keepdim=True)[0] for fc_v in fcs_v], dim=0
    )
    weight_scales_u = weight_scales_u.max(dim=0)[0].clamp(min=1e-5)
    weight_scales_v = weight_scales_v.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (weight_scales_u.pow(alpha) / weight_scales_v.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    for fc_u in fcs_u:
        fc_u.weight.div_(scales.view(-1, 1))
        fc_u.bias.div_(scales.view(-1))
    for fc_v in fcs_v:
        fc_v.weight.mul_(scales.view(1, -1))
    
    print('done')


@torch.no_grad()
def smooth_dit(model, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, DiTBlock_uv):
            qkv_u = module.attn.qkv_u
            qkv_v = module.attn.qkv_v
            smooth_fcs(qkv_u, qkv_v, alpha)

            proj_u = module.attn.proj_u
            proj_v = module.attn.proj_v
            smooth_fcs(proj_u, proj_v, alpha)

            fc1_u = module.mlp.fc1_u
            fc1_v = module.mlp.fc1_v
            smooth_fcs(fc1_u, fc1_v, alpha)

            fc2_u = module.mlp.fc2_u
            fc2_v = module.mlp.fc2_v
            smooth_fcs(fc2_u, fc2_v, alpha)

            adaln_u = module.adaLN_modulation[1]
            adaln_v = module.adaLN_modulation[2]
            smooth_fcs(adaln_u, adaln_v, alpha)