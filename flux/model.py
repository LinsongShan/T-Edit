from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flux.modules.layers import (
    DoubleStreamBlock, 
    EmbedND, 
    LastLayer,
    MLPEmbedder, 
    SingleStreamBlock,
    timestep_embedding,
)

import torch.nn.functional as F
from tensor import layer_importance 

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool



class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        info = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        
        K = []
        V = []

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        #####################################################################             
        cnt_mm = 0
        info['type'] = 'mm'
        for block in self.double_blocks:
            info['id'] = cnt_mm
            img, txt, k, v = block(img=img, txt=txt, vec=vec, pe=pe, info=info)
            if info['inverse']:
                K.append(k)
                V.append(v)
            cnt_mm += 1

        cnt = 19
        img = torch.cat((txt, img), 1) 
        info['type'] = 'single'
        for block in self.single_blocks:
            info['id'] = cnt
            # print(info)
            img, info, k, v = block(img, vec=vec, pe=pe, info=info)
            if info['inverse']:
                K.append(k)
                V.append(v)
            cnt += 1
        #####################################################################

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  
        
        #####################################################################
        importance_scores = []
        top_idx = []
        if info['inverse']:
            if info['t_step'] < info['inject_step']:
                K = torch.stack([k.detach() for k in K], dim=0)
                V = torch.stack([v.detach() for v in V], dim=0)

                importance_scores = layer_importance(V)
                importance_scores = torch.as_tensor(importance_scores, device=V.device)

                sorted_scores, sorted_idx = torch.sort(importance_scores, descending=True)
                cum_scores = torch.cumsum(sorted_scores, dim=0)
                total_score = cum_scores[-1]                
                threshold = 0.90 * total_score
                k = torch.searchsorted(cum_scores, threshold).item() + 1
                
                # top_idx = sorted_idx[:k].cpu().tolist()
                # top_idx = (top_idx + [0] * 10)[:10]
                top_idx = F.pad(sorted_idx[:k], (0, max(0, 10 - k)))[:10]

                features = {
                    f"{info['t']}_{n}K": K[n].cpu() for n in range(K.shape[0])
                }
                features.update({
                    f"{info['t']}_{n}V": V[n].cpu() for n in range(V.shape[0])
                })
                info['feature'].update(features)
    
            else:
                # top_idx = [-1] * 10
                top_idx = torch.full((10,), -1, dtype=torch.long, device=img.device)
        #####################################################################
        
        return img, info, top_idx
            
