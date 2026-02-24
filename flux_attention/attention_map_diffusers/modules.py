import math
import inspect
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union, List, Callable
import time 
import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.sattention import _chunked_feed_forward
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
# from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput
# from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
# from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
# from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
#     ASPECT_RATIO_512_BIN,
#     ASPECT_RATIO_1024_BIN,
# )
from diffusers.pipelines.flux.pipeline_flux import (
    retrieve_timesteps,
    replace_example_docstring,
    EXAMPLE_DOC_STRING,
    calculate_shift,
    XLA_AVAILABLE,
    FluxPipelineOutput
)
# from diffusers.models.transformers import FLUXTransformer2DModel
from diffusers.utils import (
    deprecate,
    BaseOutput,
    is_torch_version,
    logging,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
)

logger = logging.get_logger(__name__)


attn_maps = {}



# ========== 全局变量存储 ==========
attention_collector = {
    'is_collecting': False,
    'collected_attentions': {},  # timestep -> List[attention]
    'current_timestep': None,
}

concept_mask_dict = {
    'masks': {},  # scene_idx -> mask (1, 1, H, W)
    'apply': False,
    'alpha': 1.5,
    'num_scenes': 4,
}


# ========== Attention收集工具 ==========
def start_attention_collection():
    """开始收集attention"""
    global attention_collector
    attention_collector['is_collecting'] = True
    attention_collector['collected_attentions'] = {}
    print("✓ Started attention collection")


def stop_attention_collection():
    """停止收集attention"""
    global attention_collector
    attention_collector['is_collecting'] = False
    print(f"✓ Stopped attention collection (collected {len(attention_collector['collected_attentions'])} timesteps)")


def add_collected_attention(timestep: int, img_txt_attn: torch.Tensor):
    """添加收集的attention"""
    global attention_collector
    if not attention_collector['is_collecting']:
        return
    
    if timestep not in attention_collector['collected_attentions']:
        attention_collector['collected_attentions'][timestep] = []
    
    attention_collector['collected_attentions'][timestep].append(img_txt_attn.detach().cpu())


def get_average_attention():
    """获取平均attention"""
    global attention_collector
    
    if not attention_collector['collected_attentions']:
        return None
    
    all_attns = []
    #(timestep, layer, b, h , img_len, text_len)
    # for timestep in sorted(attention_collector['collected_attentions'].keys()):
    #     attn_list = attention_collector['collected_attentions'][timestep]

    #     print(f'timestep {timestep} attn_list len {len(attn_list)} attn_list[0] shape {attn_list[0].shape}')
    #     attn_stack = torch.stack(attn_list, dim=0)  # (num_layers, B, heads, img_len, text_len)
    #     all_attns.append(attn_stack)
    all_keys = sorted(attention_collector['collected_attentions'].keys())
    attn_list = attention_collector['collected_attentions'][all_keys[0]]

    print(f'timestep {all_keys[0]} attn_list len {len(attn_list)} attn_list[0] shape {attn_list[0].shape}')
    attn_stack = torch.stack(attn_list, dim=0)  # (num_layers, B, heads, img_len, text_len)
    all_attns.append(attn_stack)


    all_attns = torch.cat(all_attns, dim=0)  # (all_layer, B, heads, img_len, text_len)
    print(f'all_attns shape {all_attns.shape}')
    avg_attn = all_attns.mean(dim=(0, 1))  # (heads, img_len, text_len)
    print(f'avg attn shape: {avg_attn.shape}')
    # 这里可以做 timestep 与 layer平滑
    return avg_attn


import os
import matplotlib.pyplot as plt
import torch


def save_mask_image(
    mask_1d: torch.Tensor,
    save_path: str,
    title: str,
    cmap: str = "gray",
    H: int = 64,
    W: int = 64,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mask_2d = (
        mask_1d
        .view(H, W)
        .detach()
        .float()          # 🔑 关键修复：bfloat16 → float32
        .cpu()
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(mask_2d, cmap=cmap)
    # plt.title(title)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def compute_scene_token_masks(
    avg_attention: torch.Tensor,
    scene_concept_indices,
    scene_text_ranges,
    num_scenes: int = 4,
    lambda_std: float = 1.3,
    gaussian_sigma: float = 1.5,
    gaussian_kernel: int = 11,
    expand_std_lambda: float = 0.5,  # 新增参数
    mask_save_dir: str = "./mask_save",
    concept_order: int = 0,
    enable_dynamic_token_mask: bool = False
):
    token_masks = {}
    H = W = 64

    os.makedirs(mask_save_dir, exist_ok=True)

    scene_img_token_indices = {}

    for scene_idx in range(num_scenes):
        if scene_idx >= len(scene_concept_indices):
            continue

        # 现在每个 scene 有两个 concept
        # (concept_indices, partner_indices)
        concept_pair = scene_concept_indices[scene_idx]
        if not concept_pair:
            continue

        scene_dir = os.path.join(
            mask_save_dir,
            f"concept_{concept_order:04d}",
            f"scene_{scene_idx:02d}",
        )

        # ===============================
        # scene → image token indices（2×2 grid）
        # ===============================
        h_mid = H // 2
        w_mid = W // 2

        if scene_idx == 0:  # 左上
            h_range = range(0, h_mid)
            w_range = range(0, w_mid)
        elif scene_idx == 1:  # 右上
            h_range = range(0, h_mid)
            w_range = range(w_mid, W)
        elif scene_idx == 2:  # 左下
            h_range = range(h_mid, H)
            w_range = range(0, w_mid)
        elif scene_idx == 3:  # 右下
            h_range = range(h_mid, H)
            w_range = range(w_mid, W)
        else:
            continue

        scene_img_idx = [
            h * W + w
            for h in h_range
            for w in w_range
        ]
        scene_img_token_indices[scene_idx] = scene_img_idx

        # ==========================================================
        # 对两个 concept 分别计算 token mask
        # ==========================================================
        for concept_id, token_indices in enumerate(concept_pair):
            if(len(token_indices)) == 0:
            # if not token_indices:
                print(" (empty token indices, skipped)")
                continue

            concept_dir = os.path.join(
                scene_dir,
                f"concept_{concept_id}"
            )
            os.makedirs(concept_dir, exist_ok=True)

            # 1️⃣ entity attention
            # avg_attention: (heads, img_len, text_len)
            # token_indices: list[int]
            entity_attn = avg_attention[:, :, token_indices]

            # 2️⃣ average attention -> (img_len,)
            attn_map = entity_attn.mean(dim=0).mean(dim=-1)

            save_mask_image(
                attn_map,
                os.path.join(concept_dir, "attn_raw.png"),
                title="Raw Attention",
                cmap="viridis",
            )

            # 3️⃣ μ / σ
            mu = attn_map.mean()
            sigma = attn_map.std(unbiased=False)
            threshold = mu + lambda_std * sigma

            save_mask_image(
                attn_map,
                os.path.join(concept_dir, "attn_before_std.png"),
                title=f"Before STD (μ={mu:.3f}, σ={sigma:.3f})",
                cmap="viridis",
            )

            # 4️⃣ binary before gaussian
            binary_pre = (attn_map > threshold).float()

            save_mask_image(
                binary_pre,
                os.path.join(concept_dir, "binary_before_gaussian.png"),
                title="Binary Before Gaussian",
            )

            # 5️⃣ gaussian expansion（当前注释保留）
            # final_mask = expand_binary_mask_with_gaussian(
            #     binary_pre,
            #     gaussian_sigma=gaussian_sigma,
            #     gaussian_kernel=gaussian_kernel,
            #     std_lambda=expand_std_lambda,
            # )
            # save_mask_image(
            #     final_mask,
            #     os.path.join(concept_dir, "final_mask.png"),
            #     title="Final Mask",
            # )

            # 保存结果
            token_masks[(scene_idx, concept_id)] = binary_pre

            # print(
            #     f" μ={mu:.4f}, σ={sigma:.4f}, "
            #     f"ratio={final_mask.mean():.3f}"
            # )

    return token_masks, scene_img_token_indices



import torch
import torch.nn.functional as F
from typing import Dict, List


def build_gaussian_kernel_1d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
):
    """
    Build normalized 1D Gaussian kernel.
    """
    assert kernel_size % 2 == 1
    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)


def expand_binary_mask_with_gaussian(
    binary_mask: torch.Tensor,      # (img_len,) values ∈ {0,1}
    gaussian_sigma: float = 1.5,
    gaussian_kernel: int = 11,
    std_lambda: float = 0.5,
):
    """
    Make a binary mask spatially coherent using:
    Gaussian smoothing + mean/std adaptive re-thresholding.

    Returns:
        expanded binary mask (0/1), shape unchanged
    """
    device = binary_mask.device

    # ---------- 1️⃣ Gaussian smoothing (neighborhood voting) ----------
    kernel = build_gaussian_kernel_1d(
        gaussian_kernel,
        gaussian_sigma,
        device,
    )

    smooth = F.conv1d(
        binary_mask[None, None, :],
        kernel,
        padding=gaussian_kernel // 2,
    ).squeeze()

    # ---------- 2️⃣ Statistics on smoothed mask ----------
    mu = smooth.mean()
    std = smooth.std(unbiased=False)

    threshold = mu + std_lambda * std

    # ---------- 3️⃣ Re-binarize (region expansion) ----------
    expanded_mask = (smooth > threshold).float()

    return expanded_mask


def set_scene_token_masks(token_masks_dict: Dict[int, torch.Tensor], alpha: float = 1.5):
    """
    设置scene token masks用于应用阶段
    
    Args:
        token_masks_dict: {scene_idx: token_mask (img_len,), ...}
        alpha: 增强系数
    """
    global concept_mask_dict

    concept_mask_dict['token_masks'] = token_masks_dict
    concept_mask_dict['apply'] = True
    concept_mask_dict['alpha'] = alpha
    concept_mask_dict['num_scenes'] = len(token_masks_dict)
    print(f"✓ Set token masks for {len(token_masks_dict)} scenes (alpha={alpha})")
    print('?' * 60)

def clear_scene_token_masks():
    """清除scene token masks"""
    global concept_mask_dict
    concept_mask_dict['token_masks'] = {}
    concept_mask_dict['apply'] = False

import time

def FluxAttnProcessor2_0_call(
    attn,
    hidden_states,
    encoder_hidden_states = None,
    attention_mask = None,
    image_rotary_emb = None,
    ) -> torch.FloatTensor:
        
        batch_size, _, _ = hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            
            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
            
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # apply mask on attention
        hidden_states = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
            
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

# ========== 修改的Flux Attention函数 ==========
def flux_attn_call2_0(
    self,
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    height: int = None,
    timestep: Optional[torch.Tensor] = None,
    scene_text_ranges: Optional[list] = None,  # 新增参数
    img_masks = None,
    prompt_mask_list = None,
    mask_all = None,
    beta = 1.3,
    full_denoise_steps = None,
    base_radio = None,
    MULTIMODAL_VITAL_LAYERS = None,
    SINGLE_MODAL_VITAL_LAYERS = None,
    index_block = None,
    encoder_hidden_states_base = None,
    image_rotary_emb_base = None,
    global_prompt = None, 
    global_prompt_embeds = None,
    hidden_states_base = None,
    fuse_steps_start = None,
    fuse_steps_end = None,
    get_token_mask = None,
    scene_concept_indices = None,
    scene_img_token_indices = None,
    is_last_collection = None,
) -> torch.FloatTensor:
    
    #print(f'current timestep: {timestep}')
    #print(f'current encoder_hidden_states: {encoder_hidden_states.shape}')
    """
    改进的Flux Attention处理 - 支持Token级别的Mask
    支持: 1) 收集attention maps
         2) 应用per-scene token masks增强
    """
    origin_timestep = timestep
    if base_radio is not None and origin_timestep[0] <= fuse_steps_start and origin_timestep[0] >= fuse_steps_end - 1 and get_token_mask is None:
        attn_output_base = FluxAttnProcessor2_0_call(
            attn=attn,
            hidden_states=hidden_states_base if hidden_states_base is not None else hidden_states,
            encoder_hidden_states=encoder_hidden_states_base,
            attention_mask=None,
            image_rotary_emb=image_rotary_emb_base,
        )
        # single block
        if encoder_hidden_states_base is not None:
            hidden_states_base, encoder_hidden_states_base = attn_output_base
        else:
            hidden_states_base = attn_output_base
            
    global attention_collector, concept_mask_dict
    
    batch_size = hidden_states.shape[0]
    
    # ========== 标准Attention计算 ==========
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    
    # ========== Cross-Attention投影 ==========
    if encoder_hidden_states is not None:
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        
        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        
        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
        
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
    
    # ========== 旋转位置编码 ==========
    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)
    
    # ========== Attention分数计算 ==========
    attn_scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / math.sqrt(head_dim)
    if mask_all is not None:
        attn_scores = attn_scores + mask_all # mask

    attention_probs = torch.softmax(attn_scores, dim=-1)

    if hasattr(self, "store_attn_map") and encoder_hidden_states is not None:
        attention_probs_img_text = attention_probs[:, :, 512:, :512].cpu()
        self.attn_map = rearrange(attention_probs_img_text, 'b h (h1 w1) t -> b h h1 w1 t', h1=height)
        self.timestep = timestep[0].cpu().item() if timestep is not None else None


    # ========== 【阶段1】收集img-txt attention ==========
    # before softmax
    # if attention_collector['is_collecting'] and encoder_hidden_states is not None:
    if attention_collector['is_collecting'] and get_token_mask is not None and is_last_collection:
        text_len = 512
        
        # 提取img-txt attention
        # before soft max
        img_txt_attn = attn_scores[:, :, text_len:, :text_len]  # (B, heads, img_len, text_len)

        # 获取当前timestep
        current_timestep = int(timestep[0].item()) if timestep is not None else 0
        
        # 保存
        add_collected_attention(current_timestep, img_txt_attn)

    
    if concept_mask_dict['apply'] and get_token_mask is None:

        text_len = 512
        img_len = hidden_states.shape[1]
        
        # 复制attention以便修改
        attention_probs = attn_scores.clone()

        # 对每个scene应用对应的token mask
        # img → txt attention
        img_txt_attn = attention_probs[:, :, text_len:, :text_len]

        for (scene_idx, concept_id), _ in concept_mask_dict['token_masks'].items():
            token_mask = concept_mask_dict['token_masks'][(0, concept_id)] # 都使用第一个场景的mask
            token_mask = token_mask.to(attention_probs.device)

            scene_img_idx = scene_img_token_indices[scene_idx]
            if len(scene_img_idx) == 0:
                continue
            
            scene_text_start, _ = scene_text_ranges[scene_idx]
            local_concept_idx = scene_concept_indices[scene_idx][concept_id]
            if len(local_concept_idx) == 0:
                continue

            # concept_txt_idx = [
            #     scene_text_start + i
            #     for i in local_concept_idx
            #     if 0 <= scene_text_start + i < text_len
            # ]
            concept_txt_idx = [
                scene_text_start + i
                for i in local_concept_idx
            ]
            if len(concept_txt_idx) == 0:
                continue

            #print(concept_txt_idx)

            k_start = int(concept_txt_idx[0])
            k_end   = int(concept_txt_idx[-1]) + 1  

            local_attn = img_txt_attn[
                :, :,
                scene_img_idx,
                k_start:k_end
            ]

            local_mask = token_mask[scene_img_idx]          # (|Q|,), ∈ {0,1}
            # print(token_mask.shape)
            # print(local_mask.shape)
            # print(local_attn.shape)
            local_mask = local_mask.view(1, 1, -1, 1)   # (1, 1, 1024, len)

            # ===============================
            # 5️⃣ α · Attn(q,k)  (hard gating)
            # ===============================\
            alpha = concept_mask_dict['alpha'] - 1
            local_attn = local_attn * (alpha * local_mask)
            local_attn = local_attn.to(img_txt_attn.dtype)

            # ===============================
            # 6️⃣ 写回
            # ===============================
            img_txt_attn[
                :, :,
                scene_img_idx,
                k_start:k_end
            ] = local_attn

        attention_probs[:, :, text_len:, :text_len] = img_txt_attn
        attention_probs = torch.softmax(attn_scores, dim=-1)

        # attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)

    
    # ========== 计算输出 ==========
    hidden_states = torch.einsum('bhqk,bhkd->bhqd', attention_probs, value)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

        
    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        if base_radio is not None and origin_timestep[0] <= fuse_steps_start and origin_timestep[0] >= fuse_steps_end - 1 and get_token_mask is None:
            # merge hidden_states and hidden_states_base
            hidden_states = hidden_states*(1-base_radio) + hidden_states_base*base_radio
            return hidden_states, encoder_hidden_states, encoder_hidden_states_base
        else: # both regional and base input are base prompts, skip the merge
            return hidden_states, encoder_hidden_states, encoder_hidden_states
        
    else:
        if base_radio is not None and origin_timestep[0] <= fuse_steps_start and origin_timestep[0] >= fuse_steps_end - 1 and get_token_mask is None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : 512],
                hidden_states[:, 512 :],
            )
               
            encoder_hidden_states_base, hidden_states_base = (
                hidden_states_base[:, : 512],
                hidden_states_base[:, 512:],
            )

            # merge hidden_states and hidden_states_base
            hidden_states = hidden_states*(1-base_radio) + hidden_states_base * base_radio

            # concat back            
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states_base = torch.cat([encoder_hidden_states_base, hidden_states_base], dim=1)

            return hidden_states, hidden_states_base
        
        
        return hidden_states, hidden_states


@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def FluxPipeline_call(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    collection_steps: int = 28,
    start_steps: int = 0,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    true_cfg_scale: float = 1.0,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    is_use_global_prompt: bool = False
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            will be used instead
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        guidance_scale (`float`, *optional*, defaults to 7.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

    Examples:

    Returns:
        [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
        is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
        images.
    """

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    
    # add base prompt
    if joint_attention_kwargs.get('global_prompt') is not None:
        global_prompt = joint_attention_kwargs['global_prompt'] 
        (
            global_prompt_embeds,
            global_pooled_prompt_embeds,
            global_text_ids,
        ) = self.encode_prompt(
            prompt=global_prompt,
            prompt_2=global_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        encoder_hidden_states_base = global_prompt_embeds

    if do_true_cfg:
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_text_ids,
        ) = self.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )


    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    fuse_steps_start = joint_attention_kwargs['fuse_steps_start']
    fuse_steps_end = joint_attention_kwargs['fuse_steps_end']
    fuse_start = timesteps[fuse_steps_start]
    fuse_end = timesteps[fuse_steps_end - 1]
    joint_attention_kwargs['fuse_steps_start'] = fuse_start
    joint_attention_kwargs['fuse_steps_end'] = fuse_end
    print(f'timesteps {timesteps}')
    print(f'fuse_start {fuse_start}')
    print(f'fuse_end {fuse_end}')

    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None
    
    if prompt_2 is None:
        prompt_2 = prompt
    num_grid= 4
    #print(prompt_2)
    
    scene_texts, scene_text_ranges, input_ids_all = split_prompt_into_scenes_and_ranges(
        prompt_2, self.tokenizer_2, num_scenes=num_grid, is_use_global_prompt=is_use_global_prompt
    )
    pannel_ranges = []
    text_ranges = []
    for i, scene in enumerate(scene_text_ranges):
        text_ranges.append(scene)
    if joint_attention_kwargs is not None and joint_attention_kwargs.get("img_masks") is not None:
        img_masks = joint_attention_kwargs['img_masks']
        B = 1
        C = 24
        attn_scores = torch.randn(B, C, 512 + 4096, 512 + 4096)
        mask_all = torch.ones_like(attn_scores) * -1e9
        prompt_mask_list = []
        prompt_length = scene_text_ranges[num_grid-1][1]
        print(text_ranges)
        #exit(0)
        if is_use_global_prompt:
            global_start = 0
            global_end = text_ranges[0][0]

        #mask_all[:,:,global_start:global_end, ]
        if scene_text_ranges is not None:
            for i, (txt_start, txt_end) in enumerate(text_ranges):
                mask = torch.ones_like(attn_scores[:, :, : , : ]) * -1e9 # grid implementation
                mask[:, :, txt_start:txt_end, txt_start:txt_end] = 0  # text-text
                if is_use_global_prompt:
                    mask[:, :, global_start:global_end, global_start:global_end] = 0 # global

                mask[:, :, prompt_length: 512, prompt_length: 512] = 0
                # img-text
                mask[:,:, 512 : , txt_start:txt_end] = img_masks[i][:, :, :, :1].repeat(1, 1, 1, txt_end - txt_start)
                # text-img
                mask[:,:, txt_start:txt_end, 512: ] = img_masks[i][:, :, :, :1].repeat(1, 1, 1, txt_end - txt_start).transpose(-1, -2)
                
                mask[:,:, 512 : , prompt_length:512] = img_masks[i][:, :, :, :1].repeat(1, 1, 1, 512 - prompt_length)
                # text-img
                mask[:,:, prompt_length:512, 512: ] = img_masks[i][:, :, :, :1].repeat(1, 1, 1, 512 - prompt_length).transpose(-1, -2)
                if is_use_global_prompt:
                    mask[:,:, 512 : , global_start:global_end] = img_masks[i][:, :, :, :1].repeat(1, 1, 1, global_end - global_start)
                    # text-img
                    mask[:,:, global_start:global_end, 512: ] = img_masks[i][:, :, :, :1].repeat(1, 1, 1, global_end - global_start).transpose(-1, -2)
                    
                    
                img_size_masks = img_masks[i][:, :, :, :1].repeat(1, 1, 1, 4096)
                img_size_masks_transpose = img_size_masks.transpose(-1, -2)

                combine_img_size = torch.where(
                    (img_size_masks == 0) & (img_size_masks_transpose == 0),
                    torch.tensor(0.0, device=img_size_masks.device),
                    torch.tensor(-1e9, device=img_size_masks.device)
                )

                mask[:,:, 512:, 512:] = combine_img_size
                #save_mask2img(mask, img_name=f'mask_sep_img_{i}')
                mask_all = torch.where(
                    (mask_all == 0) | (mask == 0),
                    torch.tensor(0.0, device=img_size_masks.device),
                    torch.tensor(-1e9, device=img_size_masks.device)
                )
                #save_mask2img(mask_all, img_name=f'mask_all_img_{i}')
                prompt_mask_list.append(mask.to(device))
        #exit(0)
        #save_mask2img(mask_all, img_name=f'mask_all')
        joint_attention_kwargs['prompt_mask_list'] = prompt_mask_list
        joint_attention_kwargs['mask_all'] = mask_all.to(dtype=latents.dtype, device=latents.device)

    # 6. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps[start_steps:]): # start from collection steps
            # print(f'now start from t {t}')
            if self.interrupt:
                continue
            if i == collection_steps:
                break
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            if joint_attention_kwargs.get('global_prompt') is not None and i >= fuse_steps_start and i < fuse_steps_end:
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    encoder_hidden_states_base = encoder_hidden_states_base,
                    return_dict=False,
                    ##################################################
                    height=2 * (int(height) // (self.vae_scale_factor * 2)) // 2,
                    scene_text_ranges = scene_text_ranges
                    ##################################################
                )[0]
            else:
                if i == collection_steps - 1:
                    joint_attention_kwargs['is_last_collection'] = True
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    ##################################################
                    height=2 * (int(height) // (self.vae_scale_factor * 2)) // 2,
                    scene_text_ranges = scene_text_ranges
                    ##################################################
                )[0]  
                joint_attention_kwargs['is_last_collection'] = False
                    
            if do_true_cfg:
                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=negative_pooled_prompt_embeds,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    ##################################################
                    height=2 * (int(height) // (self.vae_scale_factor * 2)) // 2,
                    scene_text_ranges = scene_text_ranges
                    ##################################################
                )[0]
                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

        

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    if output_type == "latent":
        image = latents

    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()
    print(scene_text_ranges)

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)



def UNet2DConditionModelForward(
    self,
    sample: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[UNet2DConditionOutput, Tuple]:
    r"""
    The [`UNet2DConditionModel`] forward method.

    Args:
        sample (`torch.Tensor`):
            The noisy input tensor with the following shape `(batch, channel, height, width)`.
        timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
        encoder_hidden_states (`torch.Tensor`):
            The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
        class_labels (`torch.Tensor`, *optional*, defaults to `None`):
            Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
        timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
            Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
            through the `self.time_embedding` layer to obtain the timestep embeddings.
        attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
            is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
            negative values to the attention scores corresponding to "discard" tokens.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
            A tuple of tensors that if specified are added to the residuals of down unet blocks.
        mid_block_additional_residual: (`torch.Tensor`, *optional*):
            A tensor that if specified is added to the residual of the middle unet block.
        down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
            additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
        encoder_attention_mask (`torch.Tensor`):
            A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
            `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
            which adds large negative values to the attention scores corresponding to "discard" tokens.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
            tuple.

    Returns:
        [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
            otherwise a `tuple` is returned where the first element is the sample tensor.
    """
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None

    class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None:
        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    aug_emb = self.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )
    if self.config.addition_embed_type == "image_hint":
        aug_emb, hint = aug_emb
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    # 2.5 GLIGEN position net
    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

    # 3. down
    # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
    # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
    ################################################################################
    if cross_attention_kwargs is None:
        cross_attention_kwargs = {'timestep' : timestep}
    else:
        cross_attention_kwargs['timestep'] = timestep
    ################################################################################


    if cross_attention_kwargs is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        lora_scale = cross_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)

    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
    # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
    is_adapter = down_intrablock_additional_residuals is not None
    # maintain backward compatibility for legacy usage, where
    #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
    #       but can only use one or the other
    if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
        deprecate(
            "T2I should not use down_block_additional_residuals",
            "1.3.0",
            "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                    and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                    for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
            standard_warn=False,
        )
        down_intrablock_additional_residuals = down_block_additional_residuals
        is_adapter = True

    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                sample += down_intrablock_additional_residuals.pop(0)

        down_block_res_samples += res_samples

    if is_controlnet:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals
        ):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = self.mid_block(sample, emb)

        # To support T2I-Adapter-XL
        if (
            is_adapter
            and len(down_intrablock_additional_residuals) > 0
            and sample.shape == down_intrablock_additional_residuals[0].shape
        ):
            sample += down_intrablock_additional_residuals.pop(0)

    if is_controlnet:
        sample = sample + mid_block_additional_residual

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
            )

    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample)


def SD3Transformer2DModelForward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    pooled_projections: torch.FloatTensor = None,
    timestep: torch.LongTensor = None,
    block_controlnet_hidden_states: List = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    The [`SD3Transformer2DModel`] forward method.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    height, width = hidden_states.shape[-2:]

    hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
    temb = self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    for index_block, block in enumerate(self.transformer_blocks):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                **ckpt_kwargs,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
                ##########################################################################################
                timestep=timestep, height=height // self.config.patch_size,
                ##########################################################################################
            )

        # controlnet residual
        if block_controlnet_hidden_states is not None and block.context_pre_only is False:
            interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
            hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    # unpatchify
    patch_size = self.config.patch_size
    height = height // patch_size
    width = width // patch_size

    hidden_states = hidden_states.reshape(
        shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
    )

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


# modify
def FluxTransformer2DModelForward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
    ##################################################
    height: int = None,
    width: int = None,
    scene_text_ranges: Optional[list] = None,  # 新增参数
    encoder_hidden_states_base: torch.Tensor = None,
    ##################################################
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if encoder_hidden_states_base is not None:
        txt_ids_base = txt_ids
        encoder_hidden_states_base = self.context_embedder(encoder_hidden_states_base)
        
    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    # add vital layers
    if encoder_hidden_states_base is not None:
        ids_base = torch.cat((txt_ids_base, img_ids), dim=0)
        image_rotary_emb_base = self.pos_embed(ids_base)
    else:
        image_rotary_emb_base = None
    

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )

        else:

            joint_attention_kwargs['index_block'] = index_block
            if encoder_hidden_states_base is not None:
                encoder_hidden_states, hidden_states, encoder_hidden_states_base = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    
                    image_rotary_emb_base = image_rotary_emb_base,
                    encoder_hidden_states_base = encoder_hidden_states_base,
                    ##########################################################################################
                    timestep=timestep, height=height // self.config.patch_size,
                    scene_text_ranges= scene_text_ranges,  # 新增参数
                    ##########################################################################################
                )
            else:
                #print(f'now transformerfluxbloc cal')
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    # image_rotary_emb_base = image_rotary_emb_base,
                    # encoder_hidden_states_base = encoder_hidden_states_base,
                    ##########################################################################################
                    timestep=timestep, height=height // self.config.patch_size,
                    scene_text_ranges= scene_text_ranges,  # 新增参数
                    ##########################################################################################
                )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            # For Xlabs ControlNet.
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    #hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    if encoder_hidden_states_base is not None:
        hidden_states_base = torch.cat([encoder_hidden_states_base, hidden_states], dim=1)
    else:
        hidden_states_base = None
    #hidden_states_base = None
    
    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )

        else:
            #print(f'current is single block encoder_hidden_states shape is : {encoder_hidden_states.shape}')
            joint_attention_kwargs['index_block'] = index_block
            #print(joint_attention_kwargs['index_block'])
            # 拆分开来了
            if encoder_hidden_states_base is not None:
                encoder_hidden_states, hidden_states, hidden_states_base = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    #encoder_hidden_states_base = encoder_hidden_states_base,
                    hidden_states_base = hidden_states_base,
                    image_rotary_emb_base = image_rotary_emb_base,
                    # add parameter below
                    ##########################################################################################
                    timestep=timestep, height=height // self.config.patch_size,
                    scene_text_ranges= scene_text_ranges,  # 新增参数
                    ##########################################################################################
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    #encoder_hidden_states_base = encoder_hidden_states_base,
                    #hidden_states_base = hidden_states_base,
                    #image_rotary_emb_base = image_rotary_emb_base,
                    # add parameter below
                    ##########################################################################################
                    timestep=timestep, height=height // self.config.patch_size,
                    scene_text_ranges= scene_text_ranges,  # 新增参数
                    ##########################################################################################
                )

        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]


    #hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)



def FluxSingleTransformerBlockForward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ############################################################
    height: int = None,
    width: int = None,
    timestep: Optional[torch.Tensor] = None,
    scene_text_ranges: Optional[list] = None,  # 新增参数
    image_rotary_emb_base=None,
    encoder_hidden_states_base=None,
    hidden_states_base = None,
    ############################################################
) -> Tuple[torch.Tensor, torch.Tensor]:
    MULTIMODAL_VITAL_LAYERS = joint_attention_kwargs['MULTIMODAL_VITAL_LAYERS']
    SINGLE_MODAL_VITAL_LAYERS = joint_attention_kwargs['SINGLE_MODAL_VITAL_LAYERS']
    index_block = joint_attention_kwargs['index_block']
    if SINGLE_MODAL_VITAL_LAYERS is None or index_block not in SINGLE_MODAL_VITAL_LAYERS:
        text_seq_len = encoder_hidden_states.shape[1]
        # single hidden states 实现
        if hidden_states_base is not None:
        #if encoder_hidden_states_base is not None:
            #hidden_states_base = torch.cat([encoder_hidden_states_base, hidden_states], dim=1)
            residual_base = hidden_states_base
            norm_hidden_states_base, gate_base = self.norm(hidden_states_base, emb=temb)
            mlp_hidden_states_base = self.act_mlp(self.proj_mlp(norm_hidden_states_base))
        else:
            norm_hidden_states_base = None
            
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        
        joint_attention_kwargs = joint_attention_kwargs or {}
        #print(f'hidden_states shape: {norm_hidden_states.shape}')
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            ############################################################
            hidden_states_base=norm_hidden_states_base,
            image_rotary_emb_base=image_rotary_emb_base,
            timestep=timestep, height=height,
            scene_text_ranges = scene_text_ranges,
            ############################################################
            **joint_attention_kwargs,
        )
        
        # if hidden_states_base is not None:
        attn_output, attn_output_base = attn_output
        #print(f'att_output shape {attn_output.shape}')
        # else:
        #     attn_output = attn_output
        
    
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if hidden_states_base is not None:  
            hidden_states_base = torch.cat([attn_output_base, mlp_hidden_states_base], dim=2)
            gate_base = gate_base.unsqueeze(1)
            hidden_states_base = gate_base * self.proj_out(hidden_states_base)
            hidden_states_base = residual_base + hidden_states_base
            if hidden_states_base.dtype == torch.float16:
                hidden_states_base = hidden_states_base.clip(-65504, 65504)

            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            # encoder_hidden_states_base, hidden_states_base = hidden_states_base[:, :text_seq_len], hidden_states_base[:, text_seq_len:]


            #return encoder_hidden_states, hidden_states, encoder_hidden_states_base
            return encoder_hidden_states, hidden_states, hidden_states_base
    #print(f'hidden_states shape2: {norm_hidden_states.shape}')
    encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]

    return encoder_hidden_states, hidden_states


def FluxTransformerBlockForward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    ############################################################
    height: int = None,
    width: int = None,
    timestep: Optional[torch.Tensor] = None,
    scene_text_ranges: Optional[list] = None,  # 新增参数
    encoder_hidden_states_base = None,
    image_rotary_emb_base = None,
    ############################################################
):
    #print(f'now is in FluxTransformerBlockForward: {scene_text_ranges}')
    MULTIMODAL_VITAL_LAYERS = joint_attention_kwargs['MULTIMODAL_VITAL_LAYERS']
    SINGLE_MODAL_VITAL_LAYERS = joint_attention_kwargs['SINGLE_MODAL_VITAL_LAYERS']
    index_block = joint_attention_kwargs['index_block']
    if MULTIMODAL_VITAL_LAYERS is None or index_block not in MULTIMODAL_VITAL_LAYERS:
        
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        
        if encoder_hidden_states_base is not None:
            norm_encoder_hidden_states_base, c_gate_msa_base, c_shift_mlp_base, c_scale_mlp_base, c_gate_mlp_base = self.norm1_context(
                encoder_hidden_states_base, emb=temb
            )
            
        
        joint_attention_kwargs = joint_attention_kwargs or {}

        if encoder_hidden_states_base is not None:
            attn_output, context_attn_output, context_attn_output_base = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                encoder_hidden_states_base = norm_encoder_hidden_states_base,
                image_rotary_emb=image_rotary_emb,
                ############################################################
                timestep=timestep, height=height,
                scene_text_ranges = scene_text_ranges,
                image_rotary_emb_base = image_rotary_emb_base,
                ############################################################
                **joint_attention_kwargs,
            )
        else:
            attn_output, context_attn_output, context_attn_output_base = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                ############################################################
                timestep=timestep, height=height,
                scene_text_ranges = scene_text_ranges,
                #image_rotary_emb_base = image_rotary_emb_base,
                ############################################################
                **joint_attention_kwargs,
            ) 

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        # 内部更新了
        if encoder_hidden_states_base is not None:
        # if encoder_hidden_states_base is not None:  
            context_attn_output_base = c_gate_msa_base.unsqueeze(1) * context_attn_output_base
            encoder_hidden_states_base = encoder_hidden_states_base + context_attn_output_base

            norm_encoder_hidden_states_base = self.norm2_context(encoder_hidden_states_base)
            norm_encoder_hidden_states_base = norm_encoder_hidden_states_base * (1 + c_scale_mlp_base[:, None]) + c_shift_mlp_base[:, None]

            context_ff_output_base = self.ff_context(norm_encoder_hidden_states_base)
            encoder_hidden_states_base = encoder_hidden_states_base + c_gate_mlp_base.unsqueeze(1) * context_ff_output_base
            if encoder_hidden_states_base.dtype == torch.float16:
                encoder_hidden_states_base = encoder_hidden_states_base.clip(-65504, 65504)
            joint_attention_kwargs['global_prompt_embeds'] = encoder_hidden_states_base
        
        if encoder_hidden_states_base is not None:
            return encoder_hidden_states, hidden_states, encoder_hidden_states_base
        else:
            return encoder_hidden_states, hidden_states        



def attn_call(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        height: int = None,
        width: int = None,
        timestep: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        ####################################################################################################
        if hasattr(self, "store_attn_map"):
            self.attn_map = rearrange(attention_probs, 'b (h w) d -> b d h w', h=height)
            self.timestep = int(timestep.item())
        ####################################################################################################
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight


def attn_call2_0(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    height: int = None,
    width: int = None,
    timestep: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    ####################################################################################################
    if hasattr(self, "store_attn_map"):
        hidden_states, attention_probs = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        self.attn_map = rearrange(
            attention_probs,
            'batch attn_head (h w) attn_dim -> batch attn_head h w attn_dim ',
            h=height
        ) # detach height*width
        self.timestep = int(timestep.item())
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    ####################################################################################################

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # (b,attn_head,h*w,attn_dim) -> (b,h*w,attn_head*attn_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def lora_attn_call(self, attn: Attention, hidden_states, height, width, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor()
    ####################################################################################################
    attn.processor.__call__ = attn_call.__get__(attn.processor, AttnProcessor)
    ####################################################################################################

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, height, width, *args, **kwargs)


def lora_attn_call2_0(self, attn: Attention, hidden_states, height, width, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor2_0()
    ####################################################################################################
    attn.processor.__call__ = attn_call.__get__(attn.processor, AttnProcessor2_0)
    ####################################################################################################

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, height, width, *args, **kwargs)


def joint_attn_call2_0(
    self,
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    ############################################################
    height: int = None,
    timestep: Optional[torch.Tensor] = None,
    ############################################################
    *args,
    **kwargs,
) -> torch.FloatTensor:
    residual = hidden_states

    batch_size = hidden_states.shape[0]

    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # `context` projections.
    if encoder_hidden_states is not None:
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

    ####################################################################################################
    if hasattr(self, "store_attn_map") and encoder_hidden_states is not None:
        hidden_states, attention_probs = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        image_length = query.shape[2] - encoder_hidden_states_query_proj.shape[2]

        # (4,24,4429,4429) -> (4,24,4096,333)
        attention_probs = attention_probs[:,:,:image_length,image_length:].cpu()
        
        self.attn_map = rearrange(
            attention_probs,
            'batch attn_head (height width) attn_dim -> batch attn_head height width attn_dim',
            height = height
        ) # (4, 24, 4096, 333) -> (4, 24, height, width, 333)
        self.timestep = timestep[0].cpu().item() # TODO: int -> list
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    ####################################################################################################

    # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if encoder_hidden_states is not None:
        return hidden_states, encoder_hidden_states
    else:
        return hidden_states

import math
from torch.nn import functional as F
from einops import rearrange

import torch
import matplotlib.pyplot as plt
import os
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def save_mask2img(mask, img_name='mask_debug'):

    os.makedirs("mask_debug", exist_ok=True)


    mask_vis = mask[0]  # [H, Q, K]


    mask_vis_mean = mask_vis.mean(dim=0)  # [Q, K]


    min_val = mask_vis_mean.min()
    max_val = mask_vis_mean.max()
    if (max_val - min_val) < 1e-8:
        mask_img = torch.zeros_like(mask_vis_mean)
    else:
        mask_img = (mask_vis_mean - min_val) / (max_val - min_val)

    mask_np = mask_img.to(torch.float32).cpu().numpy()

    q_len, k_len = mask_np.shape
    x = np.arange(k_len)
    y = np.arange(q_len)

    plt.figure(figsize=(8, 8))
    plt.imshow(mask_np, cmap="viridis", interpolation="nearest",
               extent=[x[0], x[-1], y[-1], y[0]]) 
    plt.title(f"Attention Mask (Accurate Axes)\nmin={min_val.item():.2e}, max={max_val.item():.2e}")
    plt.xlabel("Key token index (K)")
    plt.ylabel("Query token index (Q)")
    plt.colorbar(label="Normalized Mask Value")

    step_x = max(1, k_len // 10)
    step_y = max(1, q_len // 10)
    plt.xticks(np.arange(0, k_len, step_x))
    plt.yticks(np.arange(0, q_len, step_y))

    plt.tight_layout()
    plt.savefig(f"mask_debug/{img_name}.png", dpi=300)
    plt.close()


# FluxAttnProcessor2_0
def flux_attn_call2_0_before(
    self,
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    ############################################################
    height: int = None,
    timestep: Optional[torch.Tensor] = None,
    ############################################################
) -> torch.FloatTensor:
    batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb
        

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)
    
    ####################################################################################################
    if hasattr(self, "store_attn_map") and encoder_hidden_states is not None:
        hidden_states, attention_probs = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        text_length = encoder_hidden_states_query_proj.shape[2]

        # (1,24,4608,4608) -> (1,24,4096,512)
        attention_probs = attention_probs[:,:,text_length:,:text_length].cpu() # 保留 img-txt attention部分
        
        self.attn_map = rearrange(
            attention_probs,
            'batch attn_head (height width) attn_dim -> batch attn_head height width attn_dim',
            height = height
        ) # (1,24,4096,512) -> (1,24,height,width,512)
        self.timestep = timestep[0].cpu().item() # TODO: int -> list
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    ####################################################################################################
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
    else:
        return hidden_states
    
import re
import torch
def split_prompt_into_scenes_and_ranges(
    full_prompt: str,
    tokenizer,
    num_scenes: int = 4,
    is_use_global_prompt: bool = False,
):

    pattern = r"(Panel\s+\d+:)"
    parts = re.split(pattern, full_prompt)
    scene_texts = []

    scene_text_ranges = []
    global_text = ''
    if is_use_global_prompt:
        global_text = parts[0].strip()
        global_ids = tokenizer(global_text, return_tensors="pt").input_ids[0] if global_text else torch.tensor([], dtype=torch.long)
        global_len = len(global_ids)
        current_pos = global_len - 1
        scene_texts.append(global_text)
        scene_text_ranges.append((0, global_len - 1))
    else:
        current_pos = 0
        
    for i in range(1, len(parts), 2):
        tag = parts[i].strip()
        content = parts[i + 1].strip() if (i + 1) < len(parts) else ""
        content = content.split('.')[0].strip()
        content = content+'.'
        scene_texts.append(f"{tag} {content}")
    
    input_ids_all = tokenizer(full_prompt, return_tensors="pt").input_ids[0]
    #print(input_ids_all)
    total_len = len(input_ids_all)

        

    # ---------- 3️⃣ 精确匹配 ----------


    for scene_text in scene_texts:
        #print(f'curr scene_text: {scene_text}')
        scene_ids = tokenizer(scene_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        #print(f'currr scene_text len: {len(scene_ids)}')
        for i in range(current_pos, total_len - len(scene_ids) + 1):
            #print(f'curr equal id : {input_ids_all[i:i + len(scene_ids)]}.  scene equal id: {scene_ids}')
            if torch.equal(input_ids_all[i:i + len(scene_ids)], scene_ids):
                #print(f'curr equal id : {input_ids_all[i:i + len(scene_ids)]}.  scene equal id: {scene_ids}')
                scene_text_ranges.append((i, i + len(scene_ids)))
                current_pos = i + len(scene_ids)
                break

    
    return scene_texts, scene_text_ranges, global_text