import os

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0
)
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

from .modules import *


def hook_function(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):

            timestep = module.processor.timestep

            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach \
                else module.processor.attn_map
            
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(model, hook_function, target_name, is_origin=False):
    replaced = 0
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue
        # print(name, type(module.processor))
        module.processor.store_attn_map = True
        if is_origin:
            module.processor.__class__.__call__ = flux_attn_call2_0_before
        else:
            module.processor.__class__.__call__ = flux_attn_call2_0
        replaced += 1
        print(f"  ↳ 已替换: {name}.processor → flux_attn_call2_0")
        # if isinstance(module.processor, AttnProcessor):
        #     module.processor.store_attn_map = True
        # elif isinstance(module.processor, AttnProcessor2_0):
        #     module.processor.store_attn_map = True
        # elif isinstance(module.processor, LoRAAttnProcessor):
        #     module.processor.store_attn_map = True
        # elif isinstance(module.processor, LoRAAttnProcessor2_0):
        #     module.processor.store_attn_map = True
        # elif isinstance(module.processor, JointAttnProcessor2_0):
        #     module.processor.store_attn_map = True
        # elif isinstance(module.processor, FluxAttnProcessor2_0):
        #     module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_function(name))

    return model


def replace_call_method_for_unet(model):
    if model.__class__.__name__ == 'UNet2DConditionModel':
        from diffusers.models.unets import UNet2DConditionModel
        model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'Transformer2DModel':
            from diffusers.models import Transformer2DModel
            layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)
        
        elif layer.__class__.__name__ == 'BasicTransformerBlock':
            from diffusers.models.attention import BasicTransformerBlock
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)
        
        replace_call_method_for_unet(layer)
    
    return model




def replace_call_method_for_sd3(model):
    if model.__class__.__name__ == 'SD3Transformer2DModel':
        from diffusers.models.transformers import SD3Transformer2DModel
        model.forward = SD3Transformer2DModelForward.__get__(model, SD3Transformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'JointTransformerBlock':
            from diffusers.models.attention import JointTransformerBlock
            layer.forward = JointTransformerBlockForward.__get__(layer, JointTransformerBlock)
        
        replace_call_method_for_sd3(layer)
    
    return model


def replace_call_method_for_flux(model):
    if model.__class__.__name__ == 'FluxTransformer2DModel':
        from diffusers.models.transformers import FluxTransformer2DModel
        model.forward = FluxTransformer2DModelForward.__get__(model, FluxTransformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'FluxTransformerBlock':
            from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
            layer.forward = FluxTransformerBlockForward.__get__(layer, FluxTransformerBlock)
        if layer.__class__.__name__ == 'FluxSingleTransformerBlock':
            from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
            layer.forward = FluxSingleTransformerBlockForward.__get__(layer, FluxSingleTransformerBlock)        
        
        replace_call_method_for_flux(layer)
    
    return model


def init_origin_processor(pipeline):
    
    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'SD3Transformer2DModel':
            JointAttnProcessor2_0.__call__ = joint_attn_call2_0
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)
        
        elif pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            from diffusers import FluxPipeline
            FluxAttnProcessor2_0.__call__ = FluxAttnProcessor.__call__
            FluxPipeline.__call__ = FluxPipeline_call
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn', is_origin=True)

def init_pipeline(pipeline, origin=False, skip_layers=None):
    attn_maps = {}
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call2_0
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0
    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'SD3Transformer2DModel':
            JointAttnProcessor2_0.__call__ = joint_attn_call2_0
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)
        
        elif pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            from diffusers import FluxPipeline
            if origin==False:
                FluxAttnProcessor2_0.__call__ = flux_attn_call2_0
            else:
                FluxAttnProcessor2_0.__call__ = flux_attn_call2_0_before
            FluxPipeline.__call__ = FluxPipeline_call
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn', origin)
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)

        # TODO: implement
        # elif pipeline.transformer.__class__.__name__ == 'SanaTransformer2DModel':
        #     from diffusers import SanaPipeline
        #     SanaPipeline.__call__ == SanaPipeline_call
        #     pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn2')
        #     pipeline.transformer = replace_call_method_for_sana(pipeline.transformer)

    else:
        if pipeline.unet.__class__.__name__ == 'UNet2DConditionModel':
            pipeline.unet = register_cross_attention_hook(pipeline.unet, hook_function, 'attn2')
            pipeline.unet = replace_call_method_for_unet(pipeline.unet)


    return pipeline


def process_token(token, startofword):
    if '</w>' in token:
        token = token.replace('</w>', '')
        if startofword:
            token = '<' + token + '>'
        else:
            token = '-' + token + '>'
            startofword = True
    elif token not in ['<|startoftext|>', '<|endoftext|>']:
        if startofword:
            token = '<' + token + '-'
            startofword = False
        else:
            token = '-' + token + '-'
    return token, startofword


# def save_attention_image(attn_map, tokens, batch_dir, to_pil):
#     startofword = True
#     for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
#         token, startofword = process_token(token, startofword)
#         to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}.png'))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def save_attention_image(attn_map, tokens, batch_dir, to_pil):
    """
    保存彩色 attention 图。每个 token 对应一张彩色热力图。
    """
    font = ImageFont.load_default()
    startofword = True
    
    for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
        token, startofword = process_token(token, startofword)

        # === 归一化到 [0,1] ===
        a = a.to(torch.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        a_np = (a.cpu().numpy() * 255).astype(np.uint8)

        # === 转为彩色图（伪彩色映射）===
        color_map = cv2.applyColorMap(a_np, cv2.COLORMAP_TURBO)  # 可改成 COLORMAP_JET / VIRIDIS
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)   # OpenCV 是 BGR，需要转成 RGB
        img_pil = Image.fromarray(color_map)

        # # === 可选：添加 token 文本 ===
        # draw = ImageDraw.Draw(img_pil)
        # draw.text((5, 5), token, font=font, fill=(255, 255, 255))

        # === 保存 ===
        filename = os.path.join(batch_dir, f'{i:02d}-{token}.png')
        img_pil.save(filename)

def save_attention_maps(attn_maps, tokenizer, prompts, base_dir='attn_maps', unconditional=True):
    to_pil = ToPILImage()
    #print(f'2222 {prompts}')
    token_ids = tokenizer(prompts)['input_ids']
    token_ids = token_ids if token_ids and isinstance(token_ids[0], list) else [token_ids]
    total_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]
    #print(f'3333 {total_tokens}')
    os.makedirs(base_dir, exist_ok=True)

    total_attn_map = list(list(attn_maps.values())[0].values())[0].sum(1)
    if unconditional:
        total_attn_map = total_attn_map.chunk(2)[1]  # (batch, height, width, attn_dim)
    total_attn_map = total_attn_map.permute(0, 3, 1, 2)
    total_attn_map = torch.zeros_like(total_attn_map)
    total_attn_map_shape = total_attn_map.shape[-2:]
    total_attn_map_number = 0
    
    for timestep, layers in attn_maps.items():
        timestep_dir = os.path.join(base_dir, f'{timestep}')
        os.makedirs(timestep_dir, exist_ok=True)
        
        for layer, attn_map in layers.items():
            layer_dir = os.path.join(timestep_dir, f'{layer}')
            os.makedirs(layer_dir, exist_ok=True)
            
            attn_map = attn_map.sum(1).squeeze(1).permute(0, 3, 1, 2)
            if unconditional:
                attn_map = attn_map.chunk(2)[1]
            
            resized_attn_map = F.interpolate(attn_map, size=total_attn_map_shape, mode='bilinear', align_corners=False)
            total_attn_map += resized_attn_map
            total_attn_map_number += 1
            
            for batch, (tokens, attn) in enumerate(zip(total_tokens, attn_map)):
                batch_dir = os.path.join(layer_dir, f'batch-{batch}')
                os.makedirs(batch_dir, exist_ok=True)
                save_attention_image(attn, tokens, batch_dir, to_pil)
    
    total_attn_map /= total_attn_map_number
    for batch, (attn_map, tokens) in enumerate(zip(total_attn_map, total_tokens)):
        batch_dir = os.path.join(base_dir, f'batch-{batch}')
        os.makedirs(batch_dir, exist_ok=True)
        save_attention_image(attn_map, tokens, batch_dir, to_pil)

