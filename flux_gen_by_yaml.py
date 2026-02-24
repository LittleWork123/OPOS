import os
import torch
import argparse
import yaml
import random
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from pytorch_lightning import seed_everything
from diffusers import FluxPipeline
import re
import math
import numpy as np
import time

from flux_attention.attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)

from flux_attention.attention_map_diffusers import (
    split_prompt_into_scenes_and_ranges,
    start_attention_collection,
    stop_attention_collection,
    get_average_attention,
    compute_scene_token_masks,
    set_scene_token_masks,
    clear_scene_token_masks,
)

class TokenFinder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_token_indices(self, text: str, target_token: str):
        if not target_token or not text:
            return []
        
        full_tokens = self.tokenizer(text, return_tensors="pt")['input_ids'][0]
        target_tokens = self.tokenizer(target_token, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
        
        target_token_ids = (
            target_tokens[1:-1].tolist() 
            if len(target_tokens) > 2 
            else target_tokens.tolist()
        )
        
        if not target_token_ids:
            return []
        
        full_list = full_tokens.tolist()
        indices = []
        
        for i in range(len(full_list) - len(target_token_ids) + 1):
            if full_list[i:i+len(target_token_ids)] == target_token_ids:
                indices.extend(list(range(i, i + len(target_token_ids))))
                print(f'Match found in "{text}" for "{target_token}" at indices {i} : {i + len(target_token_ids)}')
                break 
        return list(set(indices))

def read_yaml_config(filepath):
    entries = []
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        for key, items in data.items():
            if not isinstance(items, list): continue
            for entry in items:
                if isinstance(entry, dict): entries.append(entry)
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict): entries.append(entry)
    return entries

def build_grid_prompt(entry, is_use_global_prompt=False):
    subject = (entry.get("subject") or "").strip()
    concept = (entry.get("subject_1") or "").strip()
    style = (entry.get("style") or "").strip()
    settings = entry.get("settings") or []
    partner_token = entry.get("subject_2", "").strip()

    if not settings:
        settings = [""] * 4
    else:
        settings = settings[:4]
        if len(settings) < 4:
            settings += [settings[-1]] * (4 - len(settings))
    
    panels = []
    for i, s in enumerate(settings, 1):
        s = (s or "").strip().lower()
        if style and subject:
            panels.append(f"Panel {i}: {style} {subject}, {s}." if s else f"Panel {i}: {style} {subject}.")
        else:
            panels.append(f"Panel {i}: {s}." if s else f"Panel {i}: .")
    
    panel_text = " ".join(panels)
    base_prompt = (
        f"A 2x2 panel grid photo sequence about {subject}. "
        f"Each panel shows the same {concept} and {partner_token} in different scenes, "
        f"keeping consistent appearance, color tone, and camera style. "
        f"{panel_text} "
    )
    prompt = base_prompt if is_use_global_prompt else panel_text
    return " ".join(prompt.split()), " ".join(base_prompt.split())

def split_2x2_grid(grid_image, output_dir, base_name, panel_prompts):
    img_dir = os.path.join(output_dir, "img")
    txt_dir = os.path.join(output_dir, "txt")
    grid_dir = os.path.join(output_dir, "grid")
    
    os.makedirs(img_dir, exist_ok=True); os.makedirs(txt_dir, exist_ok=True); os.makedirs(grid_dir, exist_ok=True)
    
    grid_path = os.path.join(grid_dir, f"{base_name}_2x2_grid.png")
    grid_image.save(grid_path)

    width, height = grid_image.size
    pw, ph = width // 2, height // 2
    positions = [(0, 0, pw, ph, "0"), (pw, 0, width, ph, "1"), (0, ph, pw, height, "2"), (pw, ph, width, height, "3")]
    
    for idx, (x1, y1, x2, y2, panel_id) in enumerate(positions):
        panel_image = grid_image.crop((x1, y1, x2, y2))
        panel_image.save(os.path.join(img_dir, f"{base_name}_panel_{panel_id}.png"))
        with open(os.path.join(txt_dir, f"{base_name}_panel_{panel_id}.txt"), 'w', encoding='utf-8') as f:
            f.write(panel_prompts[idx] if idx < len(panel_prompts) else "")

def get_scene_token_indices(scene_texts: list, concept_token: str, partner_token: str, tokenizer) -> list:
    token_finder = TokenFinder(tokenizer)
    scene_token_indices = []
    for scene_idx, scene_text in enumerate(scene_texts):
        concept_indices = token_finder.get_token_indices(scene_text, concept_token)
        partner_indices = token_finder.get_token_indices(scene_text, partner_token)
        scene_token_indices.append((concept_indices, partner_indices))
    return scene_token_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridStory: Two-stage Multi-subject Visual Story Generation")
    parser.add_argument("--yaml_file", type=str, required=True)
    parser.add_argument("--output_base_dir", type=str, required=True)
    parser.add_argument("--save_attention", action="store_true")
    parser.add_argument("--is_use_global_prompt", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.5, help="Token mask enhancement coefficient")
    parser.add_argument("--collection_steps", type=int, default=7)
    parser.add_argument("--total_steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_grid", type=int, default=4)
    parser.add_argument("--wo_structual_mask", action="store_true")
    parser.add_argument("--full_denoise_steps", type=int, default=13)
    parser.add_argument("--base_radio", type=float, default=0.2)
    parser.add_argument("--double_layers", type=list, default=[])
    parser.add_argument("--single_layers", type=list, default=[])
    parser.add_argument("--fuse_steps_start", type=int, default=0)
    parser.add_argument("--fuse_steps_end", type=int, default=28)
    parser.add_argument("--is_w_token_masks", action="store_true", help="Enable two-stage token masking")
    parser.add_argument("--enable_dynamic_token_mask", action="store_true")
    parser.add_argument("--is_single_subject", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_base_dir, exist_ok=True)
    seed_everything(args.seed)
    entries = read_yaml_config(args.yaml_file)
    
    print("🌀 Loading FLUX Model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.bfloat16
    ).to("cuda")

    print("⭐ Initializing Pipeline...")
    pipe = init_pipeline(pipe)

    # Attention Mask Setup
    B, C, H, W = 1, 24, 64, 64
    place = [(0, 0, 512, 512), (512, 0, 1024, 512), (0, 512, 512, 1024), (512, 512, 1024, 1024)]
    img_masks = []
    for (x1, y1, x2, y2) in place:
        mask = torch.ones((1024, 1024)) * -1e9
        mask[y1:y2, x1:x2] = 0
        mask = torch.nn.functional.interpolate(mask[None, None, :, :], (H, W), mode='nearest-exact').flatten()
        img_masks.append(mask.unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 512))

    MULTIMODAL_VITAL_LAYERS = [int(x) for x in args.double_layers]
    SINGLE_MODAL_VITAL_LAYERS = [int(x) for x in args.single_layers]

    joint_attention_kwargs = {
        'img_masks': img_masks, 'beta': args.beta, 'full_denoise_steps': args.full_denoise_steps,
        'base_radio': args.base_radio, 'MULTIMODAL_VITAL_LAYERS': MULTIMODAL_VITAL_LAYERS,
        'SINGLE_MODAL_VITAL_LAYERS': SINGLE_MODAL_VITAL_LAYERS, 'fuse_steps_start': args.fuse_steps_start,
        'fuse_steps_end': args.fuse_steps_end
    }
    
    joint_attention_kwargs_before = joint_attention_kwargs.copy()
    joint_attention_kwargs_before.update({'fuse_steps_start': 0, 'fuse_steps_end': 28})

    for entry_idx, entry in enumerate(entries):
        print(f"\n🎬 Processing Story {entry_idx+1}...")
        
        concept_token = entry.get("concept_token" if args.is_single_subject else "subject_1", "").strip()
        partner_token = entry.get("subject_2", "").strip()
        
        prompt, global_prompt = build_grid_prompt(entry, args.is_use_global_prompt)
        scene_texts, scene_text_ranges, _ = split_prompt_into_scenes_and_ranges(
            prompt, pipe.tokenizer_2, num_scenes=args.num_grid, is_use_global_prompt=args.is_use_global_prompt
        )
        
        scene_token_indices = get_scene_token_indices(scene_texts, concept_token, partner_token, pipe.tokenizer_2)
        
        if args.is_w_token_masks:
            print("🎨 Stage 1: Collecting Attention Maps...")
            start_attention_collection()
            joint_attention_kwargs_before.update({'fuse_steps_start': args.fuse_steps_start, 'fuse_steps_end': args.fuse_steps_start, 'get_token_mask': True})
            
            image_latent = pipe(
                prompt=prompt, prompt_2=prompt, height=args.height, width=args.width,
                guidance_scale=7.0, num_inference_steps=args.total_steps,
                collection_steps=args.collection_steps, output_type='latent',
                joint_attention_kwargs=joint_attention_kwargs_before,
            ).images
            stop_attention_collection()
            
            avg_attn = get_average_attention()
            token_masks_dict, scene_img_token_indices = compute_scene_token_masks(
                avg_attention=avg_attn, scene_text_ranges=scene_text_ranges,
                scene_concept_indices=scene_token_indices, num_scenes=4,
                mask_save_dir=os.path.join(args.output_base_dir, 'mask_save'),
                concept_order=entry_idx, enable_dynamic_token_mask=args.enable_dynamic_token_mask,
            )
            joint_attention_kwargs.update({'scene_concept_indices': scene_token_indices, 'scene_img_token_indices': scene_img_token_indices})
            set_scene_token_masks(token_masks_dict, alpha=args.alpha)
            print("🎨 Stage 2: Denoising with Token Masks...")
        else:
            image_latent = None

        joint_attention_kwargs.update({'global_prompt': global_prompt, 'fuse_steps_start': args.fuse_steps_start, 'fuse_steps_end': args.fuse_steps_start})
        
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        image = pipe(
            prompt=prompt, prompt_2=prompt, latents=image_latent, height=args.height, width=args.width,
            guidance_scale=7.0, num_inference_steps=args.total_steps,
            start_steps=args.collection_steps if args.is_w_token_masks else 0,
            joint_attention_kwargs=joint_attention_kwargs, is_use_global_prompt=args.is_use_global_prompt
        ).images[0]
        
        torch.cuda.synchronize()
        print(f"⏱ Inference Time: {time.time() - start_time:.4f}s | Peak Mem: {torch.cuda.max_memory_allocated()/1024**2:.2f}MB")

        if args.is_w_token_masks: clear_scene_token_masks()

        base_name = f"story_{entry_idx+1}_{re.sub(r'[\\/*?:\u0022<>|]', '_', concept_token[:20])}"
        split_2x2_grid(image, args.output_base_dir, base_name, scene_texts)
        
        if args.save_attention:
            save_attention_maps(attn_maps, pipe.tokenizer, prompt, base_dir=os.path.join(args.output_base_dir, 'atten_map', f'{entry_idx}'))

        pipe = init_pipeline(pipe)

    print(f"✅ Finished! Results saved to: {args.output_base_dir}")