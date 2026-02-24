
# GridStory: Towards Coherent Multi-Subject Visual Stories via Grid-Based One Diffusion Process
![GridStory ](./assets/mainwork.jpg)

This is the official implementation of **GridStory**, a novel grid-based diffusion framework designed for multi-subject Visual Story Generation (VSG). By leveraging the native capabilities of the FLUX model, GridStory achieves consistent, multi-frame story synthesis in a single denoising process.

## 📖 Abstract

Visual Story Generation (VSG) has seen significant progress, yet maintaining identity consistency across multiple subjects remains a challenge. Existing methods often suffer from **inter-subject attribute confusion**, **inter-frame semantic leakage**, and **high computational overhead**.↳

**GridStory** addresses these by introducing:

1. **Dual-Stream Attention:** Disentangles identity consistency from scene alignment.
2. **Pseudo-mask Guided Deconfusion:** Utilizes structural clarity from early denoising attention maps to eliminate interference between subjects.
3. **O(1) Efficiency:** Unifies multi-frame synthesis into a single diffusion process, enabling parallel generation without per-subject fine-tuning.

## ✨ Key Features

- **Multi-Subject Consistency:** Keeps both primary and secondary subjects (e.g., "a cat and a dog") consistent across the entire narrative.
- **Zero Fine-tuning:** No need for LoRA or Dreambooth training; works out-of-the-box via attention manipulation.
- **High Efficiency:** Generates a 4-panel story sequence in nearly the same time as a single image.
- **Attribute Deconfusion:** Prevents visual features (like color or texture) from "leaking" between different subjects.

## 🛠 Installation

```jsx

# Create environment
conda create -n gridstory python=3.10
conda activate gridstory

# Install dependencies
pip install torch torchvision diffusers transformers pytorch-lightning pyyaml
```

## 🚀 Quick Start

### 1. Configuration

Define your story in a YAML file (e.g., `configs/demo.yaml`):

```jsx
stories:
  - subject: "a boy and his robot"
    subject_1: "boy"      # Primary subject
    subject_2: "robot"    # Secondary subject
    style: "cyberpunk oil painting of"
    settings:
      - "walking through a neon city"
      - "fixing a mechanical circuit"
      - "sharing an umbrella in the rain"
      - "looking at the sunset from a rooftop"
```

### 2. Execution

Use the following command to generate the grid story. The parameters are tuned for optimal identity preservation:

```jsx
CUDA_VISIBLE_DEVICES=0 python flux_gen_by_yaml.py \
  --yaml_file ./configs/demo.yaml \
  --output_base_dir "./outputs/grid_results" \
  --is_w_token_masks \
  --collection_steps 7 \
  --base_radio 0.2 \
  --alpha 1.5 \
  --fuse_steps_start 0 \
  --fuse_steps_end 28 \
  --seed 42
```

## ⚙️ Hyperparameter Guide

| **Parameter** | **Default** | **Function** |
| --- | --- | --- |
| `--collection_steps` | `7` | Steps for Stage-1 attention map collection. Higher values yield more precise masks. |
| `--base_radio` | `0.2` | **Critical:** Controls the global layout foundation. Higher values enhance overall grid structure; lower values allow more local scene detail. |
| `--alpha` | `1.5` | **Token Mask Strength:** Increases the focus on specified subjects to prevent identity fading. |
| `--fuse_steps_start/end` | `0 / 28` | Defines the temporal window for the dual-stream attention fusion. |