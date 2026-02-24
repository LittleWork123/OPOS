YAML_SINGLE_CONFIG="/home/data/liuqi/code/Test/assets/combined_stories_dataset_processed.yaml"
OUTPUT_DIR="./ours_consis++_real"
TEST='/home/data/liuqi/code/Test/assets/test_single.yaml'
base_radio=0.3
collection_steps=10
alpha=1.4
fuse_steps_start=10
fuse_steps_end=15
seed=2025
CUDA_VISIBLE_DEVICES=5 python flux_gen_by_yaml.py \
  --yaml_file $YAML_SINGLE_CONFIG \
  --output_base_dir "$OUTPUT_DIR/ours seed $seed token_mask_coll $collection_steps radio $base_radio alpha $alpha s $fuse_steps_start e $fuse_steps_end" \
  --collection_steps $collection_steps --base_radio $base_radio --alpha $alpha \
  --fuse_steps_start $fuse_steps_start \
  --fuse_steps_end $fuse_steps_end \
  --seed $seed \
  --is_w_token_masks \
  # --is_single_subject
  # --enable_dynamic_token_mask 