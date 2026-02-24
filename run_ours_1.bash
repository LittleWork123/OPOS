
CUDA_VISIBLE_DEVICES=2 python flux_gen_by_yaml.py \
  --yaml_file "./assets/combined_stories_dataset_processed.yaml" \
  --output_base_dir "./ours/ours_consis++_global_0.5" \
  --alpha 1.5 \
  --beta 1.3 \
  --collection_steps 13 \
  --base_radio 0.5 \
  --alpha 1.2
  #--save_attention \


# CUDA_VISIBLE_DEVICES=1 python flux_gen_by_yaml.py \
#   --yaml_file "./assets/combined_stories_dataset_processed.yaml" \
#   --output_base_dir "./ours/flux_consis++_global_0.25" \
#   --alpha 1.5 \
#   --beta 1.3 \
#   --collection_steps 13 \
#   --base_radio 0.25
#   #--save_attention \
