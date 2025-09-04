# do bump the workers amount if you do not build decord with gpu

# Using cpu:
python make_dataset.py \
    --prompt_style 2 \
    --dataset_dir datasets \
    --dataset_name qvhighlights \
    --num_frames 25 \
    --num_workers 6 \
    --pretty_json \
    --processed_dir qvhighlights-full-f25-p2-v3 \
    --frame_variation 3 
    # --num_train_samples 32 \
    # --num_val_samples 16 \
    # --num_test_samples 16 

# Using gpu:
# python make_dataset.py \
#     --prompt_style 2 \
#     --dataset_dir datasets \
#     --dataset_name qvhighlights \
#     --num_frames 25 \
#     --num_workers 6 \
#     --pretty_json \
#     --processed_dir 1k-25fps-p2-v5 \
#     --frame_variation 3 \
#     --gpu