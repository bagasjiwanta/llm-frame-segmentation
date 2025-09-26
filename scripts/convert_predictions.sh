

python blip3_mr/eval_utils.py \
    --input_path run/val_and_test_all/hl_test_predictions.json \
    --output_path run/val_and_test_all/ranked_top1/hl_test_submission.jsonl \
    --clean \
    --num_proposal 10 \
    --video_path datasets/qvhighlights/highlight_test_release.jsonl

python blip3_mr/eval_utils.py \
    --input_path run/val_and_test_all/hl_val_predictions.json \
    --output_path run/val_and_test_all/ranked_top1/hl_val_submission.jsonl \
    --clean \
    --num_proposal 10 \
    --video_path datasets/qvhighlights/highlight_val_release.jsonl