export PYTHONPATH=.

python blip3_mr/eval_mr.py \
    --submission_path run/val_and_test_all/ranked_top1/hl_val_submission.jsonl \
    --gt_path datasets/qvhighlights/highlight_val_release.jsonl \
    --save_path run/val_and_test_all/ranked_top1/metrics.json

python blip3_mr/eval_mr.py \
    --submission_path datasets/qvhighlights/sample_val_preds.jsonl \
    --gt_path datasets/qvhighlights/highlight_val_release.jsonl 