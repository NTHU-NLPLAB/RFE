EXPMT_NAME="."

allennlp evaluate \
    "${EXPMT_NAME}/models/model.tar.gz" \
    "../../data/az/az_papers/tag_bio_filt_len_clb_14062023_strict/test.jsonl" \
    --output-file "${EXPMT_NAME}/test_own_eval_metrics.json" \
    --predictions-output "${EXPMT_NAME}/test_own_predictions.jsonl" \
    --cuda-device 0 \
    --batch-size 128 \
    --overrides '{"model.top_k": 5}' \
    --include-package classifier
