EXPMT_NAME="."

allennlp evaluate \
    "${EXPMT_NAME}/models/model.tar.gz" \
    "../../data/az/az_papers/tag_bio_filt_len_14062023/test.jsonl" \
    --output-file "${EXPMT_NAME}/test_eval_metrics.json" \
    --predictions-output "${EXPMT_NAME}/test_predictions.jsonl" \
    --cuda-device 0 \
    --batch-size 128 \
    --overrides '{"model.top_k": 5}' \
    --include-package classifier
