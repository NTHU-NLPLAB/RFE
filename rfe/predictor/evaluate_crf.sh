EXPMT_NAME="experiments/crf-tagger"

allennlp evaluate \
    "${EXPMT_NAME}/models/model.tar.gz" \
    "data/az/az_papers/tag_bio/az_test_with_bio.jsonl" \
    --output-file "${EXPMT_NAME}/test_eval_metrics.jsonl" \
    --predictions-output "${EXPMT_NAME}/test_predictions.jsonl" \
    --cuda-device 0 \
    --batch-size 32 \
    --include-package classifier
