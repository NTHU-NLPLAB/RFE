# differences between `allennlp predict` and `allennlp evaluate``:
# https://github.com/allenai/allennlp/issues/2576#issuecomment-471144577
# tldr: `predict` outputs a json file, more for demo settings, no metrics
# `evaluate`` calculates metrics

EXPMT_NAME="experiments/az-simple-seq-tag-no-crf"

allennlp evaluate \
    "${EXPMT_NAME}/models/model.tar.gz" \
    "data/az/az_papers/tag_bio/az_test_with_bio.jsonl" \
    --output-file "${EXPMT_NAME}/test_eval_metrics.jsonl" \
    --predictions-output "${EXPMT_NAME}/test_predictions.jsonl" \
    --cuda-device 0 \
    --batch-size 32 \
    --include-package classifier
