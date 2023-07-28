#!/bin/zsh

EXPMT_DIR="."

PYTHONUNBUFFERED=1  CUDA_LAUNCH_BLOCKING=1 \
allennlp train \
	"${EXPMT_DIR}"/config.jsonnet \
	-s "${EXPMT_DIR}"/models \
	-f --include-package classifier
