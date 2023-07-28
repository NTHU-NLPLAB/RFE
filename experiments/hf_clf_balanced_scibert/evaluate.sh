

python ../../hf_inference.py \
    --data_is_split_into_words \
    --dataset az \
    --tokenizer "allenai/scibert_scivocab_uncased" \
    --model "./models/checkpoint-3000" \
    --data_dir ../../data/az/az_papers/tag_bio \
    --out_dir models \
    --epochs 6 \
    --lr 2e-5 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --shuffle_train \
    --early_stop_patience 3
