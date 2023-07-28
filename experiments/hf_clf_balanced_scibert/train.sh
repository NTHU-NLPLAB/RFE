

python ../../hf_finetune.py \
    --data_is_split_into_words \
    --dataset az \
    --data_dir ../../data/az/az_papers/tag_bio \
    --out_dir models \
    --model allenai/scibert_scivocab_uncased \
    --epochs 6 \
    --lr 2e-5 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --shuffle_train \
    --early_stop_patience 3 \
    --do_test \
    --oversample
