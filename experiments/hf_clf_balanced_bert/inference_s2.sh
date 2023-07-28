

python ../../hf_inference.py \
    --dataset az \
    --model bert-base-cased \
    --checkpoint "./models/checkpoint-3000" \
    --data_dir ../../data/az+s2/bio \
    --out_dir models \
    --eval_batch_size 64
