
# split=...
# base dir: data/az/az_papers

python ../../teufel_patterns/combine_patterns_lexicon.py . all_patterns_with_lexicon.json

python ../../teufel_patterns/make_all_patterns_by_label.py all_patterns_with_lexicon.json all_patterns_with_lexicon_by_label.json

target_dir="./az_papers"
out_data_dir="${target_dir}/tag_bio_filt_len_14062023"

splits=("train" "test" "dev")

for split in "${splits[@]}" 
do
    python ../../teufel_patterns/retag_teufel_patterns.py \
        --data_path "${target_dir}/clf_only/parag/${split}.jsonl" \
	--pattern_path "${target_dir}/all_patterns_with_lexicon_by_label.json" \
        --out_file "${out_data_dir}/az_${split}_matches.jsonl" \
    	--cached_parse "${target_dir}/clf_only/parag/${split}.pickle" \
	--filter_length

    python ../../match_idx_to_bio.py \
        --in_file "${out_data_dir}/az_${split}_matches.jsonl" \
        --out_file "${out_data_dir}/${split}.jsonl" \
        --check_overlap
done

python ../../resample_imbalanced.py --in_file "${out_data_dir}/train.jsonl" --out_file "${out_data_dir}/train_resampled.jsonl"
