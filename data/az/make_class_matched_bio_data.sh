
# split=...
# base dir: data/az/az_papers

#python ../../teufel_patterns/combine_patterns_lexicon.py . all_patterns_with_lexicon.json

#python ../../teufel_patterns/make_all_patterns_by_label.py all_patterns_with_lexicon.json all_patterns_with_lexicon_by_label.json

target_dir="./az_papers"
out_data_dir="${target_dr}/tag_bio_filt_len_clb_14062023"

#out_data_dir="./tmp_dir"
mkdir -p "${out_data_dir}"
mkdir -p "${out_data_dir}_strict"

splits=("train" "test" "dev")
#splits=("test" "dev")


for split in "${splits[@]}" 
do
#    python ../../teufel_patterns/retag_teufel_patterns.py \
#        --data_path "clf_only/parag/${split}.jsonl" \
#        --out_file "${out_data_dir}/az_${split}_matches.jsonl" \
#    	--cached_parse "clf_only/parag/${split}.pickle" \
#	--filter_length

    python ../../match_idx_to_class_lab_bio.py \
        --in_file "${target_dir}/tag_bio_filt_len_14062023/az_${split}_matches.jsonl" \
        --out_file "${out_data_dir}/${split}.jsonl" \
        --check_overlap

    python ../../match_idx_to_class_lab_bio.py \
	--in_file "${target_dir}/tag_bio_filt_len_14062023/az_${split}_matches.jsonl" \
	--out_file "${out_data_dir}_strict/${split}.jsonl" \
	--check_overlap \
	--strict
done

python ../../resample_imbalanced.py --in_file "${out_data_dir}/train.jsonl" --out_file "${out_data_dir}/train_resampled.jsonl"

python ../../resample_imbalanced.py --in_file "${out_data_dir}_strict/train.jsonl" --out_file "${out_data_dir}_strict/train_resampled.jsonl"
