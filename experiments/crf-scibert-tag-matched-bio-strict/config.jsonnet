# joker:/home/nlplab/lkchen/move_classifier/experiments/az-seq-tag-crf/config.jsonnet
local bert_model = "allenai/scibert_scivocab_uncased";
local extra_tokens = ['[ IMAGE ]', 'CREF','EQN', 'CITATION']; # padding token is @@PADDING@@ by default
local transformer_hidden_size = 768;
local transformer_max_length = 512;
local tag_embedding_size = 128;

{
    "dataset_reader" : {
        "type": "example_reader",
        "max_parag_length": 13,
        "dataset": 'az',
        "do_seq_labelling": true,
	"filter_length": true,
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "max_length": 128,
            "tokenizer_kwargs": {
                "additional_special_tokens": extra_tokens,
                'return_offsets_mapping': true
                }
            
        },
        "text_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
		        #"max_length": transformer_max_length
                "tokenizer_kwargs": {
                    "additional_special_tokens": extra_tokens,
                    'return_offsets_mapping': true 
                    }
            },
        },
    },
    "train_data_path": "../../data/az/az_papers/tag_bio_filt_len_clb_14062023_strict/train_resampled.jsonl",
    "validation_data_path": "../../data/az/az_papers/tag_bio_filt_len_14062023/dev.jsonl",
    "model": {
        "type": "crf_tagger",
        "label_encoding": "BIO",
        //"dataset": 'az',
        "ignore_loss_on_o_tags": false,
        "constrain_crf_decoding": true,
        "calculate_span_f1": true,
        "dropout": 0.5,
        "include_start_end_transitions": false,
        "verbose_metrics": true,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model,
                    //"train_parameters": true
                    "train_parameters": false,
                    "tokenizer_kwargs": { 
                        "additional_special_tokens": extra_tokens, 
                        'return_offsets_mapping': true,
                    }
                }
            }
        },
        "seq2seq_encoder": {
            "type": "lstm", // params: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
             //"input_size": 50 + 128, // max seq len of text encoder
            "input_size": transformer_hidden_size, //+ tag_embedding_size, // the concatenated dim size
            "hidden_size": 300,
            "num_layers": 2,
            "dropout": 0.5,
            "bidirectional": true
            // lstm outputs seq_len x batch x directions * hidden_size
            // binder's code didn't transpose this before sending it into seq2vec_encoder...
            // I'll have to assume AllenNLP does the transposing for us
        },
    },
    "data_loader": {
        "batch_size": 128,
        "shuffle": true,
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.005,
        },
        "validation_metric": "+span/overall/f1",
        "num_epochs": 45,
        "grad_norm": 7.0,
        "patience": 20,
        "cuda_device": 1,
    }
}
