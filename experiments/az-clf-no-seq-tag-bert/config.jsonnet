local bert_model = "bert-base-uncased";
local extra_tokens = ['[ IMAGE ]', 'CREF','EQN', 'CITATION']; # padding token is @@PADDING@@ by default
local transformer_hidden_size = 768;
local transformer_max_length = 512;
local tag_embedding_size = 128;

{
    "dataset_reader" : {
        "type": "example_reader",
        "max_parag_length": 13,
        "dataset": 'az',
	"do_seq_labelling": false,
	"attach_header": true,
	"filter_length": true,
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "max_length": 128,
            "tokenizer_kwargs": { "additional_special_tokens": extra_tokens, }
            
        },
        "text_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
                "tokenizer_kwargs": { "additional_special_tokens": extra_tokens, 'return_offsets_mapping': true }
            },
        },
    },
    "train_data_path": "../../data/az/az_papers/tag_bio_filt_len_14062023/train_resampled.jsonl",
    "validation_data_path": "../../data/az/az_papers/tag_bio_filt_len_14062023/dev.jsonl",
    "model": {
        "type": "move_tag_classifier",
        "do_seq_labelling": false,
        "checkpoint": "",
        "dataset": 'az',
	"label_smoothing": 0.3,
        "init_clf_weights": "random",
        "final_dropout": 0.3,
        "joint_loss_lambda": 0.2,
        "verbose_metrics": true,
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model,
                    "train_parameters": false,
                    "tokenizer_kwargs": { 
                        "additional_special_tokens": extra_tokens, 
                        'return_offsets_mapping': true,
                    }
                }
            }
        },
        "tag_embedding_size": tag_embedding_size,
        "seq2seq_encoder": {
            "type": "lstm", // params: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
             //"input_size": 50 + 128, // max seq len of text encoder
            "input_size": transformer_hidden_size, // + tag_embedding_size, // the concatenated dim size
            "hidden_size": 430,
            "num_layers": 2,
            "dropout": 0.25,
            "bidirectional": true,
            // lstm outputs seq_len x batch x directions * hidden_size
            // binder's code didn't transpose this before sending it into seq2vec_encoder...
            // I'll have to assumer AllenNLP does the transposing for us
        },
        "pooler": {  // https://docs.allennlp.org/main/api/modules/seq2vec_encoders/cnn_encoder/
            "type": "cnn", // input shape (batch_size, num_tokens, input_dim)
            # "type": bert_model,"pretrained_model": bert_model
            "embedding_dim": 860, // from lstm output size, since the lstm is bidirectional
	        "num_filters": 193, // the output dim for each convolutional layer,
	        "ngram_filter_sizes": [3, 5, 7, 10], // 4 conv layers, each with fileter ngram size 3/5/7/10
            // if output_dim is None, the result of the max pooling will be returned, 
            // which has a shape len(ngram_filter_sizes) * num_filters = 4 * 193
        },
    },
    "data_loader": {
        "batch_size": 64,
        "shuffle": true,
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0002, // binder used 0.005 for adu_best and 0.0005 for rel_best
        },
        "num_epochs": 25,
        "cuda_device": 0,
        "grad_norm": 4,
        "patience": 10,
        "learning_rate_scheduler": { // TODO: try reduce_on_plateau (read docs for configs)
            "type": "reduce_on_plateau",
            "mode": "max",
            "factor": 0.5,
            "patience": 5,
            "threshold": 1e-4,
            "threshold_mode": "rel",
        },
    }
}
