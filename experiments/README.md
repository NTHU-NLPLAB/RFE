

# Training and Evaluation
Each subdirectory contrains training and evaluation scripts for a an experiment, such as `train.sh`, `evaluate.sh` etc. 
## To train the models from scratch:
```bash
cd <experiment name>
source train.sh
```
The model weights will be stored in `experiments/<experiment_name>/models`. 

## To evaluate the models on the test split, run
```bash
# cd <experiment name> if you are not in the experiment directory
source evaluate.sh
```

## Evaluate the checkpoints
We have uploaded checkpoints for our best classifier and CRF-tagger. (Please contact us if you need checkpoints for other models.)
Download the [checkpoints](https://drive.google.com/drive/folders/1RNO9vkdbmr8YBZvRzes3rA41frAXFOM_?usp=sharing) and place the corresponding checkpoint (`model.tar.gz`) under `experiments/<experiment_name>/models`. Then, simply evaluate with 
```bash
source evaluate.sh
```


## Experiments
Classification experiments are folders with names starting with `az-clf-*`. Models that do NOT multi-task (i.e., those do NOT use RFE tagging as a auxiliary task) are denoted by `az-clf-no-seq-tag-*`. Folders starting with `hf-*` contain scripts for our baseline models, which are off-the-shelf models from HuggingFace, fine-tuned with the same data as other models.

RFE-tagging experiments are stored in `crf-*` folders.
`crf-*-seq-balanced` are models trained with a dataset that balances the RFE-tag types instead of the class labels. We find the latter to produce better RFE-tagging models, but keep the experiments for those who wish to investigate.
`crf-no-sep` and `crf-scibert-pad-ht` are extra experiments masking the `[CLS]` and `[SEP]` tokens; they did not make it into my thesis. 


# Prediction
If you would like to do prediction on unlabelled data, modify and run the below script.
The prediction script requires a `.jsonl` file with each entry formatted as a dictionary. Each dictionary is required to contain `"text"` and as key the input sentence as value. 

Example:
```python
{"text": "This is an example test sentence."}
```

```bash
allennlp predict \
    experiments/<your_model_name>/model.tar.gz \
    path/to/your_test_data.jsonl \
    --output-file path/to/your_test_data.preds.jsonl \
    --include-package rfe \
    --predictor crf_predictor \
    --batch-size 1024 \
    --cuda 1 \
    --silent
```
> Note: You can change the argument [--cuda 1] to [--cuda -1] if you don't want to use a GPU. For more documentation on the options, please refer to the [source code of AllenNLP 2.9.3](https://github.com/allenai/allennlp/blob/v2.9.3/allennlp/commands/predict.py)

-------

## Predicting on a single sentence
Also, we can use the model in programs to estimate the probability of being a good sentence.
```python
from rfe.predictor import CrfPredictor

crf = CrfPredictor.from_path('experiments/<experiment_name>/models/model.tar.gz', 'crf_predictor')

example_sent = 'In this paper, we first provide an overview of the system of accreditation and then discuss issues of accreditation as they apply to these contemporary American educational programs in Japan.'

preds = crf.predict_tags({'text': example_sent})

example_sent_preds = crf.predict_tags({'text': example_sent})
print(f'predicted tags: {example_sent_preds["pred_tags"]}')
>> ['O', 'O', 'O', 'O', 'O', 'B-OWN', 'I-OWN', 'I-OWN', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

```

# Post-processing model predictions
Because the model uses a subword tokenizer, which splits input words into subwords, the predicted BIO tags correspond to subwords, not words. We de-subword the subwords and align their corresponding BIO tags in this step wiht `../data/postprocess_span.py`.

`../data/postprocess_span.py` requires several external files:
- **`in_file`** is the model output. It should be a `.jsonl` file, each line a python dictionary with `paragraph`, `pred_tags` (predicted tags), `seq_tags` (gold tags), and optionally, `top_k_tags`, as keys.
- **`idx_to_tag_file`** is the file that maps the BIO tags used during training to integer indices. You can find this under `experiments/<experiment_name>/models/vocab` as `move_tags.txt`.
- **`idx_to_label_file`** is the file that maps class labels used during training to integer indices. This isn't generated if you train a CRF tagger and can safely ignore this option. If you use a multi-tasking model, you can find this under `experiments/<experiment_name>/models/vocab` as `move_tags.txt`.
- **`configs`** is the configuration file used for the experiments. This is used to retrieve additional tokens defined within. If you do not use this, special tokens such as `[ IMAGE ]`, `CREF`, and `CITATION`, `EQN` will not be decoded. You can find this under `experiments/<experiment_name>/` as `configs.jsonnet`.

Other information for  `../data/postprocess_span.py`:
- It outputs `<model output filename>_postproc.jsonl` in **`out_dir`**, the output directory. For example, if the model output name is `dev_predictions.jsonl`, the output file would be `out_dir/dev_predictions_postproc.jsonl`.
- **`lm`** is the tokenizer to decode the subword indices back into text. You should use the same tokenizer used during training. The defualt is `allenai/scibert_scivocab_uncased`.
- **`force_non_o`** is an experimental feature that retrieves the 2nd-best Viterbi score for the CRF model predictions. This is to see if the 2nd-best predictions of all-O predictions are of any use. To be able to use this, add the `--overrides '{"model.top_k": 2}' \` option to `allennlp evaluate` or `allennlp predict` to generate more than just 1 prediction for each sequence.
- **`batched`** is required if the **inputs** are batched. This is the case if you specify `--batch-size` at inference. 

