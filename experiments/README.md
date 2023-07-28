

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