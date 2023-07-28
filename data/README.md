
# Downloading Pre-Processed Data
You can skip data creation by downloading the preprocessed data [here](https://drive.google.com/drive/folders/1RNO9vkdbmr8YBZvRzes3rA41frAXFOM_?usp=sharing). After downloading `az_papers.tar.gz`, extract it and place the folders in `data/az/` as `az_papers`. The `data` directory should look like this:

```
├── data 
│   ├── az
│   │   ├── az_papers
│   │   │    │── clf_only/
│   │   │    │── tag_bio_filt_len/
│   │   │    │── tag_bio_filt_len_clb/
│   │   │    │── tag_bio_filt_len_clb_strict/
│   │   │    └──all_patterns_with_lexicon_by_label.json
│   │   ├── <...>.py
│   │   └── <...>.sh
│   │   
│   └── teufel_patterns
├── <...>.py
├── README.md
└── .gitignore
```
If you have successfully downloaded the data and placed it in the right place, you should be able to commence training / evaluation. Please go to [experiments](../experiments) for details.

# Pre-Processing the Data Yourself 

## Download the Original AZ Corpus
Alternatively, if you with to process the original data, you can start with downloading the original AZ Corpus [here](https://www.cl.cam.ac.uk/~sht25/AZ_corpus.html) (as of 25 Jun 2023). Note that the file `9502039.az-scixml` has syntax issues and thus should not be parsed or included in the preprocessing script.

Download the original AZ Corpus and extract the `.scixml` files under `az/az_papers/raw`. 

## Install the Data Processing Scripts as a Package
If you are able to do relative imports without problems, you can skip this step.
I installed the relevant scripts as a package because I encountered problems with relative imports. I put the methods I need in `setup.py` and ran
```bash
pip install -e .
```
Try doing the same if you encounter problems with relative immports too.

## Pre-process and Split the AZ Corpus
Next, we pre-process and split the data:
```bash
cd az
source make_az.sh
```
This creates `az/az_papers_all.jsonl`, `az/az_papers_abstract.jsonl`, `az/az_papers_body.jsonl`, and `az/az_papers.jsonl`. The pre-processed data is split and stored in `./az_papers/clf_only/parag`. 
Since the original AZ corpus comes in paragraphs, we need to first break them down into sentences. This is done by the third line in `make_az.sh`, and the output is stored in `./az_papers/clf_only/sents`. 

## Make Patterns and BIO-tag them
After we have split the data into `{train,dev,test}.jsonl` and into sentences, we may match for pre-defined patterns as RFEs and turn them into BIO tags:
```bash
source make_az_bio_data.sh
```
This may take 10 to 20 minutes (or longer), depending on the machine you are using. 
After the process finishes, we should get `{train,train_resampled,dev,test}.jsonl` and `az_{train,dev,test}_matches.jsonl` in `az_papers/tag_bio_filt_len_14062023`. We will only need the former set of `.jsonl` files.

In particular, `train_resampled.jsonl` is simply `train.jsonl` resampled such that all classes have the same number of examples as the majority class, `OWN`.

`az_papers/tag_bio_filt_len_14062023` will be the data directory used in the experiment config files. You may change the directory if you wish, but remember to change it inthe experiment config file afterwards too.


To balance the data by tag types (instead of class labels), run
```bash
source make_balanced_seq_data.sh
```

To make class-matched data or class-matched (strict) data (and their tag-type-balanced data), run
```bash
source make_class_matched_bio_data.sh
source make_balanced_seq_clb_data.sh
```
The above scripts will create `{train,train_resampled,dev,test}.jsonl` and `az_{train,dev,test}_matches.jsonl` under `az_papers/tag_bio_filt_len_clb_14062023` and `az_papers/tag_bio_filt_len_clb_14062023_strict`
