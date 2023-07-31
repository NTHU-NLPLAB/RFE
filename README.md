# Introduction
This is the implementation of my masters thesis, Identification and Classification of Rhetorical Function Categories. This repository includes the code for the classifier and the sequence tagger, baseline models, the pre- and post-processing, and the analyses.

If you find the implementation helpful, please cite my thesis:

```bash
@mastersthesis{chen2023rfe,
  author  = "Li-Kuang Chen",
  title   = "Identification and Classification of Rhetorical Function Categories",
  school  = "National Tsing Hua University",
  year    = "2023"
}
```

# Setup
> The python version used is **3.8.10**, but python **3.9.6** works fine too.

We recommend creating a virtual environment for the experiments:

```bash
python3.8 -m venv .env
source .env/bin/activate
```

After you createrd the virtual environment, clone the project and install necessary packages:

```bash
git clone https://github.com/LKChenLK/RFE.git
cd RFE
pip install -r requirements.txt
```


# Data
The pre-processed data is available [here](https://drive.google.com/drive/folders/1RNO9vkdbmr8YBZvRzes3rA41frAXFOM_?usp=sharing). Please see [data](data) for further details.

# Experiments
We used AllenNLP to build the models and run experiments. To be able to use AllenNLP, you need to install `rfe` as a package. Run
```bash
pip install -e .
```
For details in reproducing the experiments in the paper, see [experiments](experiments).
