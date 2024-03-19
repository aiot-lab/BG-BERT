# Predicting Adverse Events for Patients with Type-1 Diabetes via Self-supervised Learning

> Xinzhe Zheng, Sijie Ji, Chenshu Wu 
>
> *In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2024*

This repository contains the code of BG-BERT.

The code has two parts, the first part is data conversion, and the second part is BG-BERT model training.

### 1. Data Conversion

You can find two Jupyter Notebook files in the folder 'data', you can run them to convert the raw data into the format that can be used by the BG-BERT model. The two datasets used in the paper are [OhioT1DM](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html) and [Diatrend](https://www.synapse.org/#!Synapse:syn38187184/wiki/619490). You can achieve an agreement with the data affiliations to get the data.

### 2. BG-BERT Model Training

The BG-BERT model training contains three parts:

1. Pre-training
2. Ouput Embedding
3. Prediction

**It should be noticed that all the hyperparameters, model path, and data path are defined in 'config.py' for convenience, as you do not have to type in anything in the command line.**

#### 2.1 Pre-training

To train the BG-BERT model, you have to write the correct option in the 'config.py' in source folder. The options are:

```
MODE = "pretrain" # pretrain, output_emb, prediction
OUTPUT_EMBED = False # Whether to output the embedding
AUGMENTATION = True # Whether to use data augmentation
```

Then you can run the 'main.py' in the source folder to train the BG-BERT model.

#### 2.2 Output Embedding

To output the embedding, you have to write the correct option in the 'config.py' in the source folder. The options are:

```
MODE = "output_emb" # pretrain, output_emb, prediction
OUTPUT_EMBED = True # Whether to output the embedding
AUGMENTATION = True # Whether to use data augmentation
```

Then you can run the 'main.py' in the source folder to output the embedding.

#### 2.3 Prediction

To predict the adverse events, you have to write the correct option in the 'config.py' in the source folder. The options are:

```
MODE = "prediction" # pretrain, output_emb, prediction
OUTPUT_EMBED = False # Whether to output the embedding
AUGMENTATION = True # Whether to use data augmentation
```

Then you can run the 'main.py' in the source folder to predict the adverse events.

Please refer to more details to 'config.py' file for training options and settings.

### 3. Cite This Work
If you use the code in this repository, please cite:
```
@inproceedings{zheng2024bgbert,
  title={Predicting Adverse Events for Patients with Type-1 Diabetes Via Self-Supervised Learning},
  author={Zheng, Xinzhe and Ji, Sijie and Wu, Chenshu},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1526-1530},
  year={2024},
  organization={IEEE},
  doi={10.1109/ICASSP48485.2024.10446832}
}
```
