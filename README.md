# Predicting Adverse Events for Patients with Type-1 Diabetes via Self-supervised Learning

> Xinzhe Zheng, Sijie Ji, Chenshu Wu 
>
> *In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2024*

This repository contains the code of BG-BERT.

The code has two parts, the first part is data conversion, and the second part is BG-BERT model training.

### 1. Data Conversion

You can find two Jupeter Notebook files in the folder 'data', you can run them to convert the raw data into the format that can be used by the BG-BERT model. The two datasets used in the paper are [OhioT1DM](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html) and [Diatrend](https://www.synapse.org/#!Synapse:syn38187184/wiki/619490). You can achieve an agreement with the data affiliations to get the data.

### 2. BG-BERT Model Training

The BG-BERT model training contains three parts:

1. Pre-training
2. Ouput Embedding
3. Prediction

**It should be noticed that all the hyperparameters, model path, data path are defined in 'config.py' for convenience, as you donnot have to type in anything in the command line.**

#### 2.1 Pre-training

To train the BG-BERT model, you have to write the correct option in the 'config.py' in source folder. The options are:

```
MODE = "pretrain" # pretrain, output_emb, prediction
OUTPUT_EMBED = False # Whether to output the embedding
AUGMENTATION = True # Whether to use data augmentation
```

Then you can run the 'main.py' in source folder to train the BG-BERT model.

#### 2.2 Output Embedding

To output the embedding, you have to write the correct option in the 'config.py' in source folder. The options are:

```
MODE = "output_emb" # pretrain, output_emb, prediction
OUTPUT_EMBED = True # Whether to output the embedding
AUGMENTATION = True # Whether to use data augmentation
```

Then you can run the 'main.py' in source folder to output the embedding.

#### 2.3 Prediction

To predict the adverse events, you have to write the correct option in the 'config.py' in source folder. The options are:

```
MODE = "prediction" # pretrain, output_emb, prediction
OUTPUT_EMBED = False # Whether to output the embedding
AUGMENTATION = True # Whether to use data augmentation
```

Then you can run the 'main.py' in source folder to predict the adverse events.

Please refer more details to 'config.py' file for training options and settings.
