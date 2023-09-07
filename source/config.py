"""
config settings:
"""
from typing import NamedTuple

### Model Selection
MODE = "pretrain" # pretrain, output_emb, prediction
OUTPUT_EMBED = False # Whether to output the embedding
AUGMENTATION = True # Whether to use data augmentation
### ------------------ ###

### Training Hyperparameters
BATCH_SIZE = 1024 # Training batch size
SEED = 1 # Random seed
EPOCHS = 3200 # Training epochs
LEARNING_RATE = 0.0002 # Learning rate
SAVE_STEPS = 1000 # Save model every SAVE_STEP epochs
TOTAL_STEPS = 200000000 # Total training steps
PRED_PATIENCE = 20 # Patience for early stopping (Prediction mode)
### ------------------ ###

### Mask Settings
MASK_RATIO = 0.2 # Masking ratio
MASK_ALPHA = 6 # How much to fill the masked value
MASK_GRAM = 10 # Maximum number of tokens to be predicted in a sequence
MASK_PROB = 0.8 # Probability of replacing a masked token with a random token
REPLACE_PROB = 0.0 # Probability of replacing a masked token with a random token
### ------------------ ###

### Model Parameters
FEATURE_NUM = 5 # Number of input features
HIDDEN = 72 # Number of hidden units
HIDDEN_FF = 144 # Number of hidden units in feed forward layer
N_LAYER = 4 # Number of layers
N_HEADS = 4 # Number of heads
BACK_LENGTH = 24 # You can choose 24 or 48, which corresponds to 2 or 4 hours
FORE_LENGTH = 48 # You can choose 6 or 12, which corresponds to 30 or 60 minutes
SEQ_LEN = (BACK_LENGTH + FORE_LENGTH) # Sequence length
EMB_NORM = True # Whether to normalize the embedding
DROPOUT_RATE = 0.0 # Dropout rate
CLASS_NUM = 3 # BGBert classification number: 3 (normal, hyper, hypo)
### ------------------ ###

### Device for training
DEVICE = "cuda:0" # You can choose GPU or CPU
### ------------------ ###

### Training and testing setting
DATASET = "OhioT1DM" # Dataset name (OhioT1DM or Diatrend)
MODEL_TYPE = "BG-BERT"
DATA_DIR = '' # Dataset directory
OUT_DIR = '' # Output directory for both traning and testing
PREDICTION_DATA_DIR = OUT_DIR + '/prediction_data' # Prediction data directory
MODEL_PATH = '' # Pretrain Model path
PRED_MODEL_PATH = '' # Prediction Model path
### ------------------ ###

def load_config():
    kwargs = {}
    kwargs['mode'] = MODE
    kwargs['output_embed'] = OUTPUT_EMBED
    kwargs['augmentation'] = AUGMENTATION

    kwargs['batch_size'] = BATCH_SIZE
    kwargs['seed'] = SEED
    kwargs['epochs'] = EPOCHS
    kwargs['lr'] = LEARNING_RATE
    kwargs['save_steps'] = SAVE_STEPS
    kwargs['total_steps'] = TOTAL_STEPS
    kwargs['pred_patience'] = PRED_PATIENCE

    kwargs['mask_ratio'] = MASK_RATIO
    kwargs['mask_alpha'] = MASK_ALPHA
    kwargs['mask_gram'] = MASK_GRAM
    kwargs['mask_prob'] = MASK_PROB
    kwargs['replace_prob'] = REPLACE_PROB

    kwargs['feature_num'] = FEATURE_NUM
    kwargs['hidden'] = HIDDEN
    kwargs['hidden_ff'] = HIDDEN_FF
    kwargs['n_layers'] = N_LAYER
    kwargs['n_heads'] = N_HEADS
    kwargs['back_length'] = BACK_LENGTH
    kwargs['fore_length'] = FORE_LENGTH
    kwargs['seq_len'] = SEQ_LEN
    kwargs['emb_norm'] = EMB_NORM
    kwargs['dropout'] = DROPOUT_RATE
    kwargs['class_num'] = CLASS_NUM

    kwargs['device'] = DEVICE

    kwargs['dataset'] = DATASET
    kwargs['model_type'] = MODEL_TYPE
    kwargs['data_dir'] = DATA_DIR
    kwargs['out_dir'] = OUT_DIR
    kwargs['prediction_data_dir'] = PREDICTION_DATA_DIR
    kwargs['model_path'] = MODEL_PATH
    kwargs['pred_model_path'] = PRED_MODEL_PATH

    return kwargs
