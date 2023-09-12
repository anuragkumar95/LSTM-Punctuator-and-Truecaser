## Introduction

This repo uses RoBERTa to tokenize and provide token embeddings which are then used by a bidirectional LSTM model to jointly predict the punctuation (COMMA, PERIOD, QUESTION, EXCLAIMATION) and casing (Uppercase, Lowercase, Cardinal) of the tokens. NOTE: At this point Mixed casing is not predicted by this model. 

It will require pytorch and fairseq to be installed. Would also require the roberta model to be downloaded and stored at ```path/to/roberta```

### Data
The code is made to consume .txt files. Store the training and validation subset of text files at ```/path/to/root```. The root directory should have train and val subdirs. The code should be able to automatically read and create the respective datasets.

### Training

For training, simply run

```python train.py --exp name_of_experiment --ckpt /path/to/roberta --epochs 50 --lr 0.001 --batchsize 128 --embedding_dim roberta_emb_dim --hidden_dim hidden_dim_lstm --num_layers num_layers_LSTM --max_len max_input_seq_len --jump overlap_len_over_inp_seq --gpu --r /path/to/root --o path/to/store/checkpoints```

