# -*- coding: utf-8 -*-
"""
Created on Tues Feb 14th 2023
@author: Anurag Kumar
"""
import os
import time
import torch
from tqdm import tqdm
import argparse
from pathlib import Path
from tqdm import tqdm

from model import JointPostProcess
from dataset import Post_Processing_Dataset
from utils import PuncCaseJointLoss, FocalLoss, freeze_layers, collate, score

import wandb

from torch.utils.data import DataLoader
from fairseq.models.roberta import RobertaModel

import gc



def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True,
                        help="Root directory containing tsvs.")
    parser.add_argument("--exp", type=str, required=False, default='default', help="Experiment name.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for checkpoints. Will create one if doesn't exist")
    parser.add_argument("-pt", "--ckpt", type=str, required=False,
                        help="Path to the pretrained roberta model.")
    parser.add_argument("--epochs", type=int, required=False, default=5,
                        help="No. of epochs to be trained.")
    parser.add_argument("--lr", type=float, required=False, default=0.0001,
                        help="Training learning rate.")
    parser.add_argument("--alpha", type=float, required=False, default=0.1,
                        help="Weight to be given to joint punc+case loss.")
    parser.add_argument("--batchsize", type=int, required=False, default=4,
                        help="Training batchsize.")
    parser.add_argument("--embedding_dim", type=int, required=False, default=1024,
                        help="Speaker embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, required=False, default=128,
                        help="Speaker embedding dimension.")
    parser.add_argument("--num_layers", type=int, required=False, default=3,
                        help="Speaker embedding dimension.")
    parser.add_argument("--max_len", type=int, required=False, default=3,
                        help="Speaker embedding dimension.")
    parser.add_argument("--jump", type=int, required=False, default=3,
                        help="Speaker embedding dimension.")
    parser.add_argument("--print_freq", type=int, required=False, default=3,
                        help="Logging frequency.")
    parser.add_argument("--accum_grad", type=int, required=False, default=1,
                        help="Accumulated gradient for these many steps.")
    parser.add_argument("--punctuator", type=str, required=False,
                        help="Path to the punctuator.")
    parser.add_argument("--gpu", action='store_true',
                        help="Set this flag for gpu training.")
    parser.add_argument("--pt", type=str, required=False, default=None,
                        help="Continue training if path to saved checkpoint is provided.")
    parser.add_argument("--mode", type=str, required=False, default='PP+QAC',
                        help="Option (PP+QAC/PP)")
    parser.add_argument("--reset", action='store_true', required=False,
                        help="Reset training with new optimizer and scheduler with a saved checkpoint.")
    parser.add_argument("--smooth", action='store_true', required=False,
                        help="Smooth speaker switch labels.")
    return parser