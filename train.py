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
from utils import PuncCaseJointLoss, freeze_layers, collate, score

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
                        help="Roberta embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, required=False, default=128,
                        help="hidden dimension for rnn layers.")
    parser.add_argument("--num_layers", type=int, required=False, default=3,
                        help="Number of layers for RNN.")
    parser.add_argument("--max_len", type=int, required=False, default=3,
                        help="Maximum input sequence length. Sequences longer than this are truncated.")
    parser.add_argument("--jump", type=int, required=False, default=3,
                        help="How much to overlap between sequences.")
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
    parser.add_argument("--reset", action='store_true', required=False,
                        help="Reset training with new optimizer and scheduler with a saved checkpoint.")
    return parser


def train_one_epoch(model, epoch, train_dl, val_dl, optimizer, criterion, accum_grad=1, scheduler=None, device=None):
    """
    Function defining logic for each epoch
    ARGS:
        model     : instance to the loaded model.
        epoch     : epoch number
        train_dl  : train dataloader.
        val_dl    : validation dataloader.
        optimizer : instance of the optimizer.
        criterion : instance of the loss function.
        accum_grad: accumulate gradients before backprop
        scheduler : learning rate scheduler
        device    : gpu device if present. By default device is cpu.

    Returns
        Epoch averaged train and validation loss.
    """
    gc.collect()
    tr_loss = 0
    val_loss = 0
    train_batches = len(train_dl)
    val_batches = len(val_dl)
    pre, re, f1 = {}, {}, {}
    torch.cuda.empty_cache()

    #Train stage
    model.train()
    batch_loss = 0
    batch_punc_loss = 0
    batch_case_loss = 0
    val_punc_loss = 0
    val_case_loss = 0
    
    with tqdm(train_dl, unit='batch') as train_epoch:
        for i, (inputs, puncts, cases, lens, _, _, _) in enumerate(train_epoch):
            train_epoch.set_description(f"Epoch {epoch}")
            if device:
                inputs = inputs.to(device)
                puncts = puncts.to(device)
                cases = cases.to(device)
            
            #Filter for bad batches that may stop training
            if torch.isnan(inputs.sum()) or torch.isinf(inputs.sum()):
                continue
            if torch.isnan(puncts.sum()) or torch.isinf(puncts.sum()):
                continue
            if torch.isnan(cases.sum()) or torch.isinf(cases.sum()):
                continue
        
            punc_logits, case_logits = model(inputs, 
                                                lens, 
                                                batch_first=True)
            puncts = puncts.reshape(-1)
            cases = cases.reshape(-1)
            punc_logits = punc_logits.reshape(-1, punc_logits.shape[-1]).add(1e-08)
            case_logits = case_logits.reshape(-1, case_logits.shape[-1]).add(1e-08)
            
            punc_loss, case_loss = criterion(punc_logits,
                                             case_logits,
                                             puncts,
                                             cases)
            
            step_loss = punc_loss + case_loss
            if (i+1) % accum_grad == 0:
                batch_loss += step_loss
                batch_punc_loss += punc_loss
                batch_case_loss += case_loss
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                tr_loss += batch_loss
                wandb.log({"train_punc_loss":batch_punc_loss,
                            "train_case_loss":batch_case_loss,
                            "step":(i+1)/accum_grad + ((len(train_dl)/accum_grad)*epoch)}
                            )
                batch_loss = 0
                batch_punc_loss = 0
                batch_case_loss = 0
            else:
                batch_loss += step_loss
                batch_punc_loss += punc_loss
                batch_case_loss += case_loss

            try:
                train_epoch.set_postfix(loss=tr_loss.item()/(i+1))
            except:
                train_epoch.set_postfix(loss=tr_loss/(i+1))
            time.sleep(0.1)
    torch.cuda.empty_cache()
   
    #Validation stage
    model.eval()
    preds = {'punc':[], 'case':[], 'qac':[]}
    targets = {'punc':[], 'case':[], 'qac':[]}
    with torch.no_grad():
        with tqdm(val_dl, unit='batch') as val_epoch:
            for i, (inputs, puncts, cases, lens, _, _, _) in enumerate(val_epoch):
                val_epoch.set_description(f"Val Epoch:{epoch}")
                if device:
                    inputs = inputs.to(device)
                    puncts = puncts.to(device)
                    cases = cases.to(device)

                punc_logits, case_logits = model(inputs, 
                                                    lens, 
                                                    batch_first=True)
                puncts = puncts.reshape(-1)
                cases = cases.reshape(-1)

                punc_logits = punc_logits.reshape(-1, punc_logits.shape[-1])
                case_logits = case_logits.reshape(-1, case_logits.shape[-1])

                punc_loss, case_loss = criterion(punc_logits,
                                                    case_logits,
                                                    puncts,
                                                    cases)
                
                step_loss = punc_loss + case_loss
                val_punc_loss += punc_loss
                val_case_loss += case_loss
                val_loss += step_loss
                
                p_preds = torch.argmax(punc_logits, dim=1).detach().cpu().numpy().tolist()
                c_preds = torch.argmax(case_logits, dim=1).detach().cpu().numpy().tolist()
                
                preds['punc'].extend(p_preds)
                preds['case'].extend(c_preds)
                
                targets['punc'].extend(puncts.detach().cpu().numpy().tolist())
                targets['case'].extend(cases.detach().cpu().numpy().tolist())
                
                try:
                    val_epoch.set_postfix(loss=val_loss.item()/(i+1))
                except:
                    val_epoch.set_postfix(loss=val_loss/(i+1))
                time.sleep(0.1)

    wandb.log({"val_punc_loss":val_punc_loss/len(val_dl), 
                "val_case_loss":val_case_loss/len(val_dl), 
                "epoch":epoch}
                )        

    torch.cuda.empty_cache()
   
    pre['punc'], re['punc'], f1['punc'], _  = score(targets['punc'], preds['punc'], method='weighted')    
    pre['case'], re['case'], f1['case'], _  = score(targets['case'], preds['case'], method='weighted')

    gc.collect()  

    return tr_loss/train_batches, val_loss/val_batches, pre, re, f1


def train(model, 
          train_dl, 
          val_dl, 
          criterion, 
          optimizer, 
          n_epochs, 
          print_freq, 
          save_dir,
          accum_grad=1, 
          scheduler=None,
          start_epoch=0, 
          experiment='default', 
          device=None):
    """
    Wrapper function to run train for all epochs. Prints loss and saves checkpoints.
    ARGS:
        model      : instance to the loaded model.
        train_dl   : train dataloader.
        val_dl     : validation dataloader.
        optimizer  : instance of the optimizer.
        criterion  : instance of the loss function.
        n_epochs   : number of epochs to train.
        print_freq : logging/printing frequency for progress. Will log loss every print_freq epoch.
        save_dir   : path to dir to store checkpoints. Assumes the dir to be created.
        experiment : name of the experiment. 
        device     : gpu device if present. By default device is cpu.
    """
    
    total_tr_loss = 0
    prev_best_val_loss = float('inf')

    if device:
        model = model.to(device)
    
    gc.collect()
    
    for epoch in range(start_epoch, n_epochs+1):
        tr_loss, val_loss, _, _, f1 = train_one_epoch(model,
                                                      epoch,
                                                      train_dl,
                                                      val_dl,
                                                      optimizer,
                                                      criterion,
                                                      accum_grad,
                                                      scheduler,
                                                      device)
        
        wandb.log({"val_loss":val_loss, 
                   "train_loss":tr_loss,
                   "val_punc_F1":f1['punc'],
                   "val_case_F1":f1['case'],
                   "LR":scheduler.get_last_lr(), 
                   "epoch":epoch}
                 )  

        total_tr_loss += tr_loss
        if (epoch) % print_freq == 0:
            msg = f"Epoch:{epoch}, Train_Loss :{total_tr_loss/(epoch-start_epoch+1)}, Val_Loss :{val_loss} Val_Punc_F1 :{f1['punc']:.5f}, Val_Case_F1 :{f1['case']:.5f}, LR:{scheduler.get_last_lr()}"
            print(msg)
            with open(os.path.join(save_dir, f'{experiment}.log'), 'a') as f:
                f.write(f"{msg}\n")

        if val_loss <= prev_best_val_loss:
            prev_best_val_loss = val_loss
            save_dict = {'model_state_dict':model.state_dict(), 
                         'optim_state_dict':optimizer.state_dict(),
                         'scheduler_state_dict':scheduler.state_dict(),
                         'epoch':epoch,
                         'F1':f1,
                         'loss':val_loss,
                         'lr':scheduler.get_last_lr()
                        }
            torch.save(save_dict, f"{save_dir}/{epoch+1}_{experiment}_{val_loss:.5f}.pt")
            print(f"Found new best val_loss : {val_loss} Val_Punc_F1 :{f1['punc']:.5f}  Val_Case_F1 :{f1['case']:.5f} LR:{scheduler.get_last_lr()} checkpoint saved")
            
        torch.cuda.empty_cache()
        gc.collect()


def main(ARGS):
    gc.collect()
    device = torch.device('cpu')
    if ARGS.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

    CASE = {'LC':0,
            'UC':1,
            'CA':2}
    
    PUNC = {',':0,
            '.':1,
            '?':2,
            '!':3,
            'NA':4}
    
    roberta = RobertaModel.from_pretrained(ARGS.ckpt, checkpoint_file='model.pt')

    wandb.init(project="Joint punct-casing",
            # track hyperparameters and run metadata
            config= {
                     "learning_rate": ARGS.lr,
                     "epochs": ARGS.epochs,
                    }
    )
    train_root = f"{ARGS.root}/train"
    train_dataset = Post_Processing_Dataset(root=train_root,
                                            tokenizer=roberta,
                                            case_labels=CASE,
                                            punc_labels=PUNC)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=ARGS.batchsize,
                                  shuffle=True,
                                  collate_fn=collate)

    valid_root = f"{ARGS.root}/valid"
    valid_dataset = Post_Processing_Dataset(root=valid_root,
                                            tokenizer=roberta,
                                            case_labels=CASE,
                                            punc_labels=PUNC) 

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=ARGS.batchsize,
                                  shuffle=True,
                                  collate_fn=collate)
    
    model = JointPostProcess(encoder=roberta, 
                             embedding_dim=ARGS.embedding_dim, 
                             hidden_dim=ARGS.hidden_dim, 
                             num_layers=ARGS.num_layers,
                             punct_classes=len(PUNC),
                             case_classes=len(CASE),
                             dropout_prob=0.05, 
                             batch_first=True,
                             bi_directional=True,
                             device=device)
    
    #Freeze embedding layer of the model
    model = freeze_layers(model, ['encoder'])
    criterion = PuncCaseJointLoss(device=device, loss='CE')
    #criterion = PuncCaseJointLoss(device=device, loss='Focal')
    
    #Set optimizer and lr scheduler
    optimizer = torch.optim.Adam(filter(lambda layer:layer.requires_grad,
                                        model.parameters()),
                                lr=ARGS.lr)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=ARGS.lr*5,
                                                    steps_per_epoch=len(train_dataloader),
                                                    epochs=ARGS.epochs,
                                                    anneal_strategy='linear')
    epoch=1
    if ARGS.pt:
        #Resume training
        checkpoint = torch.load(ARGS.pt)
        saved_epoch = checkpoint['epoch']
        f1 = checkpoint['F1']
        val_loss = checkpoint['loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        if not ARGS.reset:
            #Restore optimizer and scheduler checkpoints if these are not to be reset.
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = saved_epoch

        print(f"Loaded saved checkpoint at EPOCH:{saved_epoch}, Val_loss:{val_loss}, Punc_F1:{f1['punc']}, Case_F1:{f1['case']}")
    
    output = f"{ARGS.output}/{ARGS.exp}"
    os.makedirs(output, exist_ok=True)

    train(model, 
          train_dataloader, 
          valid_dataloader,
          criterion,
          optimizer,
          ARGS.epochs,
          ARGS.print_freq,
          output,
          ARGS.accum_grad,
          ARGS.mode,
          scheduler,
          start_epoch=epoch,
          experiment=ARGS.exp,
          device=device) 

if __name__=='__main__':
    ARGS = args().parse_args()
    main(ARGS)
