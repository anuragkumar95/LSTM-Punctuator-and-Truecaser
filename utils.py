# -*- coding: utf-8 -*-
"""
Created on Sat Sept 9th 2023
@author: Anurag Kumar
"""
import torch
import torch.nn as nn
from torch.nn.utils import rnn
from tqdm import tqdm 
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support

#RoBERTa pad_index is set to 1.
PAD_ID=-1

class PuncCaseJointLoss(nn.Module):
    def __init__(self, device, loss='CE'):
        super().__init__()
        if loss == 'CE':
            self.loss = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        if loss == 'Focal':
            self.loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='sum')
        self.device = device

    def forward(self, 
                punc_logits, 
                case_logits, 
                punc_labels, 
                case_labels):
        punc_logits = punc_logits.reshape(-1, punc_logits.shape[-1]).to(self.device)
        case_logits = case_logits.reshape(-1, case_logits.shape[-1]).to(self.device)

        #print("LOSS")
        
        punc_labels = punc_labels.to(self.device)
        case_labels = case_labels.to(self.device)
        
        punc_loss = self.loss(punc_logits, punc_labels)
        case_loss = self.loss(case_logits, case_labels)
        
        return punc_loss, case_loss 
    

def collate(data):
    """DATA FORMAT -> (Tuple(tok_seq, case_seq, punct_seq), pad_token_id)"""
    
    #Input data
   
    inputs = [torch.LongTensor(sample[0]) for sample in data]
    lens = [len(i) for i in inputs]
    
    punct_labels = [torch.LongTensor(sample[1]) for sample in data]
    case_labels = [torch.LongTensor(sample[2]) for sample in data]
    spans = [sample[-1] for sample in data]

    roberta_pad_id = PAD_ID + 2
    inputs = rnn.pad_sequence(inputs, batch_first=True, padding_value=roberta_pad_id)
    cases = rnn.pad_sequence(case_labels, batch_first=True, padding_value=PAD_ID)
    puncts = rnn.pad_sequence(punct_labels, batch_first=True, padding_value=PAD_ID)
    
    return inputs, puncts, cases, lens, spans

def score(labels, predictions, print_report=False, label_names=None, method='macro'):
    """
    Returns F1, precision and recall score for 0/1 prediction. 
    ARGS:
        labels      : true labels.
        predictions : model outputs.

    Returns
        F1, precision and recall scores.
    """
    if label_names is not None:
        ignore_indices = {i:1 for i in range(len(labels)) if labels[i]==PAD_ID or predictions[i] not in label_names}
    else:
        ignore_indices = {i:1 for i in range(len(labels)) if labels[i]==PAD_ID}

    labels = [label for i, label in enumerate(labels) if i not in ignore_indices]
    predictions = [pred for i, pred in enumerate(predictions) if i not in ignore_indices]
    
    assert len(predictions) == len(labels), f"Predictions:{len(predictions)} is not equal Labels:{len(labels)}"

    precision = precision_score(labels, predictions, average=method)
    
    recall = recall_score(labels, predictions, average=method)
    
    f1 = f1_score(labels, predictions, average=method)

    class_pre, class_re, class_f1, class_support = precision_recall_fscore_support(labels, predictions, labels=label_names)

    if print_report:
        print(classification_report(labels, predictions))
    
    return precision, recall, f1, {'c_pre':class_pre, 'c_re':class_re, 'c_f1':class_f1, 'c_support':class_support}


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction='none', eps=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target):
        n = input.shape[0]
        out_size = (n,) + input.shape[2:]

        assert target.shape[1:] == input.shape[2:], f'Expected target size {out_size}, got {target.size()}'
        assert input.device == target.device, f"input and target must be in the same device. Got: {input.device} and {target.device}"
        
        # compute softmax over the classes axis
        input_soft = input.softmax(1)
        log_input_soft = input.log_softmax(1)

        # create the labels one hot tensor
        target = target + 1
        target_one_hot = nn.functional.one_hot(target, num_classes=input.shape[1]+1).float()
        target_one_hot = target_one_hot[:, 1:]

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, self.gamma)

        focal = -self.alpha * weight * log_input_soft
        loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss
    

def pre_process(text):
    text = " ".join([word for word in text.strip().split()])
    text = text.replace('!', '.')
    text = text.replace(';', ',')
    text = text.replace(':', ',')
    text = text.replace('-', ' ')
    text = text.replace('--', ' ')
    while('  ' in text):
        text = text.replace('  ', ' ')
    return text

class NLP:
    def __init__(self, string, speaker_labels=None, QAC_labels=None):
        self.headers = ['id',
                        'word',
                        'pre-punct',
                        'post-punct',
                        'case',
                        'QAC',
                        'speaker']

        self.data = self.read(string, speaker_labels, QAC_labels)
        
    def read(self, string, speakers=None, QAC_labels = None):
        words = string.split()
        data = []
        for i, word in enumerate(words):
            if len(word.strip()) == 0:
                continue
                
            attr = {head:'<NA>' for head in self.headers}
            attr['id'] = i

            pre_punct = word[0]
            if pre_punct in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                pre_punct = 'NA'
            else:
                word = word[1:]
            attr['pre-punct'] = pre_punct
            
            if len(word) > 0:
                post_punct = word[-1]
                if post_punct in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                    post_punct = 'NA'
                elif word[-2:] == '--':
                    post_punct = '--'
                    word = word[:-2]
                else:
                    word = word[:-1]
                attr['post-punct'] = post_punct

                if word == word.capitalize():
                        attr['case'] = 'UC'
                elif word == word.upper():
                    attr['case'] = 'CA'
                else:
                    attr['case'] = 'LC'
                attr['word'] = word.lower()
            data.append(attr)
            
        if speakers is not None:   
            assert len(speakers) == len(data), f"Len of speaker labels:{len(speakers)} do not match len of data:{len(data)}"
            for word, spk in zip(data, speakers):
                word['speaker'] = spk  
        
        if QAC_labels is not None:   
            assert len(QAC_labels) == len(data), f"Len of QAC labels:{len(QAC_labels)} do not match len of data:{len(data)}"
            for word, qac in zip(data, QAC_labels):
                word['QAC'] = qac  
        return data
    
    def export(self, path):
        with open(path, 'w') as f:
            f.write(f"{'|'.join(self.headers)}\n")
            for attr in self.data:
                line = []
                for header in self.headers:
                    line.append(str(attr[header]))
                line = "|".join(line)
                f.write(f"{line}\n")

def freeze_layers(model, layers):
    """
    Freezes specific layers of the model.
    ARGS:
        model : instance of the model.
        layer : list of name of the layers to be froze.
    """
    for name, param in model.named_parameters():
        for layer in layers:
            if layer in name and param.requires_grad:
                param.requires_grad = False
    return model 

def inference(model, dataloader, device=None):
    """
    Function to generate speaker verification outputs from the model.
    ARGS:
        model      : instance of the pytorch model.
        dataloader : dataloader to run samples from.
        device     : set device to run inference on gpu.

    Returns
        A list of predictions and targets.
    """
    predictions = {'punc':[], 'case':[], 'qac':[]}
    targets = {'punc':[], 'case':[], 'qac':[]}
    for (inputs, puncts, cases, qac, _, lens, _) in tqdm(dataloader):
        if device:
            inputs = inputs.to(device)
            puncts = puncts.to(device)
            cases = cases.to(device)
            qac = qac.to(device)

        punc_logits, case_logits, qac_logits = model(inputs, 
                                                     lens, 
                                                     batch_first=True)

        puncts = puncts.reshape(-1).detach().cpu().numpy().tolist()
        cases = cases.reshape(-1).detach().cpu().numpy().tolist()
        qac = qac.reshape(-1).detach().cpu().numpy().tolist()
        
        p_preds = torch.argmax(punc_logits, dim=1).detach().cpu().numpy().tolist()
        c_preds = torch.argmax(case_logits, dim=1).detach().cpu().numpy().tolist()
        q_preds = torch.argmax(qac_logits, dim=1).detach().cpu().numpy().tolist()
        
        predictions['punc'].extend(p_preds)
        predictions['case'].extend(c_preds)
        predictions['qac'].extend(q_preds)
        
        targets['punc'].extend(puncts)
        targets['case'].extend(cases)
        targets['qac'].extend(qac)

    return predictions, targets