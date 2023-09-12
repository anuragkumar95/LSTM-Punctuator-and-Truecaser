# -*- coding: utf-8 -*-
"""
Created on Sat Sept 9th 2023
@author: Anurag Kumar
"""

import torch
import torch.nn as nn
from torch.nn.utils import rnn

RoBERTA_PAD_ID = 1 

class JointPostProcess(nn.Module):
    def __init__(self, 
                 encoder, 
                 embedding_dim, 
                 hidden_dim, 
                 num_layers,
                 punct_classes,
                 case_classes, 
                 dropout_prob=0.05, 
                 batch_first=False,
                 bi_directional=False,
                 window=1, 
                 device='cpu'):
        super().__init__()
        #Word embedding
        self.encoder = encoder

        gru_input_dim = embedding_dim
        self.window = window
        if window > 1:
            gru_input_dim = gru_input_dim * self.window

        #Recurrent Layer
        self.gru = nn.GRU(input_size=gru_input_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=num_layers, 
                          batch_first=batch_first, 
                          dropout=dropout_prob,
                          bidirectional=bi_directional)
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bi_directional = bi_directional
    
        #Fully connected layers
        input_dim = hidden_dim
        if self.bi_directional:
            input_dim = 2*input_dim
        self.punct_fc = nn.Linear(input_dim, punct_classes)
        self.case_fc = nn.Linear(input_dim, case_classes)

        self.device = device

    def init_hidden_state(self, h):
        return nn.init.xavier_uniform_(h, gain=nn.init.calculate_gain('relu'))
    
    def create_win5(self, inputs):
        """
        Creates windows of input tokens of length 5
        Accepts a inputs of shape (batch, seq_len, dim) and outputs a tensor of shape (batch, seq_len, dim*5)
        """
        batchsize, seq_len, emb_dim = inputs.shape
        
        #create the padding for edges 
        pad_1 = torch.cat((torch.ones(batchsize, 1, 2*emb_dim).to(self.device), inputs[:, 0, :].unsqueeze(1), inputs[:, 1, :].unsqueeze(1), inputs[:, 2, :].unsqueeze(1)), dim=-1)
        pad_2 = torch.cat((torch.ones(batchsize, 1, emb_dim).to(self.device), inputs[:, 0, :].unsqueeze(1), inputs[:, 1, :].unsqueeze(1), inputs[:, 2, :].unsqueeze(1), inputs[:, 3, :].unsqueeze(1)), dim=-1)
        pad_3 = torch.cat((inputs[:, -4, :].unsqueeze(1), inputs[:, -3, :].unsqueeze(1), inputs[:, -2, :].unsqueeze(1), inputs[:, -1, :].unsqueeze(1), torch.ones(batchsize, 1, emb_dim).to(self.device)), dim=-1)
        pad_4 = torch.cat((inputs[:, -3, :].unsqueeze(1), inputs[:, -2, :].unsqueeze(1), inputs[:, -1, :].unsqueeze(1), torch.ones(batchsize, 1, 2*emb_dim).to(self.device)), dim=-1)

        #window the input with size 5 and step 1
        inputs = inputs.unfold(1, 5, 1)
        inputs = inputs.reshape(batchsize, seq_len-4, 5*emb_dim)
        inputs = torch.cat((pad_1, pad_2, inputs, pad_3, pad_4), dim=1)
        return inputs
    
    def get_embedding(self, packed_input, batch_size, batch_first=False):
        # Initializing hidden state for first input with zeros
        num_layers = self.num_layers
        if self.bi_directional:
            num_layers = 2*self.num_layers
        if batch_first:
            h0 = torch.zeros(num_layers, batch_size, self.hidden_dim).requires_grad_()
        else:
            h0 = torch.zeros(batch_size, num_layers, self.hidden_dim).requires_grad_()
        h0 = self.init_hidden_state(h0)

        if self.device != 'cpu':
            h0 = h0.to(self.device)
            packed_input = packed_input.to(self.device)

        if self.window > 1:
            packed_input = self.create_win5(packed_input, window_len=self.window)
        
        #Feed-forward GRU
        gru_outputs, _ = self.gru(packed_input, h0)
        gru_outputs, lens = rnn.pad_packed_sequence(gru_outputs, batch_first=True)

        return gru_outputs, lens
        
    def forward(self, inputs, lens, batch_first=False):
        """
        ARGS:
            inputs : (Long.tensor) padded tensor of shape (batch, tokens_maxlen).
            lens   : (List[Int]) lengths of tokens in each training example.
        """  
        # Initializing hidden state for first input with zeros
        num_layers = self.num_layers
        if self.bi_directional:
            num_layers = 2*self.num_layers
        if batch_first:
            h0 = torch.zeros(num_layers, inputs.shape[0], self.hidden_dim).requires_grad_()
        else:
            h0 = torch.zeros(inputs.shape[0], num_layers, self.hidden_dim).requires_grad_()
        h0 = self.init_hidden_state(h0)
        
        st = 0
        encoded = None
        while(st < inputs.shape[1]):
            end = min(inputs.shape[1], st + 512)
            part_inp = inputs[:, st:end]
            part_enc = self.encoder.extract_features(part_inp)
            if encoded is None:
                encoded = part_enc
            else:
                encoded = torch.cat((encoded, part_enc), dim = 1)
            st += 512  
                
        if self.window > 1:
            if self.window != 5:
                raise NotImplementedError
            encoded = self.create_win5(encoded, window_len=self.window)
        
        encoded = rnn.pack_padded_sequence(encoded, lens, batch_first=True, enforce_sorted=False)
        
        if self.device != 'cpu':
            h0 = h0.to(self.device)
            encoded = encoded.to(self.device)
        
        #Feed-forward GRU
        gru_outputs, _ = self.gru(encoded, h0)
        gru_outputs, lens = rnn.pad_packed_sequence(gru_outputs, batch_first=True)

        #Feed-forward punctuation, case embedding
        punct_out = self.punct_fc(gru_outputs)
        case_out = self.case_fc(gru_outputs)

        return punct_out, case_out