# -*- coding: utf-8 -*-
"""
Created on Sat Sept 9th 2023
@author: Anurag Kumar
"""

from tqdm import tqdm
import os
from torch.utils.data import Dataset


class Post_Processing_Dataset(Dataset):
    """
    This can only punctuate with COMMA, PERIOD, EXCLAMATION, QUESTION, thus we filter
    out other punctuations and replace them with one of the above four. 
    The mapping is subjective and can be changed
    """
    def __init__(self, root, tokenizer, punc_labels, case_labels):
        """
        ARGS:
            root : (str) path to the root dir containing files.
            tokenizer : (model) instance to the tokenizer.
            punc_labels : (List[int]) punctuation labels map.
            case_labels : (List[Int]) case labels map.
        """
        print(f"Initialising Punctuation and Case restoration Dataset from {root}...")
        self.tokenizer = tokenizer
        self.punc_labels = punc_labels
        self.case_labels = case_labels
        self.replace = [(':',','),
                        (';',','),
                        ('!','.'),
                        ('--',''),
                        ('*',''),
                        ('\\',' '),
                        ('  ',' '),
                        ('-',' '),
                        ('[',''),
                        (']',''),
                        ('..','.'),
                        ('-',' '),
                        ('_',' ')]
        
        paths = [os.path.join(root, path) for path in os.listdir(root)]
        self.data = self.parse_files(paths)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def generate_seqs(self, fp):
        for line in tqdm(fp.readlines()):
            TOKEN_IDS = []
            PUNCT = []
            CASE = []
            SPANS = []
            WORDS = []
            WORD_LABELS = []
            tok_idx = 0
            words = line.split()
            for k, word in enumerate(words):
                #Empty word, skip
                if len(word) == 0:
                    continue
                for key, val in self.replace:
                    word = word.replace(key, val)
                if len(word) == 0:
                    continue
                punct = None
                case = None

                #store punct labels on word level
                punct = word[-1]
                if punct not in ['.', ',', '?']:
                    punct = 'NA'
                if punct != 'NA':
                    word = word[:-1]
                     
                #Store case labels on word level
                if word == word.capitalize():
                    case = 'UC'
                elif word == word.upper():
                    case = 'CA'
                else:
                    case = 'LC'

                word = word.lower()
                token_ids = self.tokenizer.encode(word)
                span = []
                WORDS.append(word)
                WORD_LABELS.append(self.punc_labels.get(punct, 'NA'))
                for t_id in token_ids:
                    TOKEN_IDS.append(t_id)
                    if punct is not None:
                        PUNCT.append(self.punc_labels.get(punct, 'NA'))
                    if case is not None:
                        CASE.append(self.case_labels.get(case, 'LC'))
                    span.append(tok_idx)
                    tok_idx += 1
                SPANS.append(span)
            
            yield (TOKEN_IDS, PUNCT, CASE, SPANS, WORDS, WORD_LABELS)

    def parse_files(self, paths):
        print(f"Parsing files...")
        all_data = []
        for path in tqdm(paths):
            with open(path, encoding='utf-8', errors='ignore') as fp:
                for seq in self.generate_seqs(fp):
                    if len(seq[0]) > 0: 
                        all_data.append(seq)
        return all_data