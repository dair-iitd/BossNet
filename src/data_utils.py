from __future__ import absolute_import

import os
import re
import numpy as np
import tensorflow as tf
from string import punctuation

__all__ =  ["load_candidates", 
            "get_decoder_vocab", 
            "load_dialog_task", 
            "tokenize", 
            "pad_to_answer_size", 
            "substring_accuracy_score"]

###################################################################################################
#########                                  Global Variables                              ##########
###################################################################################################

stop_words=set(["a","an","the"])
PAD_INDEX = 0
UNK_INDEX = 1
GO_SYMBOL_INDEX = 2
EOS_INDEX = 3

###################################################################################################
#########                                 Dialog Manipulators                            ##########
###################################################################################################

def load_candidates(data_dir, task_id):
    ''' 
        Load Candidate Responses 
    '''
    assert task_id > 0 and task_id < 7
    candidates=[]
    candid_dic={}
    # Get Candidate File
    if task_id==6:
        candidates_f='dialog-babi-task6-dstc2-candidates.txt'
    else:
        candidates_f='dialog-babi-candidates.txt'

    with open(os.path.join(data_dir,candidates_f)) as f:
        for i,line in enumerate(f):
            candid_dic[line.strip().split(' ',1)[1]] = i
            line=tokenize(line.strip())[1:]
            candidates.append(line)
    return candidates,candid_dic

def get_decoder_vocab(data_dir, task_id):
    ''' 
        Load Candidate Vocabulary Space for Decoder 
    '''
    assert task_id > 0 and task_id < 7
    decoder_vocab_to_index={}
    decoder_index_to_vocab={}
    # Pad Symbol
    decoder_vocab_to_index['PAD']=PAD_INDEX
    decoder_index_to_vocab[PAD_INDEX]='PAD'
    # Unknown Symbol
    decoder_vocab_to_index['UNK']=UNK_INDEX
    decoder_index_to_vocab[UNK_INDEX]='UNK'
    # Start Symbol
    decoder_vocab_to_index['GO_SYMBOL']=GO_SYMBOL_INDEX
    decoder_index_to_vocab[GO_SYMBOL_INDEX]='GO_SYMBOL'
    # End Symbol
    decoder_vocab_to_index['EOS']=EOS_INDEX
    decoder_index_to_vocab[EOS_INDEX]='EOS'

    # Get Candidate File
    if task_id==6:
        candidates_f='dialog-babi-task6-dstc2-candidates.txt'
    else:
        candidates_f='dialog-babi-candidates.txt'

    with open(os.path.join(data_dir,candidates_f)) as f:
        for i,line in enumerate(f):
            line=tokenize(line.strip())[1:]
            for word in line:
                if word not in decoder_vocab_to_index:
                    index = len(decoder_vocab_to_index)
                    decoder_vocab_to_index[word]=index
                    decoder_index_to_vocab[index]=word
    return decoder_vocab_to_index,decoder_index_to_vocab

def load_dialog_task(data_dir, task_id, isOOV):
    ''' 
        Load Train, Test, Validation Dialogs 
    '''
    assert task_id > 0 and task_id < 7
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    if isOOV:
        test_file = [f for f in files if s in f and 'tst-OOV' in f][0]
    else: 
        test_file = [f for f in files if s in f and 'tst.' in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = get_dialogs(train_file)
    test_data = get_dialogs(test_file)
    val_data = get_dialogs(val_file)
    return train_data, test_data, val_data

def tokenize(sent):
    '''
        Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent=sent.lower()
    if sent=='<silence>':
        return [sent]
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() not in stop_words]
    if not result:
        result=['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result=result[:-1]
    return result

def parse_dialogs_per_response(lines):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data=[]
    context=[]
    u=None
    r=None
    dialog_id=1
    for line in lines:
        line=line.strip()
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if '\t' in line:
                u, r = line.split('\t')
                u = tokenize(u)
                r = tokenize(r)
                data.append((context[:],u[:],r[:],dialog_id))
                u.append('$u')
                u.append('#'+str(nid))
                r.append('$r')
                r.append('#'+str(nid))
                context.append(u)
                context.append(r)
            else:
                r=tokenize(line)
                r.append('$db')
                r.append('#'+str(nid))
                context.append(r)
        else:
            dialog_id=dialog_id+1
            context=[] # clear context
    return data

def get_dialogs(f):
    '''
        Given a file name, read the file, retrieve the dialogs, 
        and then convert the sentences into a single dialog.
    '''
    with open(f) as f:
        return parse_dialogs_per_response(f.readlines())


###################################################################################################
#########                           Evaluation Metrics & Helpers                         ##########
###################################################################################################

def pad_to_answer_size(pred, size):
    for i, list in enumerate(pred):
        if len(list) >= size:
            pred[i] = list[:size]
        else:
            arr = np.array([PAD_INDEX] * (size - len(list)))
            pred[i] = np.append(list, arr)
    return pred

def is_Sublist(l, s):
    sub_set = False
    if s == []: sub_set = True
    elif s == l: sub_set = True
    elif len(s) > len(l): sub_set = False
    else:
        for i in range(len(l)):
            if l[i] == s[0]:
                n = 1
                if n >= len(s) or (i+n) >= len(l): break
                while (n < len(s)) and (l[i+n] == s[n]):
                    n += 1
                    if n >= len(s) or (i+n) >= len(l): break
                if n == len(s):
                    sub_set = True

    return sub_set

def substring_accuracy_score(preds, vals):
    total_score = 0.0
    for pred, val in zip(preds, vals):
        reference = [x for x in pred if x != EOS_INDEX and x != PAD_INDEX and x != -1]
        hypothesis = [x for x in val if x != EOS_INDEX and x != PAD_INDEX]
        if is_Sublist(reference, hypothesis) == True:
            total_score += 1.0

    return (float(total_score) / len(preds))*100

