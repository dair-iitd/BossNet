from __future__ import absolute_import

import os
import re
import numpy as np
import tensorflow as tf
import nltk

# Global Variables
stop_words=set(["a","an","the"])
UNK_INDEX = 0
GO_SYMBOL_INDEX = 1
EOS_INDEX = 2

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

def vectorize_data(data, word_idx, sentence_size, batch_size, candidates_size, max_memory_size, decoder_vocab, candidate_sentence_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    SZ = []
    QZ = []
    CZ = []
    data.sort(key=lambda x:len(x[0]),reverse=True)
    for i, (story, query, answer, start) in enumerate(data):
        if i%batch_size==0:
            memory_size=max(1,min(max_memory_size,len(story)))
        ss = []
        sizes = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)
            sizes.append(len(sentence))

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]
        sizes = sizes[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)
            sizes.append(0)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq

        aq = max(0, candidate_sentence_size - len(answer) - 1)
        a = [decoder_vocab[w] if w in decoder_vocab else 0 for w in answer] + [EOS_INDEX] + [0] * aq

        S.append(np.array(ss))
        Q.append(np.array(q))
        A.append(np.array(a))
        SZ.append(np.array(sizes))
        QZ.append(np.array([len(query)]))
        CZ.append(np.array([len(answer)+1]))
    return S, Q, A, SZ, QZ, CZ

def vectorize_data_with_surface_form(data, word_idx, sentence_size, batch_size, candidates_size, max_memory_size, decoder_vocab, candidate_sentence_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    SZ = []
    QZ = []
    CZ = []
    S_in_readable_form = []
    Q_in_readable_form = []
    dialogIDs = []
    last_db_results = []

    data.sort(key=lambda x:len(x[0]),reverse=True)
    for i, (story, query, answer, dialog_id) in enumerate(data):
        if i%batch_size==0:
            memory_size=max(1,min(max_memory_size,len(story)))
        ss = []
        sizes = []
        story_string = []

        dbentries =set([])
        dbEntriesRead=False
        last_db_result=""

        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)
            sizes.append(len(sentence))

            story_element = ' '.join([str(x) for x in sentence[:-2]])
            # if the story element is a database response/result
            if 'r_' in story_element and 'api_call' not in story_element:
                dbEntriesRead = True
                if 'r_rating' in story_element:
                    dbentries.add( sentence[0] + '(' + sentence[2] + ')')
            else:
                if dbEntriesRead:
                    last_db_result = '$db : ' + ' '.join([str(x) for x in dbentries])
                    dbentries =set([])
                    dbEntriesRead = False
            
            story_string.append(' '.join([str(x) for x in sentence[-2:]]) + ' : ' + story_element)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]
        sizes = sizes[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)
            sizes.append(0)


        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq
        
        aq = max(0, candidate_sentence_size - len(answer))
        a = [decoder_vocab[w] if w in decoder_vocab else 0 for w in answer] + [0] * aq

        S.append(np.array(ss))
        Q.append(np.array(q))
        A.append(np.array(a))
        SZ.append(np.array(sizes))
        QZ.append(np.array([len(query)]))
        CZ.append(np.array([len(answer)]))
        S_in_readable_form.append(story_string)
        Q_in_readable_form.append(' '.join([str(x) for x in query]))
        last_db_results.append(last_db_result)

        dialogIDs.append(dialog_id)

    return S, Q, A, SZ, QZ, CZ, S_in_readable_form, Q_in_readable_form, last_db_results, dialogIDs

def pad_to_answer_size(pred, size):
    for i, list in enumerate(pred):
        if len(list) >= size:
            pred[i] = list[:size]
        else:
            arr = np.array([0] * (size - len(list)))
            pred[i] = np.append(list, arr)
    return pred

def bleu_accuracy_score(preds, vals, idx2voc):
    total_score = 0.0
    for pred, val in zip(preds, vals):
        reference = [idx2voc[x] for x in pred if x != EOS_INDEX]
        hypothesis = [idx2voc[x] for x in val if x != EOS_INDEX]
        total_score += nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
    return float(total_score) / len(preds)






