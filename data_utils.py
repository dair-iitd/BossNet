from __future__ import absolute_import

import os
import re
import numpy as np
import tensorflow as tf

stop_words=set(["a","an","the"])

def load_candidates(data_dir, task_id):
    assert task_id > 0 and task_id < 7
    candidates=[]
    candidates_f=None
    candid_dic={}
    if task_id==6:
        candidates_f='dialog-babi-task6-dstc2-candidates.txt'
    else:
        candidates_f='dialog-babi-candidates.txt'
    with open(os.path.join(data_dir,candidates_f)) as f:
        for i,line in enumerate(f):
            candid_dic[line.strip().split(' ',1)[1]] = i
            line=tokenize(line.strip())[1:]
            candidates.append(line)
    # return candidates,dict((' '.join(cand),i) for i,cand in enumerate(candidates))
    return candidates,candid_dic

def get_decoder_vocab(data_dir, task_id):
    assert task_id > 0 and task_id < 7
    decoder_vocab_to_index={}
    decoder_index_to_vocab={}

    decoder_vocab_to_index['UNK']=0
    decoder_index_to_vocab[0]='UNK'

    decoder_vocab_to_index['GO_SYMBOL']=1
    decoder_index_to_vocab[1]='GO_SYMBOL'

    decoder_vocab_to_index['EOS']=2
    decoder_index_to_vocab[0]='EOS'

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


def load_dialog_task(data_dir, task_id, candid_dic, isOOV):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
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
    train_data = get_dialogs(train_file,candid_dic)
    test_data = get_dialogs(test_file,candid_dic)
    val_data = get_dialogs(val_file,candid_dic)
    return train_data, test_data, val_data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
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


# def parse_dialogs(lines,candid_dic):
#     '''
#         Parse dialogs provided in the babi tasks format
#     '''
#     data=[]
#     context=[]
#     u=None
#     r=None
#     for line in lines:
#         line=str.lower(line.strip())
#         if line:
#             nid, line = line.split(' ', 1)
#             nid = int(nid)
#             if '\t' in line:
#                 u, r = line.split('\t')
#                 u = tokenize(u)
#                 r = tokenize(r)
#                 # temporal encoding, and utterance/response encoding
#                 u.append('$u')
#                 u.append('#'+str(nid))
#                 r.append('$r')
#                 r.append('#'+str(nid))
#                 context.append(u)
#                 context.append(r)
#             else:
#                 r=tokenize(line)
#                 r.append('$r')
#                 r.append('#'+str(nid))
#                 context.append(r)
#         else:
#             context=[x for x in context[:-2] if x]
#             u=u[:-2]
#             r=r[:-2]
#             key=' '.join(r)
#             if key in candid_dic:
#                 r=candid_dic[key]
#                 data.append((context, u,  r))
#             context=[]
#     return data

def parse_dialogs_per_response(lines,candid_dic):
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
                # temporal encoding, and utterance/response encoding
                # data.append((context[:],u[:],candid_dic[' '.join(r)]))
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
            # clear context
            context=[]
    return data



def get_dialogs(f,candid_dic):
    '''Given a file name, read the file, retrieve the dialogs, and then convert the sentences into a single dialog.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_dialogs_per_response(f.readlines(),candid_dic)

def vectorize_candidates_sparse(candidates,word_idx):
    shape=(len(candidates),len(word_idx)+1)
    indices=[]
    values=[]
    for i,candidate in enumerate(candidates):
        for w in candidate:
            indices.append([i,word_idx[w]])
            values.append(1.0)
    return tf.SparseTensor(indices,values,shape)

def vectorize_candidates(candidates,word_idx,sentence_size):
    shape=(len(candidates),sentence_size)
    C=[]
    for i,candidate in enumerate(candidates):
        lc=max(0,sentence_size-len(candidate))
        C.append([word_idx[w] if w in word_idx else 0 for w in candidate] + [0] * lc)
    return tf.constant(C,shape=shape)


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

        aq = max(0, candidate_sentence_size - len(answer))
        a = [decoder_vocab[w] if w in decoder_vocab else 0 for w in answer] + [0] * aq

        S.append(np.array(ss))
        Q.append(np.array(q))
        A.append(np.array(a))
        SZ.append(np.array(sizes))
        QZ.append(np.array([len(query)]))
        CZ.append(np.array([len(answer)]))
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
                    #story_string.append('$db : ' + ' '.join([str(x) for x in dbentries]))
                    last_db_result = '$db : ' + ' '.join([str(x) for x in dbentries])
                    dbentries =set([])
                    dbEntriesRead = False
                #story_string.append(' '.join([str(x) for x in sentence[-2:]]) + ' : ' + story_element)
            
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

def restaurant_reco_evluation(test_preds, testA, indx2candid):
    total = 0
    match = 0
    for idx, val in enumerate(test_preds):
        answer = indx2candid[testA[idx].item(0)]
        prediction = indx2candid[val]
        if "what do you think of this option:" in prediction:
            total = total+1
            if prediction == answer:
                match=match+1
    print('Restaurant Recommendation Accuracy : ' + str(match/float(total)) +  " (" +  str(match) +  "/" + str(total) + ")") 
    
if __name__ == '__main__':
    u = tokenize('The phone number of taj_tandoori is taj_tandoori_phone')
    print(u)