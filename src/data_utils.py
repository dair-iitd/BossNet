from __future__ import absolute_import

import os
import re
from measures import moses_multi_bleu
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from string import punctuation
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import f1_score

__all__ =  ["load_candidates", 
            "get_decoder_vocab", 
            "load_dialog_task", 
            "tokenize", 
            "pad_to_answer_size", 
            "substring_accuracy_score",
            "bleu_accuracy_score",
            "new_eval_score",
            "visualize_attention"]

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
    assert task_id > 0 and task_id < 9
    candidates=[]
    candid_dic={}
    # Get Candidate File
    if task_id==6:
        candidates_f='dialog-babi-task6-dstc2-candidates.txt'
    elif task_id==7:
        candidates_f='dialog-babi-task7-camrest676-candidates.txt'
    elif task_id==8:
        candidates_f='dialog-babi-task8-kvret-candidates.txt'
    else:
        candidates_f='dialog-babi-candidates.txt'

    with open(os.path.join(data_dir,candidates_f)) as f:
        for i,line in enumerate(f):
            candid_dic[line.strip().split(' ',1)[1]] = i
            line=tokenize(line.strip())[1:]
            candidates.append(line)
    return candidates,candid_dic

def get_decoder_vocab(data_dir, task_id, vocab_ext):
    ''' 
        Load Candidate Vocabulary Space for Decoder 
    '''
    assert task_id > 0 and task_id < 9
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

    # Jan 9: decode vocab is now generated from train responses and not from candidate file
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and vocab_ext in f][0]
    
    candidate_sentence_size = 0
    responses = get_responses(train_file)
    for response in responses:
        line=tokenize(response.strip())
        if len(line) > candidate_sentence_size:
            candidate_sentence_size = len(line)
        for word in line:
            if word not in decoder_vocab_to_index:
                index = len(decoder_vocab_to_index)
                decoder_vocab_to_index[word]=index
                decoder_index_to_vocab[index]=word
    return decoder_vocab_to_index,decoder_index_to_vocab,candidate_sentence_size+1

def load_dialog_task(data_dir, task_id, vocab_ext):
    ''' 
        Load Train, Test, Validation Dialogs 
    '''
    assert task_id > 0 and task_id < 9
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    if task_id < 6:
        oov_file = [f for f in files if s in f and 'tst-OOV' in f][0]
    test_file = [f for f in files if s in f and 'tst' in f and 'OOV' not in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    mod_file = [f for f in files if s in f and vocab_ext in f][0]
    train_data = get_dialogs(train_file)
    test_data = get_dialogs(test_file)
    val_data = get_dialogs(val_file)
    if task_id > 5:
        oov_data = None
    else:
        oov_data = get_dialogs(oov_file)
    mod_data = get_dialogs(mod_file)
    return train_data, test_data, val_data, oov_data, mod_data

def tokenize(sent):
    '''
        Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent=sent.lower()
    if sent=='<silence>':
        return [sent]
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip()] # and x.strip() not in stop_words]
    if not result:
        result=['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result=result[:-1]
    return result

def get_responses(f):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    responses=[]
    with open(f) as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip()
            if line:
                nid, line = line.split(' ', 1)
                nid = int(nid)
                if '\t' in line:
                    u, r = line.split('\t')
                    responses.append(r)
    return responses

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

def get_surface_form(index_list, word_map, oov_words):
    surface_form = []
    size = len(word_map)
    for i in index_list:
        if i in word_map:
            surface_form.append(word_map[i])
        else:
            idx = i - size
            surface_form.append(oov_words[idx])
    return surface_form

def substring_accuracy_score(preds, vals, d_ids, entities, oov_words, db_words, word_map=None, isTrain=True):
    total_sub_score = 0.0
    total_score = 0.0
    dialog_sub_dict = {}
    dialog_dict = {}
    precision_total = 0.0
    recall_total = 0.0
    entity_score = 0.0
    re = []
    pr = []
    for i, (pred, val) in enumerate(zip(preds, vals)):
        reference = [x for x in pred if x != EOS_INDEX and x != PAD_INDEX and x != -1]
        hypothesis = [x for x in val if x != EOS_INDEX and x != PAD_INDEX]
        if is_Sublist(reference, hypothesis) == True:
            total_sub_score += 1.0 
            if d_ids[i] not in dialog_sub_dict:
                dialog_sub_dict[d_ids[i]] = 1
        else:
            dialog_sub_dict[d_ids[i]] = 0
        ref_surface = get_surface_form(reference, word_map, oov_words[i])
        hyp_surface = get_surface_form(hypothesis, word_map, oov_words[i])
        if reference==hypothesis:
            total_score += 1.0
            if d_ids[i] not in dialog_dict:
                dialog_dict[d_ids[i]] = 1
        else:
            dialog_dict[d_ids[i]] = 0
            # print incorrect results while testing
            # if word_map is not None and isTrain==False:
            #     if is_Sublist(reference, hypothesis) == False:
            #         print('ground truth   : ' + str(hyp_surface))
            #         print('predictions    : ' + str(ref_surface))
            #         print('-----')
        lst = []
        re_temp = []
        pr_temp = []
        punc = ['.', ',', '!', '\'', '\"', '-']
        for j, ref_word in enumerate(hyp_surface):
            if j in entities[i] and ref_word not in punc:
                lst.append(ref_word)
                re_temp.append(1)
                pr_temp.append(0)
        for pred_word in ref_surface:
            if pred_word in lst:
                index = lst.index(pred_word)
                pr_temp[index] = 1
        re += re_temp
        pr += pr_temp

        for j, ref_word in enumerate(hyp_surface):
            if j >= len(ref_surface):
                pred_word = 'NULL'
            else:
                pred_word = ref_surface[j]
            if pred_word in db_words:
                precision_total += 1.0
            if j in entities[i]:
                recall_total += 1.0
                if pred_word == ref_word:
                    entity_score += 1.0
    new_f1 = str(f1_score(re, pr, average='micro'))
    if precision_total != 0:
        entity_precision = float(entity_score) / float(precision_total)
    else:
        entity_precision = 0.0
    if recall_total != 0:
        entity_recall = float(entity_score) / float(recall_total)
    else:
        entity_recall = 0.0
    if entity_precision == 0.0 and entity_recall == 0.0:
        macro_f1_score = str(0.0)
    else:
        macro_f1_score = str(2.0*entity_precision*entity_recall / (entity_precision + entity_recall))
    count = 0.0
    count_sub = 0.0
    count_total = 0.0
    for val in dialog_dict:
        if dialog_dict[val] == 1:
            count += 1.0
        if dialog_sub_dict[val] == 1:
            count_sub += 1.0
        count_total += 1.0
    dialog_sub_accuracy = str(float(count_sub) * 100.0 / count_total)
    dialog_accuracy = str(float(count) * 100.0 / count_total)
    return [str((float(total_sub_score) / len(preds))*100) + ' (' + dialog_sub_accuracy + ')', str((float(total_score) / len(preds))*100)  + ' (' + dialog_accuracy + ')' + ' (' + macro_f1_score + ')' + ' (' + new_f1 + ')']

def get_tokenized_response_from_padded_vector(vector, word_map):
    tokenized_response = []
    for x in vector:
        if x == EOS_INDEX or x == PAD_INDEX:
            return ' '.join(tokenized_response)
        if x in word_map:
            tokenized_response.append(word_map[x])
        else:
            tokenized_response.append('UNK')
    return ' '.join(tokenized_response)

def bleu_accuracy_score(preds, refs, word_map=None, isTrain=True):
    references = []
    hypothesis = []
    
    for pred, ref in zip(preds, refs):
        references.append(get_tokenized_response_from_padded_vector(ref, word_map))
        hypothesis.append(get_tokenized_response_from_padded_vector(pred, word_map))
        
    return moses_multi_bleu(hypothesis, references, True)

def new_eval_score(preds, vals, dbset, word_map=None):
    match=0
    total=0
    match_acc=0

    for pred, val, db_rest in zip(preds, vals, dbset):
        answer = [word_map[x] if x in word_map else 'UNK' for x in val if x != EOS_INDEX and x != PAD_INDEX]

        if ':' in answer:
            total += 1
            ans_pred = val[8]
            pred_rest = pred[8]
            if pred_rest == ans_pred:
                match_acc += 1
            if pred_rest in db_rest:
                match += 1

    return [100.0*match_acc/total, 100.0*match/total]


###################################################################################################
#########                                Visualization Tools                             ##########
###################################################################################################

def visualize_attention(data_batch, hier, line, word, p_gens, count, hierarchy):
    hier = hier.reshape(word.shape)
    answers = data_batch.readable_answers
    for i, ans in enumerate(answers):
        if ':' in ans:
            lst_hier = hier[i][8]
            lst_word = word[i][8]
            lst_line = line[i][8]
            lst_pgen = p_gens[i][8]
            lst_line = lst_line.reshape((lst_line.shape[0], 1))
            words = np.copy(data_batch.readable_stories[i])
            sizes = data_batch.story_sizes[i]
            index = 0
            for j, size in enumerate(sizes):
                words[j][size-2] = ''
                lst_hier[j][size-2:] = 0.0
                words[j][size-1] = ''
                for k in range(0, size-2):
                    if 'resto' in words[j][k]:
                        elements = [sword[:2] for sword in words[j][k].split('_')]
                        if len(elements) == 5:
                            words[j][k] = 'rest_' + elements[4][0] + '_str'
                        else:
                            words[j][k] = 'rest_' + elements[4][0] + '_' + elements[5]
                        if k == 0:
                            index = j
            indexes = list(range(0, lst_hier.shape[0]))
            n_indexes = indexes[:index+1] + indexes[-5:]
            lst_hier = lst_hier[n_indexes]
            lst_line = lst_line[n_indexes]
            words = words[n_indexes]
            sizes = sizes[n_indexes]
            size = np.max(sizes)
            minval = np.min(lst_hier[:, :size-2][np.nonzero(lst_hier[:, :size-2])])
            maxval = np.max(lst_hier[:, :size-2][np.nonzero(lst_hier[:, :size-2])])
            
            # Create Seaborn Attention Graph
            plt.clf()
            plt.figure(figsize = (12,7))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 7])
            
            plt.subplot(gs[0])
            sns.set_context("paper")
            lmaxval = np.max(lst_line)
            if hierarchy:
                ax = sns.heatmap(lst_line, fmt='f', linewidths=2, cmap='hot', cbar=False, xticklabels=False, yticklabels=False, vmax=lmaxval*1.1)
            else:
                ax = sns.heatmap(lst_line, fmt='f', linewidths=2, cmap='hot',cbar=False, xticklabels=False, yticklabels=False, mask=(lst_line!=0.0), vmax=lmaxval*1.1)

            plt.subplot(gs[1])
            sns.set_context("paper")
            print('shape', lst_hier[:, :size-2].shape)
            ax = sns.heatmap(lst_hier[:, :size-2], fmt='s', linewidths=2, cmap='hot', annot=words[:, :size-2], mask=(lst_hier[:, :size-2]==0.0), vmin=minval, vmax=maxval*1.1, yticklabels=False, xticklabels=False)
            plt.tight_layout()
            
            if hierarchy:
                plt.savefig('hier_plots/' + str(count) + '_' + str(i) + '_' + str(8) + '_' + str(lst_pgen) + '_' + "att.png")
            else:
                plt.savefig('no_hier_plots/' + str(count) + '_' + str(i) + '_' + str(8) + '_' + str(lst_pgen) + '_' + "att.png")
    sys.exit()



