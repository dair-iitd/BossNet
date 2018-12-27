from __future__ import absolute_import

import os
import re
import json
import sys
import pickle as pkl
from measures import moses_multi_bleu
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import f1_score

__all__ =  ["get_decoder_vocab", 
            "load_dialog_task", 
            "tokenize", 
            "pad_to_answer_size", 
            "substring_accuracy_score",
            "bleu_accuracy_score",
            "new_eval_score",
            "visualize_attention",
            "split_output",
            "analyse_pgens",
            "create_batches"]

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

def get_decoder_vocab(data_dir, task_id):
    ''' 
        Load Vocabulary Space for Response-Decoder 
    '''
    assert task_id > 0 and task_id < 9
    decoder_vocab_to_index={}
    decoder_vocab_to_index['PAD']=PAD_INDEX             # Pad Symbol
    decoder_vocab_to_index['UNK']=UNK_INDEX             # Unknown Symbol
    decoder_vocab_to_index['GO_SYMBOL']=GO_SYMBOL_INDEX # Start Symbol
    decoder_vocab_to_index['EOS']=EOS_INDEX             # End Symbol

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    candidate_sentence_size = 0
    responses = get_responses(train_file)
    for response in responses:
        line=tokenize(response.strip())
        candidate_sentence_size = max(len(line), candidate_sentence_size)
        for word in line:
            if word not in decoder_vocab_to_index:
                index = len(decoder_vocab_to_index)
                decoder_vocab_to_index[word]=index
    decoder_index_to_vocab = {v: k for k, v in decoder_vocab_to_index.items()}
    return decoder_vocab_to_index, decoder_index_to_vocab, candidate_sentence_size+3 #1(EOS) 2(#u/r) 3(#turn)

def get_responses(file):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    responses=[]
    with open(file) as f:
        for line in f.readlines():
            line=line.strip()
            if line and '\t' in line:
                u, r = line.split('\t')
                responses.append(r)
    return responses


def load_dialog_task(data_dir, task_id):
    ''' 
        Load Train, Test, Validation Dialogs 
    '''
    assert task_id > 0 and task_id < 9
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    test_file = [f for f in files if s in f and 'tst' in f and 'OOV' not in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = parse_dialogs(train_file)
    test_data = parse_dialogs(test_file)
    val_data = parse_dialogs(val_file)
    if task_id < 6:
        oov_file = [f for f in files if s in f and 'tst-OOV' in f][0]
        oov_data = parse_dialogs(oov_file)
    else:
        oov_data = None        
    return train_data, test_data, val_data, oov_data

def tokenize(sent):
    '''
        Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent=sent.lower()
    if sent=='<silence>': return [sent]
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    if not result:
        result=['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result=result[:-1]
    return result

def parse_dialogs(file):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data=[]; context=[]
    dialog_id=1; turn_id=1
    for line in open(file).readlines():
        line=line.strip()
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if '\t' in line:
                u, r = map(tokenize, line.split('\t'))
                data.append((context, u, r, dialog_id, turn_id))
                u.extend(['$u', '#{}'.format(nid)])
                r.extend(['$r', '#{}'.format(nid)])
                context.append(u); context.append(r)
                turn_id += 1
            else:
                r=tokenize(line)
                r.extend(['$db', '#{}'.format(nid)])
                context.append(r)
        else:
            # clear context / start of new dialog
            dialog_id+=1; turn_id=1
            context=[]
    return data

###################################################################################################
#########                                     Data Batching                              ##########
###################################################################################################

def create_batches(data, batch_size):
    '''
        Helps to partition the dialog into three groups
        1) Dialogs occuring before an API call
        2) Dialogs that have an API call
        3) Dialogs that occur after and API call
    '''
    size = len(data.stories)
    batches = zip(range(0, size - batch_size, batch_size),
                  range(batch_size, size, batch_size))
    batches = [(start, end) for start, end in batches]
    # fix to include last batch
    if batches[-1][1] < size:
        batches.append((batches[-1][1], size))
    return batches


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

def substring_accuracy_score(preds, vals, d_ids, entities, entities_kb, entities_context, oov_words, db_words, context, query, answers, inv_word_map, word_map=None, isTrain=True):
    pkl.dump(preds, open( "files/pred.pkl", "wb" ))
    pkl.dump(vals, open( "files/golds.pkl", "wb" ))
    pkl.dump(word_map, open( "files/word_map.pkl", "wb" ))
    pkl.dump(inv_word_map, open( "files/index_map.pkl", "wb" ))
    pkl.dump(entities, open( "files/entities.pkl", "wb" ))
    pkl.dump(entities_kb, open( "files/entities_kb.pkl", "wb" ))
    pkl.dump(entities_context, open( "files/entities_context.pkl", "wb" ))
    pkl.dump(d_ids, open( "files/dialog_ids.pkl", "wb" ))
    pkl.dump(oov_words, open( "files/oov_words.pkl", "wb" ))
    pkl.dump(context, open( "files/context.pkl", "wb" ))
    pkl.dump(query, open( "files/query.pkl", "wb" ))
    pkl.dump(answers, open( "files/answers.pkl", "wb" ))

    total_sub_score = 0.0
    total_score = 0.0
    dialog_sub_dict = {}
    dialog_dict = {}

    precision_total = 0.0
    recall_total = 0.0
    entity_score = 0.0

    precision_total_kb = 0.0
    recall_total_kb = 0.0
    entity_score_kb = 0.0

    precision_total_context = 0.0
    recall_total_context = 0.0
    entity_score_context = 0.0

    re = []
    pr = []
    re_kb = []
    pr_kb = []
    re_context = []
    pr_context = []

    out_actuals = {}
    out_preds = {}
    with open('output.log', 'w') as f:
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
            f.write('\nground truth   : ' + str(hyp_surface))
            f.write('\npredictions    : ' + str(ref_surface))
            
            dict_size = len(out_actuals)
            out_actuals[dict_size] = hyp_surface
            out_preds[dict_size] = ref_surface
            lst = []
            lst_kb = []
            lst_context = []
            re_temp = []
            pr_temp = []
            re_temp_kb = []
            pr_temp_kb = []
            re_temp_context = []
            pr_temp_context = []
            punc = ['.', ',', '!', '\'', '\"', '-']
            for j, ref_word in enumerate(hyp_surface):
                if j in entities[i] and ref_word not in punc:
                    lst.append(ref_word)
                    re_temp.append(1)
                    pr_temp.append(0)
                if j in entities_kb[i] and ref_word not in punc:
                    lst_kb.append(ref_word)
                    re_temp_kb.append(1)
                    pr_temp_kb.append(0)
                if j in entities_context[i] and ref_word not in punc:
                    lst_context.append(ref_word)
                    re_temp_context.append(1)
                    pr_temp_context.append(0)

            for pred_word in ref_surface:
                if pred_word in lst:
                    index = lst.index(pred_word)
                    pr_temp[index] = 1
                if pred_word in lst_kb:
                    index = lst_kb.index(pred_word)
                    pr_temp_kb[index] = 1
                if pred_word in lst_context:
                    index = lst_context.index(pred_word)
                    pr_temp_context[index] = 1
            re += re_temp
            pr += pr_temp
            re_kb += re_temp_kb
            pr_kb += pr_temp_kb
            re_context += re_temp_context
            pr_context += pr_temp_context

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
    new_f1 = str(100*f1_score(re, pr, average='micro'))
    new_f1_kb = str(100*f1_score(re_kb, pr_kb, average='micro'))
    new_f1_context = str(100*f1_score(re_context, pr_context, average='micro'))

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
        macro_f1_score = str(100*2.0*entity_precision*entity_recall / (entity_precision + entity_recall))
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
    with open('actuals_sys.txt', 'w') as f:
        json.dump(out_actuals, f)
    with open('preds_sys.txt', 'w') as f:
        json.dump(out_preds, f)
    return [str((float(total_sub_score) / len(preds))*100), dialog_sub_accuracy, str((float(total_score) / len(preds))*100), dialog_accuracy, macro_f1_score, new_f1, new_f1_kb, new_f1_context]

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

def split_output(output):
    out_dict = {}
    if len(output) > 1:
        out_dict['bleu'] = output[1]
    first = output[0]
    out_dict['sub_acc'] = first[0]
    out_dict['sub_dialog'] = first[1]
    out_dict['acc'] = first[2]
    out_dict['dialog'] = first[3]
    out_dict['my_f1'] = first[4]
    out_dict['f1'] = first[5]
    out_dict['f1_kb'] = first[6]
    out_dict['f1_context'] = first[7]
    return out_dict

###################################################################################################
#########                                Visualization Tools                             ##########
###################################################################################################

def analyse_pgens(mask, pgens):
    c0 = 0
    n0 = 0.0
    c1 = 0
    n1 = 0.0
    for i, vals in enumerate(mask):
        for j, val in enumerate(vals):
            if val == 0:
                c0+=1
                n0+=pgens[i][j][0]
            else:
                c1+=1
                n1+=pgens[i][j][0]
    return c0, n0, c1, n1

def visualize_attention(data_batch, hier, line, word, p_gens, count, hierarchy):
    hier = hier.reshape(word.shape)
    answers = data_batch.readable_answers
    pred_index = 8
    for i, ans in enumerate(answers):
        if ':' in ans:
            print(ans)
            lst_hier = hier[i][pred_index]
            lst_word = word[i][pred_index]
            lst_line = line[i][pred_index]
            lst_pgen = p_gens[i][pred_index]
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
            # n_indexes = indexes
            lst_hier = lst_hier[n_indexes]
            lst_line = lst_line[n_indexes]
            print (lst_hier)
            # lst_hier = lst_hier / lst_line[0][:,None]
            # lst_hier = np.log(lst_hier)
            print (lst_hier)
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
            ax = sns.heatmap(lst_hier[:, :size], fmt='s', linewidths=2, cmap='hot', annot=words[:, :size], mask=(lst_hier[:, :size]==0.0), vmin=minval, vmax=maxval*1.1, yticklabels=False, xticklabels=False)
            plt.tight_layout()
            
            if hierarchy:
                plt.savefig('hier_plots/' + str(count) + '_' + str(i) + '_' + str(pred_index) + '_' + str(lst_pgen) + '_' + "att.png")
            else:
                plt.savefig('no_hier_plots/' + str(count) + '_' + str(i) + '_' + str(pred_index) + '_' + str(lst_pgen) + '_' + "att.png")
            # sys.exit()



