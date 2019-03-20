import json
import numpy as np
from measures import moses_multi_bleu
from sklearn.metrics import f1_score
import pickle as pkl
from collections import defaultdict
import re
import sys
import random

stop_words = set(["a", "an", "the"])
PAD_INDEX = 0
UNK_INDEX = 1
GO_SYMBOL_INDEX = 2
EOS_INDEX = 3


def process(preds, golds):
    for i, (pred, gold) in enumerate(zip(preds, golds)):
        preds[i] = [x for x in pred if x != EOS_INDEX and x != PAD_INDEX]
        golds[i] = [x for x in gold if x != EOS_INDEX and x != PAD_INDEX]
    return preds, golds


def get_surface_form(index_list, word_map, oov_words, context=False):
    size = len(word_map)
    maxs = size + len(oov_words)
    if context:
        surfaces = []
        for story_line in index_list:
            surfaces.append(
                [word_map[i] if i in word_map else oov_words[i - size] for i in story_line])
        return surfaces
    else:
        lst = []
        for i in index_list:
            if i in word_map:
                lst.append(i)
            elif i < maxs:
                lst.append(oov_words[i - size])
        return lst


def surface(index_list, word_map, oov_words, context=False):
    surfaces = []
    for i, lst in enumerate(index_list):
        surfaces.append(get_surface_form(lst, word_map, oov_words[i], context))
    return surfaces


def accuracy(preds, golds, dialog_ids):
    total_score = 0
    dialog_dict = {}

    for i, (pred, gold) in enumerate(zip(preds, golds)):
        if pred == gold:
            total_score += 1
            if dialog_ids[i] not in dialog_dict:
                dialog_dict[dialog_ids[i]] = 1
        else:
            dialog_dict[dialog_ids[i]] = 0

    # Calculate Response Accuracy
    size = len(preds)
    response_accuracy = "{:.2f}".format(
        float(total_score) * 100.0 / float(size))

    # Calculate Dialog Accuracy
    dialog_size = len(dialog_dict)
    correct_dialogs = list(dialog_dict.values()).count(1)
    dialog_accuracy = "{:.2f}".format(
        float(correct_dialogs) * 100.0 / float(dialog_size))

    return response_accuracy, dialog_accuracy


def f1(preds, golds, entities, word_map):
    re = []
    pr = []

    punc = ['.', ',', '!', '\'', '\"', '-', '?']

    for i, (pred, gold) in enumerate(zip(preds, golds)):
        re_temp = []
        pr_temp = []
        lst = []
        for j, ref_word in enumerate(gold):
            if j in entities[i]:
                if ref_word in word_map and word_map[ref_word] in punc:
                    continue
                lst.append(ref_word)
                re_temp.append(1)
                pr_temp.append(0)
        for pred_word in pred:
            if pred_word in lst:
                index = lst.index(pred_word)
                pr_temp[index] = 1

        re += re_temp
        pr += pr_temp

    return "{:.2f}".format(100*f1_score(re, pr, average='micro'))


def get_tokenized_response_from_padded_vector(vector, word_map, oov):
    final = []
    maxs = len(oov) + len(word_map)
    for x in vector:
        if x in word_map:
            final.append(word_map[x])
        elif x < maxs:
            final.append(oov[x - len(word_map)])
        else:
            final.append('UNK')
    return ' '.join(final)


def BLEU(preds, golds, word_map, did, oovs, args):
    tokenized_preds = []
    tokenized_golds = []

    if args.logging:
        file = open(args.logs_dir + 'output.log', 'w+')

    for i, (pred, gold) in enumerate(zip(preds, golds)):
        sent_pred = get_tokenized_response_from_padded_vector(
            pred, word_map, oovs[did[i]])
        sent_gold = get_tokenized_response_from_padded_vector(
            gold, word_map, oovs[did[i]])
        tokenized_preds.append(sent_pred)
        tokenized_golds.append(sent_gold)
        if args.logging:
            file.write("PRED : {}\n".format(sent_pred))
            file.write("GOLD : {}\n".format(sent_gold))
    return "{:.2f}".format(moses_multi_bleu(tokenized_preds, tokenized_golds, True))


def tokenize(vals, dids):
    tokens = []
    punc = ['.', ',', '!', '\'', '\"', '-', '?']
    for i, val in enumerate(vals):
        sval = [x.strip() for x in re.split('(\W+)?', val) if x.strip()]
        idxs = []
        did = dids[i] + 1
        oov_word = ordered_oovs[did]
        sval = [x for x in sval if '$$$$' not in x]
        for i, token in enumerate(sval):
            if token in index_map:
                idx = index_map[token]
            elif token in oov_word:
                idx = len(index_map) + oov_word.index(token)
            else:
                idx = UNK_INDEX
            if token not in punc or i+1 < len(sval):
                idxs.append(idx)
        tokens.append(idxs)
    return tokens


def merge(ordered, gold_out=True):
    preds = []
    golds = []
    dids = []
    for i in range(1, len(ordered)+1):
        val = ordered[i]
        for dct in val:
            preds.append(dct['preds'])
            golds.append(dct['golds'])
            dids.append(i)
    return preds, golds, dids


def evaluate(args, glob, predictions, data):
    preds = predictions
    golds = data.answers.copy()
    word_map = glob['idx_decode']
    index_map = glob['decode_idx']
    entities = data.entities
    dialog_ids = data.dialog_ids
    oov_words = data.oov_words

    preds, golds = process(preds, golds)

    ordered_oovs = {}
    for num, words in zip(dialog_ids, oov_words):
        if num not in ordered_oovs:
            ordered_oovs[num] = list(words)
        else:
            if len(list(words)) > len(ordered_oovs[num]):
                ordered_oovs[num] = list(words)

    ordered_orig = defaultdict(list)

    orginal = zip(preds, golds)

    for num, org in zip(dialog_ids, orginal):
        p, g = org
        element_dict = defaultdict(list)
        element_dict['preds'] = p
        element_dict['golds'] = g
        ordered_orig[num].append(element_dict)

    preds, golds, dids = merge(ordered_orig, True)

    output = {}
    output['bleu'] = float(
        BLEU(preds, golds, word_map, dids, ordered_oovs, args))
    acc, dial = accuracy(preds, golds, dids)
    output['acc'] = float(acc)
    output['dialog'] = float(dial)
    output['f1'] = float(f1(preds, golds, entities, word_map))
    if args.bleu_score:
        output['comp'] = output['bleu']
    else:
        output['comp'] = output['acc']
    return output
