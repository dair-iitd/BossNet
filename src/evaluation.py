import json
import numpy as np 
from measures import moses_multi_bleu
from sklearn.metrics import f1_score
import pickle as pkl
from collections import defaultdict
import re

stop_words=set(["a","an","the"])
PAD_INDEX = 0
UNK_INDEX = 1
GO_SYMBOL_INDEX = 2
EOS_INDEX = 3

def process(preds, golds):
	for i, (pred, gold) in enumerate(zip(preds, golds)):
		preds[i] = [x for x in pred if x != EOS_INDEX and x != PAD_INDEX]
		golds[i] = [x for x in gold if x != EOS_INDEX and x != PAD_INDEX]
	return preds, golds

def get_surface_form(index_list, word_map, oov_words):
    size = len(word_map)
    return [word_map[i] if i in word_map else oov_words[i - size] for i in index_list]

def surface(preds, golds, word_map, oov_words):
	surface_preds = []
	surface_golds = []
	for i, (pred, gold) in enumerate(zip(preds, golds)):
		surface_preds.append(get_surface_form(pred, word_map, oov_words[i]))
		surface_golds.append(get_surface_form(gold, word_map, oov_words[i]))
	return surface_preds, surface_golds

def accuracy(preds, golds, dialog_ids):
	total_score = 0
	dialog_dict = {}

	for i, (pred, gold) in enumerate(zip(preds, golds)):
		if pred==gold:
			total_score += 1
			if dialog_ids[i] not in dialog_dict:
				dialog_dict[dialog_ids[i]] = 1
		else:
			dialog_dict[dialog_ids[i]] = 0
	
	# Calculate Response Accuracy
	size = len(preds)
	response_accuracy = "{:.2f}".format(float(total_score) * 100.0 / float(size))		

	# Calculate Dialog Accuracy
	dialog_size = len(dialog_dict)
	correct_dialogs = list(dialog_dict.values()).count(1)
	dialog_accuracy = "{:.2f}".format(float(correct_dialogs) * 100.0 / float(dialog_size))

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

def get_tokenized_response_from_padded_vector(vector, word_map):
    return ' '.join([word_map[x] if x in word_map else 'UNK' for x in vector])

def BLEU(preds, golds, word_map):
    tokenized_preds = []
    tokenized_golds = []

    for pred, gold in zip(preds, golds):
        tokenized_preds.append(get_tokenized_response_from_padded_vector(pred, word_map))
        tokenized_golds.append(get_tokenized_response_from_padded_vector(gold, word_map))

    pkl.dump(tokenized_golds, open( "files/tokenized_golds.pkl", "wb" ))
    pkl.dump(tokenized_preds, open( "files/tokenized_preds.pkl", "wb" ))

    return "{:.2f}".format(moses_multi_bleu(tokenized_preds, tokenized_golds, True))

def tokenize(vals, dids):
	tokens = []
	punc = ['.', ',', '!', '\'', '\"', '-', '?']
	for i, val in enumerate(vals):
		# val = ' \' '.join(val.split('.'))
		# sval = val.split() 
		sval = [x.strip() for x in re.split('(\W+)?', val) if x.strip()]
		idxs = []
		did = dids[i]
		oov_word = ordered_oovs[did+1]
		for i, token in enumerate(sval):
			if token in index_map:
				idx = index_map[token]
			elif token in oov_word:
				idx = len(index_map) + oov_word.index(token)
			else:
				print(token)
				print(sval)
				idx = UNK_INDEX
			if token not in punc or i+1 < len(sval) :
				idxs.append(idx)
		tokens.append(idxs)
	return tokens

def merge(ordered_orig, ordered_mem2seq):
	preds = []
	preds_mem2seq = []
	golds = []
	for i in range(1, len(ordered_orig)+1):
		val1 = ordered_orig[i]
		val2 = ordered_mem2seq[i]
		for (p, g) in val1:
			preds.append(p)
			golds.append(g)
		for (p, g) in val2:
			preds_mem2seq.append(p)
	return preds, preds_mem2seq, golds

preds = pkl.load(open( "files/pred.pkl", "rb" ))
golds = pkl.load(open( "files/golds.pkl", "rb" ))
word_map = pkl.load(open( "files/word_map.pkl", "rb" ))
index_map = pkl.load(open( "files/index_map.pkl", "rb" ))
entities = pkl.load(open( "files/entities.pkl", "rb" ))
entities2 = pkl.load(open( "files/entities_kb.pkl", "rb" ))
entities3 = pkl.load(open( "files/entities_context.pkl", "rb" ))
dialog_ids = pkl.load(open( "files/dialog_ids.pkl", "rb" ))
oov_words = pkl.load(open( "files/oov_words.pkl", "rb" ))

mem2seq_golds_surf = pkl.load(open( "files/mem2seq_golds.pkl", "rb" ))
mem2seq_pred_surf = pkl.load(open( "files/mem2seq_preds.pkl", "rb" ))
mem2seq_d_ids = pkl.load(open( "files/mem2seq_dids.pkl", "rb" ))

preds, golds = process(preds, golds)
surf_preds, surf_golds = surface(preds, golds, word_map, oov_words)

ordered_oovs = {}

for num, words in zip(dialog_ids, oov_words):
	if num not in ordered_oovs:
		ordered_oovs[num] = list(words)

mem2seq_preds = tokenize(mem2seq_pred_surf, mem2seq_d_ids)
mem2seq_golds = tokenize(mem2seq_golds_surf, mem2seq_d_ids)

ordered_orig = defaultdict(list)
ordered_mem2seq = defaultdict(list)

orginal = zip(preds, golds)
mem2seq = zip(mem2seq_preds, mem2seq_golds)

for num, org in zip(dialog_ids, orginal):
	ordered_orig[num].append(org)

for num, org in zip(mem2seq_d_ids, mem2seq):
	ordered_mem2seq[num+1].append(org)

preds, preds_mem2seq, golds = merge(ordered_orig, ordered_mem2seq)

print('BLUE : ' + BLEU(preds, golds, word_map))
acc, dial = accuracy(preds, golds, dialog_ids)
print('Accuracy : ' + acc)
print('Dialog Acc. : ' + dial)
print('f1 : ' + f1(preds, golds, entities, word_map))
print('f1 kb : ' + f1(preds, golds, entities2, word_map))
print('f1 context: ' + f1(preds, golds, entities3, word_map))

print('BLUE : ' + BLEU(preds_mem2seq, golds, word_map))
acc, dial = accuracy(preds_mem2seq, golds, dialog_ids)
print('Accuracy : ' + acc)
print('Dialog Acc. : ' + dial)
print('f1 : ' + f1(preds_mem2seq, golds, entities, word_map))
print('f1 kb : ' + f1(preds_mem2seq, golds, entities2, word_map))
print('f1 context: ' + f1(preds_mem2seq, golds, entities3, word_map))

