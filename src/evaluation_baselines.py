import json
import numpy as np 
from measures import moses_multi_bleu
from sklearn.metrics import f1_score
import pickle as pkl
from collections import defaultdict
import re
import sys
import random

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

def get_surface_form(index_list, word_map, oov_words, context=False):
    size = len(word_map)
    if context:
    	surfaces = []
    	for story_line in index_list:
    		surfaces.append([word_map[i] if i in word_map else oov_words[i - size] for i in story_line])
    	return surfaces
    else:
    	lst = []
    	for i in index_list:
    		if i in word_map:
    			lst.append(i)
    		elif i - size < len(oov_words):
    			lst.append(oov_words[i - size])
    		else:
    			print(i)
    			print(size)
    			print(oov_words)
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
			if token not in punc or i+1 < len(sval) :
				idxs.append(idx)
		tokens.append(idxs)
	return tokens

def merge(ordered, gold_out=True):
	preds = []
	if gold_out:
		golds = []
		dids = []
	if turk == 1:
		queries = []; context = []; answers = []
	else:
		queries = None; context = None; answers = None
	for i in range(1, len(ordered)+1):
		val = ordered[i]
		for dct in val:
			preds.append(dct['preds'])
			if gold_out:
				golds.append(dct['golds'])
				dids.append(i)
			if turk == 1:
				queries.append(dct['queries'])
				context.append(dct['context'])
				answers.append(dct['answers'])
	if gold_out:
		return preds, golds, queries, context, answers, dids
	else:
		return preds

turk = int(sys.argv[1])

## BoSsNet Files
preds = pkl.load(open( "files/pred.pkl", "rb" ))
golds = pkl.load(open( "files/golds.pkl", "rb" ))
word_map = pkl.load(open( "files/word_map.pkl", "rb" ))
index_map = pkl.load(open( "files/index_map.pkl", "rb" ))
entities = pkl.load(open( "files/entities.pkl", "rb" ))
entities2 = pkl.load(open( "files/entities_kb.pkl", "rb" ))
entities3 = pkl.load(open( "files/entities_context.pkl", "rb" ))
dialog_ids = pkl.load(open( "files/dialog_ids.pkl", "rb" ))
oov_words = pkl.load(open( "files/oov_words.pkl", "rb" ))

if turk == 1:
	context = pkl.load(open( "files/context.pkl", "rb" ))
	query = pkl.load(open( "files/query.pkl", "rb" ))
	answers = pkl.load(open( "files/answers.pkl", "rb" ))
else:
	context = None
	query = None
	answers = None

print(query)

## Mem2Seq Files
mem2seq_golds_surf = pkl.load(open( "files/mem2seq_golds.pkl", "rb" ))
mem2seq_preds_surf = pkl.load(open( "files/mem2seq_preds.pkl", "rb" ))
mem2seq_d_ids = pkl.load(open( "files/mem2seq_dids.pkl", "rb" ))

## PTRUNK Files
PTRUNK_golds_surf = pkl.load(open( "files/PTRUNK_golds.pkl", "rb" ))
PTRUNK_preds_surf = pkl.load(open( "files/PTRUNK_preds.pkl", "rb" ))
PTRUNK_d_ids = pkl.load(open( "files/PTRUNK_dids.pkl", "rb" ))

## Vanilla Files
vanilla_golds_surf = pkl.load(open( "files/vanilla_golds.pkl", "rb" ))
vanilla_preds_surf = pkl.load(open( "files/vanilla_preds.pkl", "rb" ))
vanilla_d_ids = pkl.load(open( "files/vanilla_dids.pkl", "rb" ))

preds, golds = process(preds, golds)

ordered_oovs = {}
for num, words in zip(dialog_ids, oov_words):
	if num not in ordered_oovs:
		ordered_oovs[num] = list(words)
	else:
		if len(list(words)) > len(ordered_oovs[num]):
			ordered_oovs[num] = list(words)

mem2seq_preds = tokenize(mem2seq_preds_surf, mem2seq_d_ids)
mem2seq_golds = tokenize(mem2seq_golds_surf, mem2seq_d_ids)

PTRUNK_preds = tokenize(PTRUNK_preds_surf, PTRUNK_d_ids)
PTRUNK_golds = tokenize(PTRUNK_golds_surf, PTRUNK_d_ids)

vanilla_preds = tokenize(vanilla_preds_surf, vanilla_d_ids)
vanilla_golds = tokenize(vanilla_golds_surf, vanilla_d_ids)

ordered_orig = defaultdict(list)
ordered_mem2seq = defaultdict(list)
ordered_PTRUNK = defaultdict(list)
ordered_vanilla = defaultdict(list)

if turk == 1:
	orginal = zip(preds, golds, query, context, answers)
else:
	orginal = zip(preds, golds)

mem2seq = zip(mem2seq_preds, mem2seq_golds)
PTRUNK = zip(PTRUNK_preds, PTRUNK_golds)
vanilla = zip(vanilla_preds, vanilla_golds)

for num, org in zip(dialog_ids, orginal):
	if turk == 1:
		p, g, q, c, a = org
		element_dict = defaultdict(list)
		element_dict['preds'] = p
		element_dict['golds'] = g
		element_dict['queries'] = q
		element_dict['context'] = c
		element_dict['answers'] = a
		ordered_orig[num].append(element_dict)
	else:
		p, g = org
		element_dict = defaultdict(list)
		element_dict['preds'] = p
		element_dict['golds'] = g
		ordered_orig[num].append(element_dict)

for num, org in zip(mem2seq_d_ids, mem2seq):
	p, g = org
	element_dict = defaultdict(list)
	element_dict['preds'] = p
	element_dict['golds'] = g
	ordered_mem2seq[num+1].append(element_dict)

for num, org in zip(PTRUNK_d_ids, PTRUNK):
	p, g = org
	element_dict = defaultdict(list)
	element_dict['preds'] = p
	element_dict['golds'] = g
	ordered_PTRUNK[num+1].append(element_dict)

for num, org in zip(vanilla_d_ids, vanilla):
	p, g = org
	element_dict = defaultdict(list)
	element_dict['preds'] = p
	element_dict['golds'] = g
	ordered_vanilla[num+1].append(element_dict)

preds, golds, queries, context, answers, dids = merge(ordered_orig, True)
preds_mem2seq = merge(ordered_mem2seq, False)
preds_PTRUNK = merge(ordered_PTRUNK, False)
preds_vanilla = merge(ordered_vanilla, False)

print('\nBoSsNet')
print('BLUE : ' + BLEU(preds, golds, word_map))
acc, dial = accuracy(preds, golds, dialog_ids)
print('Accuracy : ' + acc)
print('Dialog Acc. : ' + dial)
print('f1 : ' + f1(preds, golds, entities, word_map))
print('f1 kb : ' + f1(preds, golds, entities2, word_map))
print('f1 context: ' + f1(preds, golds, entities3, word_map))

print('\nMem2Seq')
print('BLUE : ' + BLEU(preds_mem2seq, golds, word_map))
acc, dial = accuracy(preds_mem2seq, golds, dialog_ids)
print('Accuracy : ' + acc)
print('Dialog Acc. : ' + dial)
print('f1 : ' + f1(preds_mem2seq, golds, entities, word_map))
print('f1 kb : ' + f1(preds_mem2seq, golds, entities2, word_map))
print('f1 context: ' + f1(preds_mem2seq, golds, entities3, word_map))

print('\nPTRUNK')
print('BLUE : ' + BLEU(preds_PTRUNK, golds, word_map))
acc, dial = accuracy(preds_PTRUNK, golds, dialog_ids)
print('Accuracy : ' + acc)
print('Dialog Acc. : ' + dial)
print('f1 : ' + f1(preds_PTRUNK, golds, entities, word_map))
print('f1 kb : ' + f1(preds_PTRUNK, golds, entities2, word_map))
print('f1 context: ' + f1(preds_PTRUNK, golds, entities3, word_map))

print('\nVanilla')
print('BLUE : ' + BLEU(preds_vanilla, golds, word_map))
acc, dial = accuracy(preds_vanilla, golds, dialog_ids)
print('Accuracy : ' + acc)
print('Dialog Acc. : ' + dial)
print('f1 : ' + f1(preds_vanilla, golds, entities, word_map))
print('f1 kb : ' + f1(preds_vanilla, golds, entities2, word_map))
print('f1 context: ' + f1(preds_vanilla, golds, entities3, word_map))

if turk:
	size = len(preds)
	print('context_size : {}'.format(len(context)))
	print('query_size : {}'.format(len(queries)))
	print('golds_size : {}'.format(len(golds)))
	print('preds_size : {}'.format(len(preds)))
	print('preds_mem2seq_size : {}'.format(len(preds_mem2seq)))
	print('preds_PTRUNK_size : {}'.format(len(preds_PTRUNK)))
	print('preds_vanilla_size : {}'.format(len(preds_vanilla)))
	c = list(range(1, size+1))
	sample = random.sample(c, 10)
	output_dict = {}
	for i in sample:
		output_dict[i] = {}
		output_dict[i]['d_id'] = dids[i]
		output_dict[i]['context'] = context[i]
		output_dict[i]['query'] = queries[i]
		output_dict[i]['gold'] = answers[i]
		output_dict[i]['boss'] = preds[i]
		output_dict[i]['mem2seq'] = preds_mem2seq[i]
		output_dict[i]['ptrunk'] = preds_PTRUNK[i]
		output_dict[i]['vanilla'] = preds_vanilla[i]
	print(output_dict)
	pkl.dump(output_dict, open( "files/turk.pkl", "wb" ))

