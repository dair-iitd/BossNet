from __future__ import absolute_import
from __future__ import print_function

import json
import logging
import numpy as np
import os
import pdb
import sys
import tensorflow as tf
from data import Data, Batch
from data_utils import *
from evaluation import evaluate
from itertools import chain
from memn2n.memn2n_dialog_generator import MemN2NGeneratorDialog
from operator import itemgetter
from params import get_params, print_params
from six.moves import range, reduce
from sklearn import metrics
from tqdm import tqdm

args = get_params()
glob = {}

class chatBot(object):

	def __init__(self):
		# Create Model Store Directory
		self.model_dir = args.model_dir + "task" + str(args.task_id) + "_" + args.data_dir.split('/')[-2] + "_lr-" + str(args.learning_rate) + "_hops-" + str(args.hops) + "_emb-size-" + str(args.embedding_size) + "_sw-" + str(args.soft_weight) + "_wd-" + str(args.word_drop_prob) + "_pw-" + str(args.p_gen_loss_weight) + "_model/"
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

		''' Two Vocabularies
		1) Decoder Vocab [decode_idx, idx_decode] 	# Used to Encode Response by Response-Decoder
		2) Context Vocab [word_idx, idx_word] 		# Used to Encode Context Input by Encoder
		'''

		# 1) Load Response-Decoder Vocabulary
		glob['decode_idx'], glob['idx_decode'], glob['candidate_sentence_size'] = get_decoder_vocab(args.data_dir, args.task_id)
		print("Decoder Vocab Size : {}".format(len(glob['decode_idx'])))
		print("candidate_sentence_size : {}".format(glob['candidate_sentence_size'])); sys.stdout.flush()
		# Retreive Task Data
		self.trainData, self.testData, self.valData, self.testOOVData = load_dialog_task(args.data_dir, args.task_id)
		
		# 2) Build the Context Vocabulary
		self.build_vocab(self.trainData)

		# Define MemN2N + Generator Model
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		glob['session'] = tf.Session(config=config)
		glob['optimizer'] = tf.train.AdamOptimizer(learning_rate=args.learning_rate, epsilon=args.epsilon)
		self.model = MemN2NGeneratorDialog(args, glob)
		self.saver = tf.train.Saver(max_to_keep=4)

	def build_vocab(self, data):
		'''
			Get vocabulary from the Train data
		'''
		vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, a, _, _ in data))
		vocab = sorted(vocab)
		glob['word_idx'] = dict((c, i + 2) for i, c in enumerate(vocab))
		glob['word_idx']['']=0
		glob['word_idx']['UNK']=1
		glob['vocab_size'] = len(glob['word_idx']) + 1  # +1 for nil word
		glob['idx_word'] = {v: k for k, v in glob['word_idx'].items()}
		print("Context Vocab Size : {}".format(glob['vocab_size'])); sys.stdout.flush()

		sentence_size = max(map(len, chain.from_iterable(s for s, _, _, _, _ in data)))
		query_size = max(map(len, (q for _, q, _, _, _ in data)))
		glob['sentence_size'] = max(query_size, sentence_size)

	def train(self):
		'''
			Train the model
		'''
		print("------------------------")
		# Get Data in usable form
		Data_train = Data(self.trainData, args, glob)
		n_train = len(Data_train.stories)
		print("Training Size", n_train)

		Data_val = Data(self.valData, args, glob)
		n_val = len(Data_val.stories)
		print("Validation Size", n_val)

		Data_test = Data(self.testData, args, glob)
		n_test = len(Data_test.stories)
		print("Test Size", n_test)

		if args.task_id < 6:
			Data_test_OOV = Data(self.testOOVData, args, glob)
			n_oov = len(Data_test_OOV.stories)
			print("Test OOV Size", n_oov)
		sys.stdout.flush()

		# Create Batches
		batches_train = create_batches(Data_train, args.batch_size)
		batches_val = create_batches(Data_val, args.batch_size)
		batches_test = create_batches(Data_test, args.batch_size)
		if args.task_id < 6:
			batches_oov = create_batches(Data_test_OOV, args.batch_size)

		# Look for previously saved checkpoint
		if args.save:
			ckpt = tf.train.get_checkpoint_state(self.model_dir)
			if ckpt and ckpt.model_checkpoint_path:
				self.saver.restore(glob['session'], ckpt.model_checkpoint_path)
			else:
				print("...no checkpoint found...")
			print('*Predict Validation*'); sys.stdout.flush()
			val_accuracies = self.batch_predict(Data_val, n_val)
			best_validation_accuracy = val_accuracies['comp']
		else:
			best_validation_accuracy = 0

		# Train Model in Batch Mode
		for epoch in range(1, args.epochs + 1):
			print('************************')
			print('Epoch {}'.format(epoch)); sys.stdout.flush()
			total_cost = self.batch_train(Data_train, batches_train)
			print('Total Cost: {}'.format(total_cost))
			
			# Evaluate Model	
			if epoch % args.evaluation_interval == 0:
				print('*Predict Train*'); sys.stdout.flush()
				train_accuracies = self.batch_predict(Data_train, batches_train)
				print('*Predict Validation*'); sys.stdout.flush()
				val_accuracies = self.batch_predict(Data_val, batches_val)
				print('-----------------------')
				print('SUMMARY')
				print('Epoch {}'.format(epoch))
				print('Loss: {}'.format(total_cost))
				if args.bleu_score:
					print('{0:30} : {1:6f}'.format("Train BLEU", train_accuracies['bleu']))
				print('{0:30} : {1:6f}'.format("Train Accuracy", train_accuracies['acc']))
				print('{0:30} : {1:6f}'.format("Train Dialog", train_accuracies['dialog']))
				print('{0:30} : {1:6f}'.format("Train F1", train_accuracies['f1']))
				print('------------')
				if args.bleu_score:
					print('{0:30} : {1:6f}'.format("Validation BLEU", val_accuracies['bleu']))
				print('{0:30} : {1:6f}'.format("Validation Accuracy", val_accuracies['acc']))
				print('{0:30} : {1:6f}'.format("Validation Dialog", val_accuracies['dialog']))
				print('{0:30} : {1:6f}'.format("Validation F1", val_accuracies['f1']))
				print('------------')
				sys.stdout.flush()
				
				# Save best model
				val_to_compare = val_accuracies['comp']
				if val_to_compare >= best_validation_accuracy:
					best_validation_accuracy = val_to_compare
					self.saver.save(glob['session'], self.model_dir + 'model.ckpt', global_step=epoch)
					print('MODEL SAVED')
				
				# Evaluate on Test datasets
				print('*Predict Test*'); sys.stdout.flush()
				test_accuracies = self.batch_predict(Data_test, batches_test)
				if args.task_id < 6:
					print('*Predict OOV*'); sys.stdout.flush()
					test_oov_accuracies = self.batch_predict(Data_test_OOV, batches_oov)
				
				print('-----------------------')
				print('SUMMARY')
				if args.bleu_score:
					print('{0:30} : {1:6f}'.format("Test BLEU", test_accuracies['bleu']))
				print('{0:30} : {1:6f}'.format("Test Accuracy", test_accuracies['acc']))
				print('{0:30} : {1:6f}'.format("Test Dialog", test_accuracies['dialog']))
				print('{0:30} : {1:6f}'.format("Test F1", test_accuracies['f1']))
				if args.task_id < 6:
					print('------------')
					if args.bleu_score:
						print('{0:30} : {1:6f}'.format("Test OOV BLEU", test_oov_accuracies['bleu']))
					print('{0:30} : {1:6f}'.format("Test OOV Accuracy", test_oov_accuracies['acc']))
					print('{0:30} : {1:6f}'.format("Test OOV Dialog", test_oov_accuracies['dialog']))
					print('{0:30} : {1:6f}'.format("Test OOV F1", test_oov_accuracies['f1']))
				print('-----------------------')
				sys.stdout.flush()
			
	def test(self):
		'''
			Test the model
		'''
		# Look for previously saved checkpoint
		ckpt = tf.train.get_checkpoint_state(self.model_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(glob['session'], ckpt.model_checkpoint_path)
		else:
			print("...no checkpoint found...")

		if args.OOV:
			Data_test = Data(self.testOOVData, args, glob)
			n_test = len(Data_test_OOV.stories)
			print("Test OOV Size", n_test)
		else:
			Data_test = Data(self.testData, args, glob)
			n_test = len(Data_test.stories)
			print("Test Size", n_test)
		sys.stdout.flush()
		
		batches_test = create_batches(Data_test, args.batch_size)

		print('*Predict Test*'); sys.stdout.flush()
		test_accuracies = self.batch_predict(Data_test, batches_test)
		print('-----------------------')
		print('SUMMARY')
		if args.bleu_score:
			print('{0:30} : {1:6f}'.format("Test BLEU", test_accuracies['bleu']))
		print('{0:30} : {1:6f}'.format("Test Accuracy", test_accuracies['acc']))
		print('{0:30} : {1:6f}'.format("Test Dialog", test_accuracies['dialog']))
		print('{0:30} : {1:6f}'.format("Test F1", test_accuracies['f1']))
		print("------------------------")
		sys.stdout.flush()

	def batch_train(self, data, train_batches):
		'''
			Train Model for a Batch of Input Data
		'''
		batches = train_batches.copy()
		np.random.shuffle(batches)
		total_cost = 0.0	# Total Loss
		total_seq = 0.0		# Sequence Loss
		total_pgen = 0.0	# Pgen Loss

		pbar = tqdm(enumerate(batches),total=len(batches))
		for i, (start, end) in pbar:
			batch_entry = Batch(data, start, end, args, train=True)
			cost_t, seq_loss, pgen_loss = self.model.batch_fit(batch_entry)
			total_seq += seq_loss
			total_pgen += pgen_loss
			total_cost += cost_t
			pbar.set_description('TL:{:.2f}, SL:{:.2f}, PL:{:.2f}'.format(total_cost/(i+1),total_seq/(i+1),total_pgen/(i+1)))
		return total_cost

	def batch_predict(self, data, batches):
		'''
			Get Predictions for Input Data batchwise
		'''
		predictions = []

		pbar = tqdm(enumerate(batches),total=len(batches))
		for i, (start, end) in pbar:
			# Get predictions
			data_batch = Batch(data, start, end, args)
			preds = self.model.predict(data_batch)

			# Store prediction outputs
			predictions += pad_to_answer_size(list(preds), glob['candidate_sentence_size'])

		# Evaluate metrics
		return evaluate(args, glob, predictions, data)

	def close_session(self):
		glob['session'].close()

''' Main Function '''
if __name__ == '__main__': 

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

	chatbot = chatBot()
	print("CHATBOT READY"); sys.stdout.flush();
	print_params(logging, args)

	chatbot.train() if args.train else chatbot.test()

	chatbot.close_session()