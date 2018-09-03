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
from itertools import chain
from memn2n.memn2n_dialog_generator import MemN2NGeneratorDialog
from operator import itemgetter
from six.moves import range, reduce
from sklearn import metrics
from tqdm import tqdm

# Model Params
tf.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("word_drop_prob", 0.0, "value to set, if word_drop is set to True")
tf.flags.DEFINE_float("p_gen_loss_weight", 0.75, 'relative weight to p_gen loss, > 1 gives more weight to p_gen loss')
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 4, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 4000, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 128, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 100, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_integer("unk_size", 2, "Number of random unk words per batch")
tf.flags.DEFINE_integer("char_emb_length", 1, "Number of letters treated as an input token for character embeddings")
tf.flags.DEFINE_integer("shift_size", 2, "Amount of shift allowed for Location Based Addressing")
tf.flags.DEFINE_integer("soft_weight", 1, "Weight given to softmax function")
tf.flags.DEFINE_integer("eos_weight", 2, "Weight given to eos error")
tf.flags.DEFINE_boolean('interactive', False, 'if True, interactive')
tf.flags.DEFINE_boolean('dropout', False, 'if True, uses dropout on p_gen')
tf.flags.DEFINE_boolean('word_drop', True, 'if True, drop db words in story')
tf.flags.DEFINE_boolean("char_emb_overlap", True, 'if False, no overlap of word character tokens during embeddings')
tf.flags.DEFINE_boolean("reduce_states", False, 'if True, reduces embedding size of encoder states')
tf.flags.DEFINE_boolean("p_gen_loss", True, 'if True, uses additional p_gen loss during training')
tf.flags.DEFINE_boolean('lba', False, 'if True, uses location based addressing')

# Model Type
tf.flags.DEFINE_boolean("char_emb", False, 'if True, uses character embeddings')
tf.flags.DEFINE_boolean('pointer', True, 'if True, uses pointer network')
tf.flags.DEFINE_boolean("hierarchy", True, "if True, uses hierarchy pointer attention")
tf.flags.DEFINE_boolean("gated", False, "if True, uses gated memory network")
tf.flags.DEFINE_boolean("word_softmax", True, "if True, uses gated memory network")
tf.flags.DEFINE_boolean("line_softmax", True, "if True, uses gated memory network")
tf.flags.DEFINE_boolean("rnn", True, "if True, uses bi-directional-rnn to encode, else Bag of Words")
tf.flags.DEFINE_boolean("position_emb", False, "if True, uses temporal embedding for stories")
# not implemented yet
tf.flags.DEFINE_boolean("copy_first", False, "copy by default, if sentinal is selected, then generate")

# Output and Evaluation Specifications
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.flags.DEFINE_boolean("bleu_score", True, 'if True, uses BLUE word score to compute best model')
tf.flags.DEFINE_boolean("new_eval", False, 'if True, uses new evaluation score')
tf.flags.DEFINE_boolean("visualize", False, "if True, uses visualize_attention tool")

# Task Type
tf.flags.DEFINE_integer("task_id", 6, "bAbI task id, 1 <= id <= 8")
tf.flags.DEFINE_boolean('train', False, 'if True, begin to train')
tf.flags.DEFINE_boolean('OOV', False, 'if True, use OOV test set')

# File Locations
tf.flags.DEFINE_string("data_dir", "../data/dialog-bAbI-tasks/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("logs_dir", "logs/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("model_dir", "model/", "Directory containing memn2n model checkpoints")
tf.flags.DEFINE_string("vocab_ext", "trn", "Data Set used to build the decode vocabulary")

FLAGS = tf.flags.FLAGS

class chatBot(object):

	def __init__(self):
		# Define Parameters of ChatBot
		self.data_dir = FLAGS.data_dir
		self.task_id = FLAGS.task_id
		self.model_dir = FLAGS.model_dir + "task" + str(FLAGS.task_id) + "_" + FLAGS.data_dir.split('/')[-2] + "_lr-" + str(FLAGS.learning_rate) + "_hops-" + str(FLAGS.hops) + "_emb-size-" + str(FLAGS.embedding_size) + "_sw-" + str(FLAGS.soft_weight) + "_wd-" + str(FLAGS.word_drop_prob) + "_pw-" + str(FLAGS.p_gen_loss_weight) + "_model/"
		self.logs_dir = FLAGS.logs_dir
		self.isInteractive = FLAGS.interactive
		self.OOV = FLAGS.OOV
		self.memory_size = FLAGS.memory_size
		self.random_state = FLAGS.random_state
		self.batch_size = FLAGS.batch_size
		self.learning_rate = FLAGS.learning_rate
		self.epsilon = FLAGS.epsilon
		self.max_grad_norm = FLAGS.max_grad_norm
		self.evaluation_interval = FLAGS.evaluation_interval
		self.hops = FLAGS.hops
		self.epochs = FLAGS.epochs
		self.embedding_size = FLAGS.embedding_size
		self.pointer = FLAGS.pointer
		self.dropout = FLAGS.dropout
		self.word_drop_flag = FLAGS.word_drop
		self.word_drop = FLAGS.word_drop
		self.unk_size = FLAGS.unk_size
		self.is_train = FLAGS.train
		self.char_emb = FLAGS.char_emb
		self.char_emb_length = FLAGS.char_emb_length
		self.char_emb_overlap = FLAGS.char_emb_overlap
		self.reduce_states = FLAGS.reduce_states
		self.p_gen_loss = FLAGS.p_gen_loss
		self.new_eval = FLAGS.new_eval
		self.gated = FLAGS.gated
		self.hierarchy = FLAGS.hierarchy
		self.visualize = FLAGS.visualize
		self.rnn = FLAGS.rnn
		self.shift_size = FLAGS.shift_size
		self.lba = FLAGS.lba
		self.vocab_ext = FLAGS.vocab_ext
		self.word_softmax = FLAGS.word_softmax
		self.line_softmax = FLAGS.line_softmax
		self.soft_weight = FLAGS.soft_weight
		self.position_emb = FLAGS.position_emb
		self.copy_first = FLAGS.copy_first
		self.word_drop_prob = FLAGS.word_drop_prob
		self.p_gen_loss_weight = FLAGS.p_gen_loss_weight
		self.eos_weight = FLAGS.eos_weight

		if self.task_id >= 6:
			self.bleu_score=True
		else:
			self.bleu_score = FLAGS.bleu_score

		# Create Model Store Directory
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		
		# Print Task Information
		if self.OOV:
			print("Task ", self.task_id, " learning rate : ", self.learning_rate, " with OOV")
		else:
			print("Task ", self.task_id, " learning rate : ", self.learning_rate)
		
		# Load Decoder Vocabulary
		self.decoder_vocab_to_index, self.decoder_index_to_vocab, self.candidate_sentence_size = get_decoder_vocab(self.data_dir, self.task_id, self.vocab_ext)
		print("Decoder Vocab Size : ", len(self.decoder_vocab_to_index))
		sys.stdout.flush()

		# Retreive Task Data
		self.trainData, self.testData, self.valData, self.testOOVData, self.modData = load_dialog_task(self.data_dir, self.task_id, self.vocab_ext)
		
		# Build vocab only with modified data
		self.build_vocab(self.trainData)

		# Define MemN2N + Generator Model
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
		self.sess = tf.Session()
		self.model = MemN2NGeneratorDialog(self.batch_size, self.vocab_size, self.sentence_size, 
										   self.embedding_size, self.decoder_vocab_to_index, self.candidate_sentence_size, 
										   session=self.sess, hops=self.hops, max_grad_norm=self.max_grad_norm, 
										   optimizer=self.optimizer, task_id=self.task_id, pointer=self.pointer,
										   dropout=self.dropout, char_emb=self.char_emb, rnn=self.rnn,
										   reduce_states=self.reduce_states, char_emb_size=256**self.char_emb_length, p_gen_loss=self.p_gen_loss,
										   gated=self.gated, hierarchy=self.hierarchy, shift_size=self.shift_size, lba=self.lba, 
										   word_softmax=self.word_softmax, line_softmax=self.line_softmax, soft_weight=self.soft_weight,
										   position_emb=self.position_emb, p_gen_loss_weight=self.p_gen_loss_weight, eos_weight=self.eos_weight)
		self.saver = tf.train.Saver(max_to_keep=4)

	def build_vocab(self, data):
		'''
			Get vocabulary from the Train data
		'''
		vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, a, start in data))
		vocab = sorted(vocab)
		# Jan 6 : UNK was missing in train
		self.word_idx = dict((c, i + 2) for i, c in enumerate(vocab))
		self.word_idx['']=0
		self.word_idx['UNK']=1
		max_story_size = max(map(len, (s for s, _, _, _ in data)))
		self.sentence_size = max(map(len, chain.from_iterable(s for s, _, _, _ in data)))
		query_size = max(map(len, (q for _, q, _, _ in data)))
		self.memory_size = min(self.memory_size, max_story_size)
		self.vocab_size = len(self.word_idx) + 1  # +1 for nil word
		self.sentence_size = max(query_size, self.sentence_size)
		self.idx_word = {v: k for k, v in self.word_idx.items()}
		print("Input Vocab Size : {}".format(self.vocab_size))

	def print_params(self):
		'''
			Print important model parameters
		'''
		logging.info('[{}] : {}'.format('model_dir', self.model_dir))
		logging.info('[{}] : {}'.format('task_id', self.task_id))
		logging.info('[{}] : {}'.format('data_dir', self.data_dir.split('/')[-2]))
		logging.info('[{}] : {}'.format('learning_rate', self.learning_rate))
		logging.info('[{}] : {}'.format('hops', self.hops))
		logging.info('[{}] : {}'.format('embedding_size', self.embedding_size))
		logging.info('[{}] : {}'.format('soft_weight', self.soft_weight))
		logging.info('[{}] : {}'.format('word_drop_prob', self.word_drop_prob))
		logging.info('[{}] : {}'.format('p_gen_loss_weight', self.p_gen_loss_weight))

	def train(self):
		'''
			Train the model
		'''
		# Get Data in usable form
		Data_train = Data(self.trainData, self.word_idx, self.idx_word, self.sentence_size, 
						  self.batch_size, self.memory_size,
						  self.decoder_vocab_to_index, self.candidate_sentence_size, 
						  self.char_emb_length, self.char_emb_overlap, self.copy_first)
		Data_val = Data(self.valData, self.word_idx, self.idx_word, self.sentence_size, 
						self.batch_size, self.memory_size,
						self.decoder_vocab_to_index, self.candidate_sentence_size, 
						self.char_emb_length, self.char_emb_overlap, self.copy_first)
		Data_test = Data(self.testData, self.word_idx, self.idx_word, self.sentence_size, 
						self.batch_size, self.memory_size, 
						self.decoder_vocab_to_index, self.candidate_sentence_size, 
						self.char_emb_length, self.char_emb_overlap, self.copy_first)
		if self.task_id < 6:
			Data_test_OOV = Data(self.testOOVData, self.word_idx, self.idx_word, self.sentence_size, 
							self.batch_size, self.memory_size,
							self.decoder_vocab_to_index, self.candidate_sentence_size, 
							self.char_emb_length, self.char_emb_overlap, self.copy_first)
		
		# Create Batches
		n_train = len(Data_train.stories)
		n_val = len(Data_val.stories)
		n_test = len(Data_test.stories)
		print("Training Size", n_train)
		print("Validation Size", n_val)
		print("Test Size", n_test)
		
		if self.task_id < 6:
			n_oov = len(Data_test_OOV.stories)
			print("Test OOV Size", n_oov)
		
		sys.stdout.flush()
		tf.set_random_seed(self.random_state)
		batches = zip(range(0, n_train - self.batch_size, self.batch_size),
					  range(self.batch_size, n_train, self.batch_size))
		batches = [(start, end) for start, end in batches]
		# fix to include last batch
		if batches[-1][1] < n_train:
			batches.append((batches[-1][1], n_train))

		best_validation_accuracy = 0
		model_count = 0

		# (start, end) = batches[0]
		# batch = Batch(Data_val, start, end, self.unk_size, False, 0)
		# print(batch.stories[8])
		# for story in batch.stories[8]:
		# 	print(len(story))
		# print(type(batch.stories[8]))
		# print(batch.oov_ids[8])
		# print(type(batch.oov_ids[8]))
		# print(batch.readable_stories[8])
		# print(type(batch.readable_stories[8]))
		# print()
		# print()
		# print(batch.stories[9])
		# print(type(batch.stories[9]))
		# print(batch.oov_ids[9])
		# print(type(batch.oov_ids[9]))
		# print(batch.readable_stories[9])
		# print(type(batch.readable_stories[9]))
		# sys.exit()
		
		# Train Model in Batch Mode
		for t in range(1, self.epochs + 1):
			print('************************')
			print('Epoch', t); sys.stdout.flush()
			total_cost = self.batch_train(Data_train, batches, t)
			print('\nTotal Cost:', total_cost); sys.stdout.flush()
			
			# Evaluate Model	
			if t % self.evaluation_interval == 0:
				print('Predict Train'); sys.stdout.flush()
				train_accuracies = self.batch_predict(Data_train, n_train)
				print('\nPredict Validation'); sys.stdout.flush()
				val_accuracies = self.batch_predict(Data_val, n_val)
				print('\n-----------------------')
				print('SUMMARY')
				print('Epoch', t)
				print('Loss:', total_cost)
				if self.bleu_score:
					print("Train BLEU		: ", train_accuracies['bleu'])
				print("Train Accuracy		: ", train_accuracies['acc'])
				print("Train Dialog		: ", train_accuracies['dialog'])
				print("Train F1		: ", train_accuracies['f1'])
				print('------------')
				if self.bleu_score:
					print("Validation BLEU		: ", val_accuracies['bleu'])
				print("Validation Accuracy	: ", val_accuracies['acc'])
				print("Validation Dialog	: ", val_accuracies['dialog'])
				print("Validation F1		: ", val_accuracies['f1'])
				print('-----------------------')
				sys.stdout.flush()
				
				# Save best model
				val_to_compare = val_accuracies['acc']
				if self.bleu_score:
					val_to_compare = val_accuracies['bleu']
					
				if val_to_compare >= best_validation_accuracy:
					model_count += 1
					best_validation_accuracy = val_to_compare
					self.saver.save(self.sess, self.model_dir + 'model.ckpt', global_step=t)
				
				print('Predict Test'); sys.stdout.flush()
				test_accuracies = self.batch_predict(Data_test, n_test)
				if self.task_id < 6:
					print('\nPredict OOV'); sys.stdout.flush()
					test_oov_accuracies = self.batch_predict(Data_test_OOV, n_oov)
				
				print('\n-----------------------')
				print('SUMMARY')
				if self.bleu_score:
					print("Test BLEU		: ", test_accuracies['bleu'])
				print("Test Accuracy		: ", test_accuracies['acc'])
				print("Test Dialog		: ", test_accuracies['dialog'])
				print("Test F1			: ", test_accuracies['f1'])
				if self.task_id < 6:
					print('------------')
					if self.bleu_score:
						print("Test OOV BLEU		: ", test_oov_accuracies['bleu'])
					print("Test OOV Accuracy	: ", test_oov_accuracies['acc'])
					print("Test OOV Dialog		: ", test_oov_accuracies['dialog'])
					print("Test OOV F1		: ", test_oov_accuracies['f1'])
				print('-----------------------')
				sys.stdout.flush()
			
	def test(self):
		'''
			Test the model
		'''
		# Look for previously saved checkpoint
		ckpt = tf.train.get_checkpoint_state(self.model_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			print("...no checkpoint found...")

		if self.isInteractive:
			self.interactive()
		else:
			if not self.OOV:
				Data_test = Data(self.testData, self.word_idx, self.idx_word, self.sentence_size, 
							 self.batch_size, self.memory_size, 
							 self.decoder_vocab_to_index, self.candidate_sentence_size, 
							 self.char_emb_length, self.char_emb_overlap, self.copy_first)
			else:
				Data_test = Data(self.testOOVData, self.word_idx, self.idx_word, self.sentence_size, 
							 self.batch_size, self.memory_size,
							 self.decoder_vocab_to_index, self.candidate_sentence_size, 
							 self.char_emb_length, self.char_emb_overlap, self.copy_first)
			n_test = len(Data_test.stories)
			print("Test Size", n_test)
			
			print('Predict Test'); sys.stdout.flush()
			test_accuracies = self.batch_predict(Data_test, n_test)
			print('\n-----------------------')
			print('SUMMARY')
			if self.bleu_score:
				print("Test BLEU		: ", test_accuracies['bleu'])
			print("Test Accuracy		: ", test_accuracies['acc'])
			print("Test Dialog		: ", test_accuracies['dialog'])
			print("Test F1			: ", test_accuracies['f1'])
			print("------------------------")
			sys.stdout.flush()

	def interactive(self):
		print("To Be Implemented")

	def batch_train(self, data, batches, t):
		'''
			Train Model for a Batch of Input Data
		'''
		EPOCH_TO_PRINT = 5000
		np.random.shuffle(batches)
		total_cost = 0.0
		total_seq = 0.0
		total_pgen = 0.0

		one_total = 0
		one_prob_total = 0.0

		zero_total = 0
		zero_prob_total = 0.0

		for i in tqdm(range(0, len(batches))):
			(start, end) = batches[i]
			#print(start, end)
			if self.pointer:
				batch_entry = Batch(data, start, end, self.unk_size, self.word_drop, self.word_drop_prob)
				cost_t, logits, seq_loss, pgen_loss, pgens = self.model.batch_fit(batch_entry)

				if t == EPOCH_TO_PRINT+1:
					sys.exit()

				answers = batch_entry.answers
				index = 0
				for answer in answers:
					if len(pgens) <= index or len(batch_entry._intersection_set) <= index:
						continue
					pgen = pgens[index]
					gt = batch_entry._intersection_set[index]
					word_index = 0
					for w in answer:
						if len(gt) <= word_index or len(pgen) <= word_index:
							continue
						if w in self.decoder_index_to_vocab:
							if self.decoder_index_to_vocab[w] != "PAD":
								if gt[word_index] == 1:
									one_total += 1
									one_prob_total += pgen[word_index][0]
								else:
									zero_total += 1
									zero_prob_total += pgen[word_index][0]
						else:
							if gt[word_index] == 1:
								one_total += 1
								one_prob_total += pgen[word_index][0]
							else:
								zero_total += 1
								zero_prob_total += pgen[word_index][0]
						if t == EPOCH_TO_PRINT:
							print(self.decoder_index_to_vocab[w], gt[word_index], pgen[word_index][0])
						word_index+=1
					index+=1
					if t == EPOCH_TO_PRINT:
						print("")
			else:
				cost_t, logits, seq_loss, pgen_loss = self.model.batch_fit(Batch(data, start, end, self.unk_size, self.word_drop, self.word_drop_prob))
			total_seq += seq_loss
			total_pgen += pgen_loss
			#print("(", start, end, ")", cost_t, seq_loss, pgen_loss)
			total_cost += cost_t
		# print('Epoch', t, ' Total Cost:',total_cost, '(', total_seq, '+',total_pgen, ')')
		# print('\t',zero_total, zero_prob_total, one_total, one_prob_total)
		
		return total_cost

	def batch_predict(self, data, n):
		'''
			Get Predictions for Input Data batchwise
		'''
		if self.pointer:
			preds = []
			d_ids = []
			entities = []
			oov_words = []
		else:
			preds = []
		count = 0

		batches = zip(range(0, n - self.batch_size, self.batch_size),
					  range(self.batch_size, n, self.batch_size))
		batches = [(start, end) for start, end in batches]
		# fix to include last batch
		if batches[-1][1] < n:
			batches.append((batches[-1][1], n))

		# for start in range(0, n, self.batch_size):
		for i in tqdm(range(0, len(batches))):
			(start, end) = batches[i]
			# end = start + self.batch_size
			count += 1
			
			data_batch = Batch(data, start, end, self.unk_size, False, 0)
			# if count >= n / self.batch_size:
			# 	break
			if self.pointer:
				old_pred, new_pred, hier, line, word, p_gens = self.model.predict(data_batch)
				preds += pad_to_answer_size(list(new_pred), self.candidate_sentence_size)
				d_ids += data_batch._dialog_ids
				entities += data_batch._entities
				oov_words += data_batch._oov_words
				if self.visualize: # and count == 31:
					print(count)
					visualize_attention(data_batch, hier, line, word, p_gens, count, self.hierarchy)
			else:
				pred = self.model.predict(data_batch)
				preds += pad_to_answer_size(list(pred), self.candidate_sentence_size)
		output = [substring_accuracy_score(preds, data.answers, d_ids, entities, oov_words, data.entity_words, word_map=self.decoder_index_to_vocab, isTrain=self.is_train)]
		if self.bleu_score:
			output += [bleu_accuracy_score(preds, data.answers, word_map=self.decoder_index_to_vocab,isTrain=self.is_train)]
		output = split_output(output)
		return output

	def close_session(self):
		self.sess.close()

''' Main Function '''
if __name__ == '__main__': 

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

	chatbot = chatBot()
	
	print("CHATBOT READY"); sys.stdout.flush();
	chatbot.print_params()

	if FLAGS.train: 
		chatbot.train()
	else: 
		chatbot.test()

	chatbot.close_session()