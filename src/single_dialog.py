from __future__ import absolute_import
from __future__ import print_function

from data_utils import *
from sklearn import metrics
from memn2n.memn2n_dialog_generator import MemN2NGeneratorDialog
from itertools import chain
from six.moves import range, reduce
from operator import itemgetter
from data import Data, Batch
import sys
import tensorflow as tf
import numpy as np
import os
import pdb
import json

# Model Params
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 400, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 32, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 100, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_boolean('interactive', False, 'if True, interactive')
tf.flags.DEFINE_boolean('dropout', False, 'if True, uses dropout on p_gen')
tf.flags.DEFINE_boolean('word_drop', False, 'if True, uses random word dropout')
tf.flags.DEFINE_boolean("char_emb_overlap", True, 'if False, no overlap of word character tokens during embeddings')
tf.flags.DEFINE_boolean("reduce_states", False, 'if True, reduces embedding size of encoder states')
tf.flags.DEFINE_boolean("p_gen_loss", False, 'if True, uses additional p_gen loss during training')
tf.flags.DEFINE_integer("unk_size", 2, "Number of random unk words per batch")
tf.flags.DEFINE_integer("char_emb_length", 1, "Number of letters treated as an input token for character embeddings")
tf.flags.DEFINE_boolean('lba', False, 'if True, uses location based addressing')
tf.flags.DEFINE_integer("shift_size", 2, "Amount of shift allowed for Location Based Addressing")
tf.flags.DEFINE_integer("soft_weight", 2, "Weight given to softmax function")

# Model Type
tf.flags.DEFINE_boolean("char_emb", True, 'if True, uses character embeddings')
tf.flags.DEFINE_boolean('pointer', True, 'if True, uses pointer network')
tf.flags.DEFINE_boolean("hierarchy", True, "if True, uses hierarchy pointer attention")
tf.flags.DEFINE_boolean("gated", False, "if True, uses gated memory network")
tf.flags.DEFINE_boolean("word_softmax", True, "if True, uses gated memory network")
tf.flags.DEFINE_boolean("line_softmax", True, "if True, uses gated memory network")
tf.flags.DEFINE_boolean("rnn", True, "if True, uses bi-directional-rnn to encode, else Bag of Words")

# Output and Evaluation Specifications
tf.flags.DEFINE_integer("evaluation_interval", 4, "Evaluate and print results every x epochs")
tf.flags.DEFINE_boolean("bleu_score", False, 'if True, uses BLUE word score to compute best model')
tf.flags.DEFINE_boolean("new_eval", True, 'if True, uses new evaluation score')
tf.flags.DEFINE_boolean("visualize", False, "if True, uses visualize_attention tool")

# Task Type
tf.flags.DEFINE_boolean('train', False, 'if True, begin to train')
tf.flags.DEFINE_integer("task_id", 3, "bAbI task id, 1 <= id <= 8")
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
		self.model_dir = FLAGS.model_dir + "task" + str(FLAGS.task_id) + "_" + FLAGS.data_dir.split('/')[-2] + "_lr-" + str(FLAGS.learning_rate) + "_hops-" + str(FLAGS.hops) + "_emb-size-" + str(FLAGS.embedding_size) + "_sw-" + str(FLAGS.soft_weight) + "_model/"
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
		self.unk_size = FLAGS.unk_size
		self.bleu_score = FLAGS.bleu_score
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

		if self.task_id == 7:
			self.bleu_score=True

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
		# self.trainDataVocab, self.testDataVocab, self.valDataVocab = load_dialog_task(self.data_dir, self.task_id, False)
		
		# Build vocab only with modified data
		self.build_vocab(self.modData)

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
										   word_softmax=self.word_softmax, line_softmax=self.line_softmax, soft_weight=self.soft_weight)
		self.saver = tf.train.Saver(max_to_keep=4)

	def build_vocab(self, data):
		'''
			Get vocabulary from the Train data
		'''
		vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, a, start in data))
		vocab = sorted(vocab)
		# Jan 6 : UNK was missing in train
		self.word_idx = dict((c, i + 2) for i, c in enumerate(vocab))
		self.word_idx['UNK']=1
		max_story_size = max(map(len, (s for s, _, _, _ in data)))
		self.sentence_size = max(map(len, chain.from_iterable(s for s, _, _, _ in data)))
		query_size = max(map(len, (q for _, q, _, _ in data)))
		self.memory_size = min(self.memory_size, max_story_size)
		self.vocab_size = len(self.word_idx) + 1  # +1 for nil word
		self.sentence_size = max(query_size, self.sentence_size)
		print("Input Vocab Size   : ", self.vocab_size)

	def train(self):
		'''
			Train the model
		'''
		# Get Data in usable form
		Data_train = Data(self.trainData, self.word_idx, self.sentence_size, 
						  self.batch_size, self.memory_size, 
						  self.decoder_vocab_to_index, self.candidate_sentence_size, 
						  self.char_emb_length, self.char_emb_overlap)
		Data_val = Data(self.valData, self.word_idx, self.sentence_size, 
						self.batch_size, self.memory_size, 
						self.decoder_vocab_to_index, self.candidate_sentence_size, 
						self.char_emb_length, self.char_emb_overlap)
		Data_test = Data(self.testData, self.word_idx, self.sentence_size, 
						self.batch_size, self.memory_size, 
						self.decoder_vocab_to_index, self.candidate_sentence_size, 
						self.char_emb_length, self.char_emb_overlap)
		if self.task_id < 6:
			Data_test_OOV = Data(self.testOOVData, self.word_idx, self.sentence_size, 
							self.batch_size, self.memory_size, 
							self.decoder_vocab_to_index, self.candidate_sentence_size, 
							self.char_emb_length, self.char_emb_overlap)
		
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
		self.word_drop = False

		# Train Model in Batch Mode
		print('-----------------------')
		for t in range(1, self.epochs + 1):
			total_cost = self.batch_train(Data_train, batches)
			print('Epoch', t, ' Total Cost:', total_cost)
			
			# Evaluate Model
			if t % self.evaluation_interval == 0:
				train_accuracies = self.batch_predict(Data_train, n_train)
				val_accuracies = self.batch_predict(Data_val, n_val)
				print('-----------------------')
				print('Epoch', t)
				print('Total Cost:', total_cost)
				if self.pointer:
					if self.bleu_score:
						print("Train BLEU      : ", train_accuracies[2], train_accuracies[3])
						print("Validation BLEU : ", val_accuracies[2], val_accuracies[3])
					else:
						print("Train Accuracy (Substring / Actual)      : ", train_accuracies[1][0], train_accuracies[1][1])
						print("Train Accuracy + Attention               : ", train_accuracies[0][0], train_accuracies[0][1])
						print("Validation Accuracy (Substring / Actual) : ", val_accuracies[1][0], val_accuracies[1][1])
						print("Validation Accuracy + Attention          : ", val_accuracies[0][0], val_accuracies[0][1])
				else:
					if self.bleu_score:
						print("Train BLEU      : ", train_accuracies[1])
						print("Validation BLEU : ", val_accuracies[1])
					else:
						print("Train Accuracy (Substring / Actual)      : ", train_accuracies[0][0], train_accuracies[0][1])
						print("Validation Accuracy (Substring / Actual) : ", val_accuracies[0][0], val_accuracies[0][1])
				print('-----------------------')
				sys.stdout.flush()
				
				# Save best model

				val_score = val_accuracies[0][1]
				if self.bleu_score:
					idx = 1
					if self.pointer:
						idx = 2
					val_score = val_accuracies[idx]
					
				if val_score >= best_validation_accuracy:
					model_count += 1
					best_validation_accuracy = val_score
					self.saver.save(self.sess, self.model_dir + 'model.ckpt', global_step=t)
					test_accuracies = self.batch_predict(Data_test, n_test)
					if self.task_id < 6:
						test_oov_accuracies = self.batch_predict(Data_test_OOV, n_oov)
					if self.pointer:
						
						if self.bleu_score:
							print("Test BLEU       : ", test_accuracies[2], test_accuracies[3])
						else:
							print("Test Accuracy (Substring / Actual)       : ", test_accuracies[1][0], test_accuracies[1][1])
							print("Test Accuracy + Attention                : ", test_accuracies[0][0], test_accuracies[0][1])
							
						if self.task_id < 6:
							if self.bleu_score:
								print("Test OOV BLEU   : ", test_oov_accuracies[2], test_oov_accuracies[3])
							else:
								print("Test OOV Accuracy (Substring / Actual)   : ", test_oov_accuracies[1][0], test_oov_accuracies[1][1])
								print("Test OOV Accuracy + Attention            : ", test_oov_accuracies[0][0], test_oov_accuracies[0][1])
							
					else:
						
						if self.bleu_score:
							print("Test BLEU       : ", test_accuracies[1])
						else:
							print("Test Accuracy (Substring / Actual)       : ", test_accuracies[0][0], test_accuracies[0][1])
							
						if self.task_id < 6:
							if self.bleu_score:
								print("Test OOV BLEU   : ", test_accuracies[1])
							else:
								print("Test OOV Accuracy (Substring / Actual)   : ", test_oov_accuracies[0][0], test_oov_accuracies[0][1])
				
					print('-----------------------')
				
				sys.stdout.flush()

				if model_count >= 10 and self.word_drop_flag:
					self.word_drop = True

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
				Data_test = Data(self.testData, self.word_idx, self.sentence_size, 
							 self.batch_size, self.memory_size, 
							 self.decoder_vocab_to_index, self.candidate_sentence_size, 
							 self.char_emb_length, self.char_emb_overlap)
			else:
				Data_test = Data(self.testOOVData, self.word_idx, self.sentence_size, 
							 self.batch_size, self.memory_size, 
							 self.decoder_vocab_to_index, self.candidate_sentence_size, 
							 self.char_emb_length, self.char_emb_overlap)
			n_test = len(Data_test.stories)
			print("Test Size", n_test)
			test_accuracies = self.batch_predict(Data_test, n_test)

			print("Test Size      : ", n_test)
			if self.pointer:
				print("Test Accuracy (Substring / Actual) : ", test_accuracies[1][0], test_accuracies[1][1])
				print("Test Accuracy + Attention : ", test_accuracies[0][0], test_accuracies[0][1])
				idx1 = 2; idx2 = 3   
				if self.bleu_score:
					print('Test Bleu Score:', test_accuracies[idx1], test_accuracies[idx2])
					idx += 2; idx2 += 2
				if self.new_eval and (self.task_id==3 or self.task_id==5):
					print('Restaurant Recommendation Accuracy : ', test_accuracies[idx1][0], test_accuracies[idx2][0])
					print('Restaurant Recommendation from DB Accuracy : ', test_accuracies[idx1][1], test_accuracies[idx2][1])
			else:
				print("Test Accuracy (Substring / Actual) : ", test_accuracies[0][0], test_accuracies[0][1])
				idx = 1
				if self.bleu_score:
					print('Test Bleu Score:', test_accuracies[idx])
					idx += 1
				if self.new_eval and (self.task_id==3 or self.task_id==5):
					print('Restaurant Recommendation Accuracy : ', test_accuracies[idx][0])
					print('Restaurant Recommendation from DB Accuracy : ', test_accuracies[idx][1])
			print("------------------------")

	def interactive(self):
		print("To Be Implemented")

	def batch_train(self, data, batches):
		'''
			Train Model for a Batch of Input Data
		'''
		np.random.shuffle(batches)
		total_cost = 0.0
		for i, (start, end) in enumerate(batches):
			cost_t, logits = self.model.batch_fit(Batch(data, start, end, self.unk_size, self.word_drop))
			total_cost += cost_t
		return total_cost

	def batch_predict(self, data, n):
		'''
			Get Predictions for Input Data batchwise
		'''
		if self.pointer:
			old_preds = []
			new_preds = []
		else:
			preds = []
		count = 0
		for start in range(0, n, self.batch_size):
			end = start + self.batch_size
			count += 1
			data_batch = Batch(data, start, end)
			if count >= n / self.batch_size:
				break
			if self.pointer:
				old_pred, new_pred, hier, line, word, p_gens = self.model.predict(data_batch)
				old_preds += pad_to_answer_size(list(old_pred), self.candidate_sentence_size)
				new_preds += pad_to_answer_size(list(new_pred), self.candidate_sentence_size)
				if self.visualize and count == 31:
					visualize_attention(data_batch, hier, line, word, p_gens, count, self.hierarchy)
			else:
				pred = self.model.predict(data_batch)
				preds += pad_to_answer_size(list(pred), self.candidate_sentence_size)
		if self.pointer:
			output = [substring_accuracy_score(new_preds, data.answers,word_map=self.decoder_index_to_vocab,isTrain=self.is_train), substring_accuracy_score(old_preds, data.answers)]
			if self.bleu_score:
				output += [bleu_accuracy_score(old_preds, data.answers, word_map=self.decoder_index_to_vocab), bleu_accuracy_score(new_preds, data.answers,word_map=self.decoder_index_to_vocab,isTrain=self.is_train)]
			if self.new_eval and (self.task_id==3 or self.task_id==5):
				output += [new_eval_score(old_preds, data.answers, data.db_values_set, word_map=self.decoder_index_to_vocab), new_eval_score(new_preds, data.answers, data.db_values_set, word_map=self.decoder_index_to_vocab)]
		else:
			output = [substring_accuracy_score(preds, data.answers,word_map=self.decoder_index_to_vocab,isTrain=self.is_train)]
			if self.bleu_score:
				output += [bleu_accuracy_score(preds, data.answers,word_map=self.decoder_index_to_vocab,isTrain=self.is_train)]
			if self.new_eval and (self.task_id==3 or self.task_id==5):
				output += [new_eval_score(preds, data.answers, data.db_values_set, word_map=self.decoder_index_to_vocab)]
		return output

	def close_session(self):
		self.sess.close()

''' Main Function '''
if __name__ == '__main__': 

	chatbot = chatBot()
	
	print("CHATBOT READY"); sys.stdout.flush();

	if FLAGS.train: chatbot.train()
	else: chatbot.test()

	chatbot.close_session()
