from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import numpy as np
from six.moves import range
from datetime import datetime
from tensorflow.python.ops import rnn
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from memn2n.dynamic_decoder import *
from memn2n.attention_wrapper import *

###################################################################################################
#########                                  Helper Functions                              ##########
###################################################################################################

def zero_nil_slot(t, name=None):
	"""
	Overwrites the nil_slot (first row) of the input Tensor with zeros.

	The nil_slot is a dummy slot and should not be trained and influence
	the training algorithm.
	"""
	with tf.name_scope(name, "zero_nil_slot", [t]) as name:
		t = tf.convert_to_tensor(t, name="t")
		s = tf.shape(t)[1]
		z = tf.zeros(tf.stack([1, s]))
		return tf.concat([z, tf.slice(t, [1, 0], [-1, -1])], 0, name=name)

###################################################################################################
#########                                     Model Class                                ##########
###################################################################################################

class MemN2NGeneratorDialog(object):
	"""End-To-End Memory Network with a generative decoder."""

	def __init__(self, args, glob):

		# Initialize Model Variables
		self._batch_size = args.batch_size
		self._candidate_sentence_size = glob['candidate_sentence_size']
		self._debug = args.debug
		self._decode_idx = glob['decode_idx']
		self._embedding_size = args.embedding_size
		self._hierarchy = args.hierarchy
		self._hops = args.hops
		self._init = tf.random_normal_initializer(stddev=0.1)
		self._max_grad_norm = args.max_grad_norm
		self._name = 'MemN2N'
		self._opt = glob['optimizer']
		self._p_gen_loss = args.p_gen_loss
		self._p_gen_loss_weight = args.p_gen_loss_weight
		self._rnn = args.rnn
		self._sentence_size = glob['sentence_size']
		self._soft_weight = args.soft_weight
		self._task_id = args.task_id
		self._vocab_size = glob['vocab_size']

		# Add unk and eos
		self.UNK = self._decode_idx["UNK"]
		self.EOS = self._decode_idx["EOS"]
		self.GO_SYMBOL = self._decode_idx["GO_SYMBOL"]

		self._decoder_vocab_size = len(self._decode_idx)

		self._build_inputs()
		self._build_vars()

		## Encoding ##
		encoder_states, line_memory, word_memory = self._encoder(self._stories, self._queries)
		
		## Training ##
		self.loss_op = self._decoder_train(encoder_states, line_memory, word_memory)

		## Predicting ##
		self.predict_op = self._decoder_runtime(encoder_states, line_memory, word_memory)

		# gradient pipeline
		grads_and_vars = self._opt.compute_gradients(self.loss_op[0])
		grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars if g != None]
		nil_grads_and_vars = [(zero_nil_slot(g), v) if v.name in self._nil_vars else (g, v) for g, v, in grads_and_vars]
		self.train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

		init_op = tf.global_variables_initializer()
		self._sess = glob['session']
		self._sess.run(init_op)

	def _build_inputs(self):
		'''
			Define Input Variables to be given to the model
		'''
		## Encode Ids ##
		self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
		self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
		self._answers = tf.placeholder(tf.int32, [None, self._candidate_sentence_size], name="answers")
		
		## Sizes ##
		self._sentence_sizes = tf.placeholder(tf.int32, [None, None], name="sentence_sizes")
		self._query_sizes = tf.placeholder(tf.int32, [None, 1], name="query_sizes")
		self._answer_sizes = tf.placeholder(tf.int32, [None, 1], name="answer_sizes")

		## OOV Helpers ##
		self._oov_ids = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="oov_ids")
		self._oov_sizes = tf.placeholder(tf.int32, [None], name="oov_sizes")

		## Train Helpers ###
		self._intersection_mask = tf.placeholder(tf.float32, [None, self._candidate_sentence_size], name="intersection_mask")
		self._answers_emb_lookup = tf.placeholder(tf.int32, [None, self._candidate_sentence_size], name="answers_emb")
		self._keep_prob = tf.placeholder(tf.float32)

	def _build_vars(self):
		'''
			Define Model specific variables used to train and fit and test the model
		'''
		with tf.variable_scope(self._name):
			nil_word_slot = tf.zeros([1, self._embedding_size])

			# Initialize Embedding for Encoder
			A = tf.concat([nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size])], 0)
			self.A = tf.Variable(A, name="A")
			
			# Initialize Embedding for Response-Decoder
			C = tf.concat([nil_word_slot, self._init([self._decoder_vocab_size, self._embedding_size])], 0)
			self.C = tf.Variable(C, name="C")

			# Hop Context Vector to Output Query 
			self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")

			with tf.variable_scope("encoder"):
				self.encoder_fwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)
				self.encoder_bwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)

			with tf.variable_scope('decoder'):
				self.decoder_cell = tf.contrib.rnn.GRUCell(self._embedding_size)
				self.projection_layer = layers_core.Dense(self._decoder_vocab_size, use_bias=False)

			with tf.variable_scope('reduce_bow'):
				# Define weights and biases to reduce the cell and reduce the state
				self.w_reduce_bow = tf.Variable(self._init([self._embedding_size * 2, self._embedding_size]), name="w_reduce_bow")
				self.bias_reduce_bow = tf.Variable(self._init([self._embedding_size]), name="bias_reduce_bow")

		self._nil_vars = set([self.A.name])

	def _reduce_to_bow(self, emb):
		with tf.variable_scope('reduce_bow'):

			# Apply linear layer
			old_c = tf.reshape(emb, [-1, self._embedding_size * 2])
			new_c = tf.nn.relu(tf.matmul(old_c, self.w_reduce_bow) + self.bias_reduce_bow) # Get new cell from old cell
			return new_c # Return new cell state

	###################################################################################################
	#########                                  	  Encoder                                    ##########
	###################################################################################################

	def _encoder(self, stories, queries):
		'''
			Arguments:
				stories 	-	batch_size x memory_size x sentence_size
				queries 	-	batch_size x sentence_size
			Outputs:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	-	batch_size x memory_size x sentence_size x embedding_size

		'''
		with tf.variable_scope(self._name):

			### Set Variables ###
			self._batch_size = tf.shape(stories)[0]
			self._memory_size = tf.shape(stories)[1]

			### Transform Queries ###
			# query_emb : batch_size x sentence_size x embedding_size
			query_emb = tf.nn.embedding_lookup(self.A, queries)

			if self._rnn:
				query_sizes = tf.reshape(self._query_sizes, [-1])
				with tf.variable_scope("encoder"):
					(outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(self.encoder_fwd, self.encoder_bwd, query_emb, sequence_length=query_sizes, dtype=tf.float32)
				(f_state, b_state) = output_states
				u_0 = tf.concat(axis=1, values=[f_state, b_state])
			else:
				u_0 = tf.reduce_sum(query_emb, 1)
			# u_0 : batch_size x embedding_size
			u = [u_0]
			
			### Transform Stories ###
			# memory_word_emb : batch_size x memory_size x sentence_size x embedding_size
			memory_word_emb = tf.nn.embedding_lookup(self.A, stories)
			memory_emb = tf.reshape(memory_word_emb, [-1, self._sentence_size, self._embedding_size])

			if self._rnn:
				sentence_sizes = tf.reshape(self._sentence_sizes, [-1])
				with tf.variable_scope("encoder", reuse=True):
					(outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(self.encoder_fwd, self.encoder_bwd, memory_emb, sequence_length=sentence_sizes, dtype=tf.float32)
				(f_state, b_state) = output_states
				
				line_memory = tf.concat(axis=1, values=[f_state, b_state])
				# line_memory : batch_size x memory_size x embedding_size
				line_memory = tf.reshape(line_memory, [self._batch_size, self._memory_size, self._embedding_size])
			else:
				memory_emb = tf.reshape(memory_emb, [self._batch_size, self._memory_size, self._sentence_size, self._embedding_size])
				line_memory = tf.reduce_sum(memory_emb, 2)
			
			if self._rnn:
				(f_states, b_states) = outputs
				word_memory = tf.concat(axis=2, values=[f_states, b_states]) 
				word_memory = tf.reshape(word_memory, [self._batch_size, self._memory_size, self._sentence_size, self._embedding_size])
			else:
				word_memory = memory_emb

			### Implement Hop Network ###
			for hop_index in range(self._hops):
				
				# hack to get around no reduce_dot
				u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
				dotted = tf.reduce_sum(line_memory * u_temp, 2)

				# Calculate probabilities
				probs = tf.nn.softmax(dotted)
				probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
				c_temp = tf.transpose(line_memory, [0, 2, 1])
				o_k = tf.reduce_sum(c_temp * probs_temp, 2)
				u_k = tf.matmul(u[-1], self.H) + o_k
				u.append(u_k)
			
			return u_k, line_memory, word_memory

	###################################################################################################
	#########                                  	 Decoders                                    ##########
	###################################################################################################

	def _get_decoder(self, encoder_states, line_memory, word_memory, helper, batch_size):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
		'''
		with tf.variable_scope(self._name):
			with tf.variable_scope('decoder'):
				# make the shape concrete to prevent ValueError caused by (?, ?, ?)
				reshaped_line_memory = tf.reshape(line_memory,[batch_size, -1, self._embedding_size])
				reshaped_word_memory = tf.reshape(word_memory,[batch_size, -1, self._sentence_size, self._embedding_size])
				attention_mechanism = CustomAttention(self._embedding_size, reshaped_line_memory, reshaped_word_memory, hierarchy=self._hierarchy, soft_weight=self._soft_weight)
				decoder_cell_with_attn = AttentionWrapper(self.decoder_cell, attention_mechanism, self._keep_prob, output_attention=False)			
				wrapped_encoder_states = decoder_cell_with_attn.zero_state(batch_size, tf.float32).clone(cell_state=encoder_states)
				decoder = BasicDecoder(decoder_cell_with_attn, helper, wrapped_encoder_states, self._decoder_vocab_size, self._oov_sizes, self._oov_ids, output_layer=self.projection_layer)
				return decoder

	def _decoder_train(self, encoder_states, line_memory, word_memory=None):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:
				loss 	- 	Total Loss (Sequence Loss + PGen Loss) (Float)
		'''
		with tf.variable_scope(self._name):
			with tf.variable_scope('decoder'):
				
				## Create Training Helper ##
				batch_size = tf.shape(self._stories)[0]
				# decoder_input = batch_size x candidate_sentence_size
				decoder_input = tf.concat([tf.fill([batch_size, 1], self.GO_SYMBOL), self._answers_emb_lookup[:, :]],axis=1)
				# decoder_emb_inp = batch_size x candidate_sentence_size x embedding_size
				decoder_emb_inp = tf.nn.embedding_lookup(self.C, decoder_input)
				answer_sizes = tf.reshape(self._answer_sizes,[-1])

				## Run Decoder ##
				helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, answer_sizes)
				decoder = self._get_decoder(encoder_states, line_memory, word_memory, helper, batch_size)
				outputs,p_gens = dynamic_decode(decoder, self._batch_size)
				
				## Prepare Loss Helpers ##
				final_dists = outputs.rnn_output
				max_length = tf.reduce_max(answer_sizes, reduction_indices=[0])
				ans = self._answers[:, :max_length]
				target_weights = tf.reshape(self._answer_sizes,[-1])
				target_weights = tf.sequence_mask(target_weights, self._candidate_sentence_size, dtype=tf.float32)
				target_weights = target_weights[:, :max_length]

				## Calculate Sequence Loss ##
				max_oov_len = tf.reduce_max(self._oov_sizes, reduction_indices=[0])
				extended_vsize =  self._decoder_vocab_size + max_oov_len
				y_pred = tf.clip_by_value(final_dists,1e-20,1.0)
				y_true = tf.one_hot(ans, extended_vsize)
				seq_loss_comp = -tf.reduce_sum(y_true*tf.log(y_pred))

				## Calculate PGen Loss ##
				intersect_mask = self._intersection_mask[:, :max_length]
				reshaped_p_gens=tf.reshape(tf.squeeze(p_gens), [-1])
				p = tf.reshape(intersect_mask, [-1])
				q = tf.clip_by_value(reshaped_p_gens,1e-20,1.0)
				one_minus_q = tf.clip_by_value(1-reshaped_p_gens,1e-20,1.0)
				p_gen_loss = p*tf.log(q) + (1-p)*tf.log(one_minus_q)
				pgen_loss_comp = -tf.reduce_sum(p_gen_loss * tf.reshape(target_weights, [-1]))

				loss = seq_loss_comp + self._p_gen_loss_weight*pgen_loss_comp

				return loss, seq_loss_comp, pgen_loss_comp


	def _decoder_runtime(self, encoder_states, line_memory, word_memory=None):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:

		'''
		with tf.variable_scope(self._name):
			with tf.variable_scope('decoder', reuse=True):
				batch_size = tf.shape(self._stories)[0]
				helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.C,tf.fill([batch_size], self.GO_SYMBOL), self.EOS)
				decoder = self._get_decoder(encoder_states, line_memory, word_memory, helper, batch_size)
				outputs,_ = dynamic_decode(decoder, self._batch_size, maximum_iterations=2*self._candidate_sentence_size)
			return tf.argmax(outputs.rnn_output, axis=-1)

	def check_shape(self, name, array):
		shape = array[0].shape
		for i, arr in enumerate(array):
			sh = arr.shape
			if sh != shape:
				print(name, i, shape, sh)

	def print_feed(self, feed_dict, train):
		self.check_shape('Stories: ', feed_dict[self._stories])
		self.check_shape('Story Sizes: ', feed_dict[self._sentence_sizes])
		self.check_shape('Queries: ', feed_dict[self._queries])
		self.check_shape('Queries Sizes: ', feed_dict[self._query_sizes])
		self.check_shape('oov ids: ', feed_dict[self._oov_ids])
		self.check_shape('oov sizes: ', feed_dict[self._oov_sizes])
		self.check_shape('intersection mask: ', feed_dict[self._intersection_mask])
		if train:
			self.check_shape('_answers: ', feed_dict[self._answers])
			self.check_shape('_answers_emb_lookup: ', feed_dict[self._answers_emb_lookup] )
			self.check_shape('_answer_sizes: ', feed_dict[self._answer_sizes])

	def _make_feed_dict(self, batch, train=True):
		"""Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

		Args:
		  batch: Batch object
		  just_enc: Boolean. If True, only feed the parts needed for the encoder.
		"""
		feed_dict = {}
		feed_dict[self._stories] = np.array(batch.stories)
		feed_dict[self._queries] = np.array(batch.queries)
		feed_dict[self._sentence_sizes] = np.array(batch.story_sizes)
		feed_dict[self._query_sizes] = np.array(batch.query_sizes)
		feed_dict[self._oov_ids] = np.array(batch.oov_ids)
		feed_dict[self._oov_sizes] = np.array(batch.oov_sizes)
		feed_dict[self._intersection_mask] = np.array(batch.intersection_set)
		if train:
			feed_dict[self._answers] = np.array(batch.answers)
			feed_dict[self._answers_emb_lookup] = np.array(batch.answers_emb_lookup)
			feed_dict[self._answer_sizes] = np.array(batch.answer_sizes)
			feed_dict[self._keep_prob] = 0.5 
		else:
			feed_dict[self._keep_prob] = 1.0 
		if self._debug:
			self._print_feed(feed_dict, train)
		return feed_dict

	def batch_fit(self, batch):
		"""Runs the training algorithm over the passed batch
		Returns:
			loss: floating-point number, the loss computed for the batch
		"""
		feed_dict = self._make_feed_dict(batch)
		loss, _= self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
		return loss

	def predict(self, batch):
		"""Predicts answers as one-hot encoding.
		Returns:
			answers: Tensor (None, vocab_size)
		"""
		feed_dict = self._make_feed_dict(batch, train=False)
		return self._sess.run(self.predict_op, feed_dict=feed_dict)
