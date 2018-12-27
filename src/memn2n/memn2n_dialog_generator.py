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


def add_gradient_noise(t, stddev=1e-3, name=None):
	"""
	Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

	The input Tensor `t` should be a gradient.

	The output will be `t` + gaussian noise.

	0.001 was said to be a good fixed value for memory networks [2].
	"""
	with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
		t = tf.convert_to_tensor(t, name="t")
		gn = tf.random_normal(tf.shape(t), stddev=stddev)
		return tf.add(t, gn, name=name)


###################################################################################################
#########                                     Model Class                                ##########
###################################################################################################

class MemN2NGeneratorDialog(object):
	"""End-To-End Memory Network with a generative decoder."""

	def __init__(self, args):

		# Initialize Model Variables
		self._batch_size = args.batch_size
		self._beam_width = args.beam_width
		self._candidate_sentence_size = args.candidate_sentence_size
		self._char_emb = args.char_emb
		self._decode_idx = args.decode_idx
		self._embedding_size = args.embedding_size
		self._hierarchy = args.hierarchy
		self._hops = args.hops
		self._init = tf.random_normal_initializer(stddev=0.1)
		self._max_grad_norm = args.max_grad_norm
		self._name = 'MemN2N'
		self._opt = args.optimizer
		self._p_gen_loss = args.p_gen_loss
		self._p_gen_loss_weight = args.p_gen_loss_weight
		self._rnn = args.rnn
		self._sentence_size = args.sentence_size
		self._soft_weight = args.soft_weight
		self._token_emb_size = args.char_embedding_size
		self._task_id = args.task_id
		self._vocab_size = args.vocab_size

		# Add unk and eos
		self.UNK = self._decode_idx["UNK"]
		self.EOS = self._decode_idx["EOS"]
		self.GO_SYMBOL = self._decode_idx["GO_SYMBOL"]

		self._decoder_vocab_size = len(self._decode_idx)

		self._build_inputs()
		self._build_vars()

		# define summary directory
		timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
		self.root_dir = "%s_%s_%s_%s/" % ('task',
										  str(self._task_id), 'summary_output', timestamp)

		encoder_states, line_memory, word_memory, attn_arr = self._encoder(self._stories, self._story_positions, self._queries)
		
		# train_op 
		loss_op, logits, seq_loss_op, pgen_loss_op, p_gens, intersect_mask = self._decoder_train(encoder_states, line_memory, word_memory)

		# gradient pipeline
		grads_and_vars = self._opt.compute_gradients(loss_op)
		grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars if g != None]
		nil_grads_and_vars = []
		for g, v in grads_and_vars:
			if v.name in self._nil_vars:
				nil_grads_and_vars.append((zero_nil_slot(g), v))
			else:
				nil_grads_and_vars.append((g, v))
		train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

		# predict ops
		predict_op = self._decoder_runtime(encoder_states, line_memory, word_memory)

		# assign ops
		self.loss_op = loss_op, logits, seq_loss_op, pgen_loss_op, p_gens, intersect_mask
		self.predict_op = predict_op
		self.train_op = train_op

		self.graph_output = self.loss_op

		init_op = tf.global_variables_initializer()
		self._sess = args.session
		self._sess.run(init_op)

	def _build_inputs(self):
		'''
			Define Input Variables to be given to the model
		'''
		self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
		self._story_positions = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="storie_positions")
		self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
		self._answers = tf.placeholder(tf.int32, [None, self._candidate_sentence_size], name="answers")
		self._intersection_mask = tf.placeholder(tf.float32, [None, self._candidate_sentence_size], name="intersection_mask")
		self._answers_emb_lookup = tf.placeholder(tf.int32, [None, self._candidate_sentence_size], name="answers_emb")
		self._sentence_sizes = tf.placeholder(tf.int32, [None, None], name="sentence_sizes")
		self._sentence_tokens = tf.placeholder(tf.int32, [None, None, self._sentence_size, None], name="story_tokens")
		self._sentence_word_sizes = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="sentence_word_sizes")
		self._query_sizes = tf.placeholder(tf.int32, [None, 1], name="query_sizes")
		self._query_tokens = tf.placeholder(tf.int32, [None, self._sentence_size, None], name="query_tokens")
		self._query_word_sizes = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries_word_sizes")
		self._answer_sizes = tf.placeholder(tf.int32, [None, 1], name="answer_sizes")
		self._oov_ids = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="oov_ids")
		self._oov_sizes = tf.placeholder(tf.int32, [None], name="oov_sizes")
		self._keep_prob = tf.placeholder(tf.float32)
		self._token_size = tf.placeholder(tf.int32)

	def _build_vars(self):
		'''
			Define Model specific variables used to train and fit and test the model
		'''
		with tf.variable_scope(self._name):
			nil_word_slot = tf.zeros([1, self._embedding_size])
			A = tf.concat([nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size])], 0)
			self.A = tf.Variable(A, name="A")
			
			C = tf.concat([nil_word_slot, self._init([self._decoder_vocab_size, self._embedding_size])], 0)
			self.C = tf.Variable(C, name="C")

			self.Z = tf.Variable(self._init([self._token_emb_size, self._embedding_size]), name="Z")

			self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")

			with tf.variable_scope('decoder'):
				self.decoder_cell = tf.contrib.rnn.GRUCell(self._embedding_size)
				self.projection_layer = layers_core.Dense(self._decoder_vocab_size, use_bias=False)

			with tf.variable_scope("encoder"):
				self.encoder_fwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)
				self.encoder_bwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)

			if self._char_emb:
				with tf.variable_scope("char_emb"):
					self.char_fwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)
					self.char_bwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)

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


	def _encoder(self, stories, story_positions, queries):
		with tf.variable_scope(self._name):

			### Set Variables ###
			self._batch_size = tf.shape(stories)[0]
			self._memory_size = tf.shape(stories)[1]

			### Transform Queries ###
			# queries : batch_size x sentence_size
			# query_word_emb : batch_size x sentence_size x embedding_size
			query_word_emb = tf.nn.embedding_lookup(self.A, queries)

			if self._char_emb:
				query_token_sizes = tf.reshape(self._query_word_sizes, [-1])
				query_token_emb = tf.nn.embedding_lookup(self.Z, self._query_tokens)
				query_token_emb =  tf.reshape(query_token_emb, tf.stack([-1, self._token_size, self._embedding_size]))
				with tf.variable_scope("char_emb"):
					(outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(self.char_fwd, self.char_bwd, query_token_emb, sequence_length=query_token_sizes, dtype=tf.float32)
				(f_state, b_state) = output_states
				query_char_emb = tf.concat(axis=1, values=[f_state, b_state])
				query_char_emb = tf.reshape(query_char_emb, [self._batch_size, self._sentence_size, self._embedding_size])
				# query_emb : batch_size x sentence_size x embedding_size*2
				query_emb = tf.concat(axis=2, values=[query_char_emb, query_word_emb])
			else:
				# query_emb : batch_size x sentence_size x embedding_size
				query_emb = query_word_emb

			if self._rnn:
				query_sizes = tf.reshape(self._query_sizes, [-1])
				with tf.variable_scope("encoder"):
					(outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(self.encoder_fwd, self.encoder_bwd, query_emb, sequence_length=query_sizes, dtype=tf.float32)
				(f_state, b_state) = output_states

				# u_0 : batch_size x embedding_size
				u_0 = tf.concat(axis=1, values=[f_state, b_state])
			else:
				if self._char_emb:
					query_emb = self._reduce_to_bow(query_emb)
					query_emb = tf.reshape(query_emb, [self._batch_size, self._sentence_size, self._embedding_size])
				u_0 = tf.reduce_sum(query_emb, 1)
			u = [u_0]
			
			### Transform Stories ###
			# stories : batch_size x memory_size x sentence_size
			# memory_word_emb : batch_size x memory_size x sentence_size x embedding_size
			memory_word_emb = tf.nn.embedding_lookup(self.A, stories)
			
			if self._char_emb:
				sentence_token_sizes = tf.reshape(self._sentence_word_sizes, [-1])
				sentence_token_emb = tf.nn.embedding_lookup(self.Z, self._sentence_tokens)
				sentence_token_emb =  tf.reshape(sentence_token_emb, tf.stack([-1, self._token_size, self._embedding_size]))
				with tf.variable_scope("char_emb", reuse=True):
					(outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(self.char_fwd, self.char_bwd, sentence_token_emb, sequence_length=sentence_token_sizes, dtype=tf.float32)
				(f_state, b_state) = output_states
				memory_char_emb = tf.concat(axis=1, values=[f_state, b_state])
				memory_char_emb = tf.reshape(memory_char_emb, [self._batch_size, self._memory_size, self._sentence_size, self._embedding_size])
				memory_word_emb = tf.nn.dropout(memory_word_emb, self._keep_prob)
				memory_emb = tf.concat(axis=3, values=[memory_char_emb, memory_word_emb])
				memory_emb = tf.reshape(memory_emb, [-1, self._sentence_size, self._embedding_size*2])
			else:
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
				if self._char_emb:
					memory_emb = self._reduce_to_bow(memory_emb)
				memory_emb = tf.reshape(memory_emb, [self._batch_size, self._memory_size, self._sentence_size, self._embedding_size])
				line_memory = tf.reduce_sum(memory_emb, 2)
			
			if self._rnn:
				(f_states, b_states) = outputs
				word_memory = tf.concat(axis=2, values=[f_states, b_states]) 
				word_memory = tf.reshape(word_memory, [self._batch_size, self._memory_size, self._sentence_size, self._embedding_size])
			else:
				word_memory = memory_emb

			### Implement Hop Network ###
			attn_arr = []
			for hop_index in range(self._hops):
				
				# hack to get around no reduce_dot
				u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
				dotted = tf.reduce_sum(line_memory * u_temp, 2)

				# Calculate probabilities
				probs = tf.nn.softmax(dotted)
				attn_arr.append(probs)
				probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
				c_temp = tf.transpose(line_memory, [0, 2, 1])
				o_k = tf.reduce_sum(c_temp * probs_temp, 2)
				
				u_k = tf.matmul(u[-1], self.H) + o_k

				u.append(u_k)
			
			return u_k, line_memory, word_memory, attn_arr

	def _get_decoder(self, encoder_states, line_memory, word_memory, helper, batch_size):
		with tf.variable_scope(self._name):
			with tf.variable_scope('decoder'):
				# make the shape concrete to prevent ValueError caused by (?, ?, ?)
				reshaped_line_memory = tf.reshape(line_memory,[batch_size, -1, self._embedding_size])
				reshaped_word_memory = tf.reshape(word_memory,[batch_size, -1, self._sentence_size, self._embedding_size])
				self.attention_mechanism = CustomAttention(self._embedding_size, reshaped_line_memory, reshaped_word_memory, hierarchy=self._hierarchy, soft_weight=self._soft_weight)
				decoder_cell_with_attn = AttentionWrapper(self.decoder_cell, self.attention_mechanism, self._keep_prob, output_attention=False)			
				wrapped_encoder_states = decoder_cell_with_attn.zero_state(batch_size, tf.float32).clone(cell_state=encoder_states)
				decoder = BasicDecoder(decoder_cell_with_attn, helper, wrapped_encoder_states, output_layer=self.projection_layer)
				return decoder

	def _decoder_train(self, encoder_states, line_memory, word_memory=None):
		
		# encoder_states = batch_size x embedding_size
		# answers = batch_size x candidate_sentence_size
		with tf.variable_scope(self._name):
			
			batch_size = tf.shape(self._stories)[0]
			# decoder_input = batch_size x candidate_sentence_size
			decoder_input = tf.concat([tf.fill([batch_size, 1], self.GO_SYMBOL), self._answers_emb_lookup[:, :]],axis=1)
			# decoder_emb_inp = batch_size x candidate_sentence_size x embedding_size
			decoder_emb_inp = tf.nn.embedding_lookup(self.C, decoder_input)

			with tf.variable_scope('decoder'):
				
				answer_sizes = tf.reshape(self._answer_sizes,[-1])
				helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, answer_sizes)
				decoder = self._get_decoder(encoder_states, line_memory, word_memory, helper, batch_size)
				outputs,_,_,p_gens,_,_,_ = dynamic_decode(decoder, self._batch_size, self._decoder_vocab_size, self._oov_sizes, self._oov_ids, impute_finished=False)
				final_dists = outputs.rnn_output
				max_length = tf.reduce_max(answer_sizes, reduction_indices=[0])
				ans = self._answers[:, :max_length]
				
				target_weights = tf.reshape(self._answer_sizes,[-1])
				target_weights = tf.sequence_mask(target_weights, self._candidate_sentence_size, dtype=tf.float32)
				target_weights_eos = target_weights[:, max_length-1]
				target_weights = target_weights[:, :max_length]

				intersect_mask = self._intersection_mask[:, :max_length]
				#p_gen_logits = tf.concat([1-p_gens, 50*p_gens], 2)
				#p_gen_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intersect_mask, logits=p_gen_logits)
				#pgen_loss_comp = tf.reduce_sum(p_gen_loss * target_weights)

				reshaped_p_gens=tf.reshape(tf.squeeze(p_gens), [-1])
				p = tf.reshape(intersect_mask, [-1])
				q = tf.clip_by_value(reshaped_p_gens,1e-20,1.0)
				one_minus_q = tf.clip_by_value(1-reshaped_p_gens,1e-20,1.0)
				p_gen_loss = p*tf.log(q) + (1-p)*tf.log(one_minus_q)
				pgen_loss_comp = -tf.reduce_sum(p_gen_loss * tf.reshape(target_weights, [-1]))

				####
				max_oov_len = tf.reduce_max(self._oov_sizes, reduction_indices=[0])
				extended_vsize =  self._decoder_vocab_size + max_oov_len
				y_pred = tf.clip_by_value(final_dists,1e-20,1.0)
				y_true = tf.one_hot(ans, extended_vsize)
				seq_loss_comp = -tf.reduce_sum(y_true*tf.log(y_pred))
				####
				#crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ans, logits=final_dists)
				#seq_loss_comp = tf.reduce_sum(crossent * target_weights)
				#crossent = crossent[:, max_length-1]
				#seq_loss_comp_eos = tf.reduce_sum(crossent * target_weights_eos)
				
				#loss = seq_loss_comp + self._eos_weight*seq_loss_comp_eos + self._p_gen_loss_weight*pgen_loss_comp
				loss = seq_loss_comp + self._p_gen_loss_weight*pgen_loss_comp

				# crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ans, logits=final_dists)
				# seq_loss_comp = tf.reduce_sum(crossent * target_weights)
				# crossent = crossent[:, max_length-1]
				# seq_loss_comp_eos = tf.reduce_sum(crossent * target_weights_eos)
				
				# loss = seq_loss_comp + self._eos_weight*seq_loss_comp_eos + self._p_gen_loss_weight*pgen_loss_comp

				return loss, final_dists, seq_loss_comp, pgen_loss_comp, p_gens, intersect_mask


	def _decoder_runtime(self, encoder_states, line_memory, word_memory=None):
		
		# encoder_states = batch_size x 1
		# answers = batch_size x candidate_sentence_size
		with tf.variable_scope(self._name):
			batch_size = tf.shape(self._stories)[0]
			with tf.variable_scope('decoder', reuse=True):
				helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.C,tf.fill([batch_size], self.GO_SYMBOL), self.EOS)
				decoder = self._get_decoder(encoder_states, line_memory, word_memory, helper, batch_size)
				
				outputs,_,_, p_gens, hier, line, word = dynamic_decode(decoder, self._batch_size, self._decoder_vocab_size, self._oov_sizes, self._oov_ids, maximum_iterations=2*self._candidate_sentence_size)
				final_dists = outputs.rnn_output

			old_translations = outputs.sample_id
			new_translations = tf.argmax(final_dists, axis=-1)
			return old_translations, new_translations, hier, line, word, p_gens

	def check_shape(self, name, array):
		shape = array[0].shape
		for i, arr in enumerate(array):
			sh = arr.shape
			if sh != shape:
				print(name, i, shape, sh)

	def print_feed(self, feed_dict):
		self.check_shape('Stories: ', feed_dict[self._stories])
		self.check_shape('Story Positions: ', feed_dict[self._story_positions])
		self.check_shape('Story Sizes: ', feed_dict[self._sentence_sizes])
		self.check_shape('Queries: ', feed_dict[self._queries])
		self.check_shape('Queries Sizes: ', feed_dict[self._query_sizes])
		self.check_shape('oov ids: ', feed_dict[self._oov_ids])
		self.check_shape('oov sizes: ', feed_dict[self._oov_sizes])
		self.check_shape('intersection mask: ', feed_dict[self._intersection_mask])
		if self._char_emb:
			self.check_shape('_sentence_tokens: ', feed_dict[self._sentence_tokens])
			self.check_shape('_query_tokens: ', feed_dict[self._query_tokens])
			self.check_shape('_sentence_word_sizes: ', feed_dict[self._sentence_word_sizes])
			self.check_shape('_query_word_sizes: ', feed_dict[self._query_word_sizes])

	def _make_feed_dict(self, batch, train=True):
		"""Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

		Args:
		  batch: Batch object
		  just_enc: Boolean. If True, only feed the parts needed for the encoder.
		"""
		feed_dict = {}
		feed_dict[self._stories] = batch.stories
		feed_dict[self._queries] = batch.queries
		feed_dict[self._sentence_sizes] = batch.story_sizes
		feed_dict[self._query_sizes] = batch.query_sizes
		feed_dict[self._oov_ids] = batch.oov_ids
		feed_dict[self._oov_sizes] = batch.oov_sizes
		feed_dict[self._intersection_mask] = batch.intersection_set
		if self._char_emb:
			feed_dict[self._sentence_tokens] = batch.story_tokens
			feed_dict[self._query_tokens] = batch.query_tokens
			feed_dict[self._sentence_word_sizes] = batch.story_word_sizes
			feed_dict[self._query_word_sizes] = batch.query_word_sizes
			feed_dict[self._token_size] = batch.token_size
		if train:
			feed_dict[self._answers] = batch.answers
			feed_dict[self._answers_emb_lookup] = batch.answers_emb_lookup
			feed_dict[self._answer_sizes] = batch.answer_sizes
			feed_dict[self._keep_prob] = 0.5 
		else:
			feed_dict[self._keep_prob] = 1.0 
		return feed_dict

	def batch_fit(self, batch):
		"""Runs the training algorithm over the passed batch

		Args:
			stories: Tensor (None, memory_size, sentence_size)
			queries: Tensor (None, sentence_size)
			answers: Tensor (None, vocab_size)
			sentence_sizes: Tensor (None, memory_size)
			query_sizes: Tensor (None, 1)
			answer_sizes: Tensor (None, 1)

		Returns:
			loss: floating-point number, the loss computed for the batch
		"""
		feed_dict = self._make_feed_dict(batch)
		loss, _= self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
		return loss

	def predict(self, batch):
		"""Predicts answers as one-hot encoding.

		Args:
			stories: Tensor (None, memory_size, sentence_size)
			queries: Tensor (None, sentence_size)
			sentence_sizes: Tensor (None, 1)
			query_sizes: Tensor (None, 1)

		Returns:
			answers: Tensor (None, vocab_size)S
		"""
		feed_dict = self._make_feed_dict(batch, train=False)
		return self._sess.run(self.predict_op, feed_dict=feed_dict)
