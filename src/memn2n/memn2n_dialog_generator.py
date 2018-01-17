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

    def __init__(self, batch_size, vocab_size, sentence_size, embedding_size,
                 decoder_vocab_to_index,candidate_sentence_size, 
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                 session=tf.Session(),
                 name='MemN2N',
                 task_id=1,
                 use_beam_search=False,
                 use_attention=False,
                 dropout=False,
                 char_emb=False,
                 reduce_states=False):

        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            candidates_vec: The numpy array of candidates encoding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._name = name
        self._candidate_sentence_size = candidate_sentence_size
        self._beam_width = 10
        self._use_beam_search = use_beam_search
        self._use_attention = use_attention
        self._dropout = dropout
        self._char_emb = char_emb
        self._reduce_states = reduce_states
        
        # add unk and eos
        self.UNK = decoder_vocab_to_index["UNK"]
        self.EOS = decoder_vocab_to_index["EOS"]
        self.GO_SYMBOL = decoder_vocab_to_index["GO_SYMBOL"]

        self._decoder_vocab_size = len(decoder_vocab_to_index)
        self._decoder_vocab_to_index = decoder_vocab_to_index

        self._build_inputs()
        self._build_vars()

        # define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = "%s_%s_%s_%s/" % ('task',
                                          str(task_id), 'summary_output', timestamp)

        encoder_states, line_memory, word_memory, attn_arr = self._encoder(self._stories, self._queries)
            
        # train_op 
        loss_op, logits = self._decoder_train(encoder_states, line_memory, word_memory)
        
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
        self.loss_op = loss_op, logits
        self.predict_op = predict_op
        self.train_op = train_op

        self.graph_output = self.loss_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        '''
            Define Input Variables to be given to the model
        '''
        self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._candidate_sentence_size], name="answers")
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
        self._token_size = tf.placeholder(tf.float32)

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

            self.Z = tf.Variable(self._init([256, self._embedding_size]), name="Z")

            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")

            with tf.variable_scope('decoder'):
                self.decoder_cell = tf.contrib.rnn.GRUCell(self._embedding_size)
                self.projection_layer = layers_core.Dense(self._decoder_vocab_size, use_bias=False)

            if self._reduce_states:
                with tf.variable_scope("encoder"):
                    self.encoder_fwd = tf.contrib.rnn.GRUCell(self._embedding_size)
                    self.encoder_bwd = tf.contrib.rnn.GRUCell(self._embedding_size)

                with tf.variable_scope('reduce_final_st'):
                    # Define weights and biases to reduce the cell and reduce the state
                    self.w_reduce = tf.Variable(self._init([self._embedding_size * 2, self._embedding_size]), name="w_reduce")
                    self.bias_reduce = tf.Variable(self._init([self._embedding_size]), name="bias_reduce")

                with tf.variable_scope('reduce_word_st'):
                    # Define weights and biases to reduce the cell and reduce the state
                    self.w_reduce_word = tf.Variable(self._init([self._embedding_size * 2, self._embedding_size]), name="w_reduce_word")
                    self.bias_reduce_word = tf.Variable(self._init([self._embedding_size]), name="bias_reduce_word")

                if self._char_emb:
                    with tf.variable_scope("char_emb"):
                        self.char_fwd = tf.contrib.rnn.GRUCell(self._embedding_size)
                        self.char_bwd = tf.contrib.rnn.GRUCell(self._embedding_size)

                    with tf.variable_scope('reduce_char_st'):
                        # Define weights and biases to reduce the cell and reduce the state
                        self.w_reduce_char = tf.Variable(self._init([self._embedding_size * 2, self._embedding_size]), name="w_reduce_char")
                        self.bias_reduce_char = tf.Variable(self._init([self._embedding_size]), name="bias_reduce_char")
                
            else:
                with tf.variable_scope("encoder"):
                    self.encoder_fwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)
                    self.encoder_bwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)

                if self._char_emb:
                    with tf.variable_scope("char_emb"):
                        self.char_fwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)
                        self.char_bwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)

        self._nil_vars = set([self.A.name])

    def _reduce_states_fn(self, fw_st, bw_st):
        with tf.variable_scope('reduce_final_st'):

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st, bw_st]) # Concatenation of fw and bw cell
            new_c = tf.nn.relu(tf.matmul(old_c, self.w_reduce) + self.bias_reduce) # Get new cell from old cell
            return new_c # Return new cell state

    def _reduce_word_states(self, fw_st, bw_st):
        with tf.variable_scope('reduce_word_st'):

            # Apply linear layer
            old_c = tf.concat(axis=2, values=[fw_st, bw_st]) # Concatenation of fw and bw cell
            old_c = tf.reshape(old_c, [-1, self._embedding_size * 2])
            new_c = tf.nn.relu(tf.matmul(old_c, self.w_reduce_word) + self.bias_reduce_word) # Get new cell from old cell
            return new_c # Return new cell state

    def _reduce_char_states(self, fw_st, bw_st):
        with tf.variable_scope('reduce_char_st'):

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st, bw_st]) # Concatenation of fw and bw cell
            new_c = tf.nn.relu(tf.matmul(old_c, self.w_reduce_char) + self.bias_reduce_char) # Get new cell from old cell
            return new_c # Return new cell state

    def _encoder(self, stories, queries):
        with tf.variable_scope(self._name):
            attn_arr = []
            # queries : batch_size x sentence_size
            # q_emb : batch_size x sentence_size x embedding_size
            q_emb = tf.nn.embedding_lookup(self.A, queries)

            q_sizes = tf.reshape(self._query_sizes, [-1])
            with tf.variable_scope("encoder"):
                (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(self.encoder_fwd, self.encoder_bwd, q_emb, sequence_length=q_sizes, dtype=tf.float32)
            (f_state, b_state) = output_states
            if self._reduce_states:
                u_0 = self._reduce_states_fn(f_state, b_state)
            else:
                u_0 = tf.concat(axis=1, values=[f_state, b_state])
            # u_0 : batch_size x embedding_size
            # u_0 = tf.concat([f_state, b_state], 1)

            u = [u_0]

            memory_size = tf.shape(stories)[1]
            batch_size = tf.shape(stories)[0]
            self._memory_size = memory_size

            # stories : batch_size x memory_size x sentence_size
            # m_emb : batch_size x memory_size x sentence_size x embedding_size
            m_emb = tf.nn.embedding_lookup(self.A, stories)

            sentences = tf.reshape(m_emb, [-1, self._sentence_size, self._embedding_size])
            sizes = tf.reshape(self._sentence_sizes, [-1])
            with tf.variable_scope("encoder", reuse=True):
                (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(self.encoder_fwd, self.encoder_bwd, sentences, sequence_length=sizes, dtype=tf.float32)
            (f_state, b_state) = output_states
            (f_states, b_states) = outputs
            
            if self._reduce_states:
                new_state = self._reduce_states_fn(f_state, b_state)
                m_word = self._reduce_states_fn(f_states, b_states)
            else:
                new_state = tf.concat(axis=1, values=[f_state, b_state])
                old_c = tf.concat(axis=2, values=[f_states, b_states]) # Concatenation of fw and bw cell
                m_word = tf.reshape(old_c, [-1, self._embedding_size])

            # m : batch_size x memory_size x embedding_size
            # m = tf.reshape(tf.concat([f_state, b_state], 1), [batch_size, memory_size, -1])
            m = tf.reshape(new_state, [batch_size, memory_size, self._embedding_size])
            max_length = self._sentence_size - tf.reduce_max(sizes)
            m_padding = tf.zeros([batch_size, memory_size, max_length, self._embedding_size])
            m_word = tf.reshape(m_word, [batch_size, memory_size, -1, self._embedding_size])
            
            if self._char_emb:
                s_sizes = tf.reshape(self._sentence_word_sizes, [-1])
                token_emb = tf.nn.embedding_lookup(self.Z, self._sentence_tokens)
                s_tokens =  tf.reshape(token_emb, [-1, 45, self._embedding_size])
                with tf.variable_scope("char_emb"):
                    (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(self.char_fwd, self.char_bwd, s_tokens, sequence_length=s_sizes, dtype=tf.float32)
                (f_state, b_state) = output_states
                if self._reduce_states:
                    char_query = self._reduce_char_states(f_state, b_state)
                else:
                    char_query = tf.concat(axis=1, values=[f_state, b_state])
                char_query = tf.reshape(char_query, [batch_size, memory_size, -1, self._embedding_size])
                m_word = tf.nn.dropout(m_word, self._keep_prob)
                word_level_emb = tf.concat(axis=3, values=[char_query, m_word])
            else:
                word_level_emb = m_word

            for hop_index in range(self._hops):
                
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)
                attn_arr.append(probs)
                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_temp = tf.transpose(m, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                u_k = tf.matmul(u[-1], self.H) + o_k

                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)
            
            return u_k, m, word_level_emb, attn_arr

    def _get_decoder(self, encoder_states, line_memory, word_memory, helper, batch_size):
        with tf.variable_scope(self._name):
            with tf.variable_scope('decoder'):
                if self._use_attention:
                    # make the shape concrete to prevent ValueError caused by (?, ?, ?)
                    reshaped_line_memory = tf.reshape(line_memory,[batch_size, -1, self._embedding_size])
                    if self._char_emb:
                        reshaped_word_memory = tf.reshape(word_memory,[batch_size, -1, self._sentence_size, self._embedding_size * 2])
                    else:
                        reshaped_word_memory = tf.reshape(word_memory,[batch_size, -1, self._sentence_size, self._embedding_size])
                    self.attention_mechanism = CustomAttention(self._embedding_size, reshaped_line_memory, reshaped_word_memory, char_emb=self._char_emb)
                    decoder_cell_with_attn = AttentionWrapper(self.decoder_cell, self.attention_mechanism, self._keep_prob, output_attention=False, dropout=self._dropout)
                
                    # added wrapped_encoder_states to overcome https://github.com/tensorflow/tensorflow/issues/11540
                    wrapped_encoder_states = decoder_cell_with_attn.zero_state(batch_size, tf.float32).clone(cell_state=encoder_states)
                    decoder = BasicDecoder(decoder_cell_with_attn, helper, wrapped_encoder_states, output_layer=self.projection_layer)
                else:    
                    decoder = BasicDecoder(self.decoder_cell, helper, encoder_states, output_layer=self.projection_layer)

                return decoder

    def _decoder_train(self, encoder_states, line_memory, word_memory):
        
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
                outputs,_,_ = dynamic_decode(self._get_decoder(encoder_states, line_memory, word_memory, helper, batch_size), self._batch_size, self._decoder_vocab_size, self._oov_sizes, self._oov_ids, impute_finished=False)
                final_dists = outputs.rnn_output
                max_length = tf.reduce_max(answer_sizes, reduction_indices=[0])
                ans = self._answers[:, :max_length]
                
                target_weights = tf.reshape(self._answer_sizes,[-1])
                target_weights = tf.sequence_mask(target_weights, self._candidate_sentence_size, dtype=tf.float32)
                target_weights = target_weights[:, :max_length] 

                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ans, logits=final_dists)
                loss = tf.reduce_sum(crossent * target_weights)

        return loss, final_dists

    def _decoder_runtime(self, encoder_states, line_memory, word_memory):
        
        # encoder_states = batch_size x 1
        # answers = batch_size x candidate_sentence_size
        with tf.variable_scope(self._name):
            batch_size = tf.shape(self._stories)[0]
            with tf.variable_scope('decoder', reuse=True):
                
                if self._use_beam_search:
                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        self.decoder_cell,self.C,tf.fill([batch_size], self.GO_SYMBOL),
                        self.EOS, tf.contrib.seq2seq.tile_batch(encoder_states, multiplier=self._beam_width), 
                        self._beam_width, output_layer = self.projection_layer)
                else:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.C,tf.fill([batch_size], self.GO_SYMBOL), self.EOS)
                    decoder = self._get_decoder(encoder_states, line_memory, word_memory, helper, batch_size)

                outputs,_,_ = dynamic_decode(decoder, self._batch_size, self._decoder_vocab_size, self._oov_sizes, self._oov_ids, maximum_iterations=2*self._candidate_sentence_size)
                final_dists = outputs.rnn_output

                if self._use_beam_search:
                    predicted_ids = outputs.predicted_ids
                    old_translations = tf.gather(predicted_ids,0,axis=2)
                    new_translations = old_translations
                else:
                    old_translations = outputs.sample_id
                    new_translations = tf.argmax(final_dists, axis=-1)
                
        return old_translations, new_translations

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
        if self._char_emb:
            feed_dict[self._sentence_tokens] = batch.story_tokens
        #     feed_dict[self._query_tokens] = batch.query_tokens
            feed_dict[self._sentence_word_sizes] = batch.story_word_sizes
            feed_dict[self._token_size] = 45
        #     feed_dict[self._query_word_sizes] = batch.query_word_sizes
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
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, batch):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            sentence_sizes: Tensor (None, 1)
            query_sizes: Tensor (None, 1)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = self._make_feed_dict(batch, train=False)
        return self._sess.run(self.predict_op, feed_dict=feed_dict)
