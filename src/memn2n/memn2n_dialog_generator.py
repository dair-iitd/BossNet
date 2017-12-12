from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import numpy as np
from six.moves import range
from datetime import datetime


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


class MemN2NGeneratorDialog(object):
    """End-To-End Memory Network with a generative decoder."""

    def __init__(self, batch_size, vocab_size, candidates_size, sentence_size, embedding_size,
                 decoder_vocab_to_index,candidate_sentence_size, 
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                 session=tf.Session(),
                 name='MemN2N',
                 task_id=1):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            candidates_size: The size of candidates

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
        self._candidates_size = candidates_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._name = name
        self._candidate_sentence_size = candidate_sentence_size
        
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

        encoder_states,attn_arr = self._encoder(self._stories, self._queries)
        
        # train_op 
        loss_op, logits = self._decoder_train(encoder_states)
        
        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                          for g, v in grads_and_vars]

        # grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = self._decoder_runtime(encoder_states)
        
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
        self._sentence_sizes = tf.placeholder(tf.int32, [None, None], name="sentence_sizes")
        self._query_sizes = tf.placeholder(tf.int32, [None, 1], name="query_sizes")
        self._answer_sizes = tf.placeholder(tf.int32, [None, 1], name="answer_sizes")

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

            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            
            with tf.variable_scope("encoder"):
                self.encoder_fwd = tf.contrib.rnn.GRUCell(self._embedding_size/2)
                self.encoder_bwd = tf.contrib.rnn.GRUCell(self._embedding_size/2)

            with tf.variable_scope('decoder'):
                self.decoder_cell = tf.contrib.rnn.GRUCell(self._embedding_size)
                self.projection_layer = layers_core.Dense(self._decoder_vocab_size, use_bias=False)
                
        self._nil_vars = set([self.A.name])

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
            # u_0 : batch_size x embedding_size
            u_0 = tf.concat([f_state, b_state], 1)
            u = [u_0]

            memory_size = tf.shape(stories)[1]
            batch_size = tf.shape(stories)[0]

            for hop_index in range(self._hops):
                # stories : batch_size x memory_size x sentence_size
                # m_emb : batch_size x memory_size x sentence_size x embedding_size
                m_emb = tf.nn.embedding_lookup(self.A, stories)
                
                sentences = tf.reshape(m_emb, [-1, self._sentence_size, self._embedding_size])
                sizes = tf.reshape(self._sentence_sizes, [-1])
                with tf.variable_scope("encoder", reuse=True):
                    (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(self.encoder_fwd, self.encoder_bwd, sentences, sequence_length=sizes, dtype=tf.float32)
                (f_state, b_state) = output_states
                # m : batch_size x memory_size x embedding_size
                m = tf.reshape(tf.concat([f_state, b_state], 1), [batch_size, memory_size, -1])
                    
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
            
            return u_k, attn_arr
            
    def _decoder_train(self, encoder_states):
        
        # encoder_states = batch_size x embedding_size
        # answers = batch_size x candidate_sentence_size
        with tf.variable_scope(self._name):
            
            batch_size = tf.shape(self._stories)[0]
            # decoder_input = batch_size x candidate_sentence_size
            decoder_input = tf.concat([tf.fill([batch_size, 1], self.GO_SYMBOL), self._answers[:, :]],axis=1)
            # decoder_emb_inp = batch_size x candidate_sentence_size x embedding_size
            decoder_emb_inp = tf.nn.embedding_lookup(self.C, decoder_input)

            with tf.variable_scope('decoder'):
                answer_sizes = tf.reshape(self._answer_sizes,[-1])
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, answer_sizes)
                decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, encoder_states, output_layer=self.projection_layer)
                outputs,_ ,_ = tf.contrib.seq2seq.dynamic_decode(decoder)
                logits = outputs.rnn_output
                max_length = tf.reduce_max(answer_sizes, reduction_indices=[0])
                ans = self._answers[:, :max_length]
                
                target_weights = tf.reshape(self._answer_sizes,[-1])
                target_weights = tf.sequence_mask(target_weights, self._candidate_sentence_size, dtype=tf.float32)
                target_weights = target_weights[:, :max_length] 

                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ans, logits=logits)
                loss = tf.reduce_sum(crossent * target_weights)

        return loss, logits

    def _decoder_runtime(self, encoder_states):
        
        # encoder_states = batch_size x 1
        # answers = batch_size x candidate_sentence_size
        with tf.variable_scope(self._name):
            batch_size = tf.shape(self._stories)[0]
            with tf.variable_scope('decoder', reuse=True):
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.C,tf.fill([batch_size], self.GO_SYMBOL), self.EOS)
                decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, encoder_states,output_layer=self.projection_layer)
                outputs,_ ,_ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=2*self._candidate_sentence_size)
                translations = outputs.sample_id

        return translations

    def batch_fit(self, stories, queries, answers, sentence_sizes, query_sizes, answer_sizes):
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
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, 
                self._sentence_sizes: sentence_sizes, self._query_sizes: query_sizes, self._answer_sizes: answer_sizes}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, stories, queries, sentence_sizes, query_sizes):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            sentence_sizes: Tensor (None, 1)
            query_sizes: Tensor (None, 1)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._sentence_sizes: sentence_sizes, self._query_sizes: query_sizes}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)
