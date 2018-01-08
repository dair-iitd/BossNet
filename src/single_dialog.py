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
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_boolean('interactive', False, 'if True, interactive')
tf.flags.DEFINE_boolean('use_beam_search', False, 'if True, uses beam search for dcoding, else uses greedy decoding')
tf.flags.DEFINE_boolean('use_attention', False, 'if True, uses attention')
tf.flags.DEFINE_boolean('dropout', False, 'if True, uses dropout on p_gen')
tf.flags.DEFINE_boolean('word_drop', False, 'if True, uses random word dropout')
tf.flags.DEFINE_integer("unk_size", 2, "Number of random unk words per batch")

# Output Specifications
tf.flags.DEFINE_boolean('game', False, 'if True, show infinite game results')
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")

# Task Type
tf.flags.DEFINE_boolean('train', False, 'if True, begin to train')
tf.flags.DEFINE_integer("task_id", 3, "bAbI task id, 1 <= id <= 6")
tf.flags.DEFINE_boolean('OOV', False, 'if True, use OOV test set')

# File Locations
tf.flags.DEFINE_string("data_dir", "../data/dialog-bAbI-tasks/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("logs_dir", "logs/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("model_dir", "model/", "Directory containing memn2n model checkpoints")

FLAGS = tf.flags.FLAGS

class chatBot(object):

    def __init__(self):
        # Define Parameters of ChatBot
        self.data_dir = FLAGS.data_dir
        self.task_id = FLAGS.task_id
        word_drop_str = ""
        if FLAGS.word_drop:
            word_drop_str="_words_dropped_" + str(FLAGS.unk_size)
        self.model_dir = FLAGS.model_dir + "task" + str(FLAGS.task_id) + "_" + FLAGS.data_dir.split('/')[-2] + "_lr-" + str(FLAGS.learning_rate) + "_hops-" + str(FLAGS.hops) + word_drop_str + "_model/"
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
        self.use_beam_search= FLAGS.use_beam_search
        self.use_attention = FLAGS.use_attention
        self.dropout = FLAGS.dropout
        self.word_drop = FLAGS.word_drop
        self.unk_size = FLAGS.unk_size
        self.is_train = FLAGS.train

        # Create Model Store Directory
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Print Task Information
        if self.OOV:
            print("Task ", self.task_id, " learning rate : ", self.learning_rate, " with OOV")
        else:
            print("Task ", self.task_id, " learning rate : ", self.learning_rate)
        
        # Load Decoder Vocabulary
        self.decoder_vocab_to_index, self.decoder_index_to_vocab, self.candidate_sentence_size = get_decoder_vocab(self.data_dir, self.task_id)
        print("Decoder Vocab Size : ", len(self.decoder_vocab_to_index))

        # Retreive Task Data
        self.trainData, self.testData, self.valData = load_dialog_task(self.data_dir, self.task_id, self.OOV)
        self.trainDataVocab, self.testDataVocab, self.valDataVocab = load_dialog_task(self.data_dir, self.task_id, False)
        # self.dataVocab = self.trainDataVocab + self.testDataVocab + self.valDataVocab
        # Jan 8 : Build vocab only with training data
        self.build_vocab(self.trainDataVocab)

        # Define MemN2N + Generator Model
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.sess = tf.Session()
        self.model = MemN2NGeneratorDialog(self.batch_size, self.vocab_size, self.sentence_size, 
                                           self.embedding_size, self.decoder_vocab_to_index, self.candidate_sentence_size, 
                                           session=self.sess, hops=self.hops, max_grad_norm=self.max_grad_norm, 
                                           optimizer=self.optimizer, task_id=self.task_id, use_beam_search=self.use_beam_search,
                                           use_attention=self.use_attention, dropout=self.dropout)
        self.saver = tf.train.Saver(max_to_keep=50)

    def build_vocab(self, data):
        '''
            Get vocabulary from the data (Train + Validation + Test)
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
                          self.decoder_vocab_to_index, self.candidate_sentence_size)
        Data_val = Data(self.valData, self.word_idx, self.sentence_size, 
                        self.batch_size, self.memory_size, 
                        self.decoder_vocab_to_index, self.candidate_sentence_size)
        # Create Batches
        n_train = len(Data_train.stories)
        n_val = len(Data_val.stories)
        print("Training Size", n_train)
        print("Validation Size", n_val)
        tf.set_random_seed(self.random_state)
        batches = zip(range(0, n_train - self.batch_size, self.batch_size),
                      range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]
        best_validation_accuracy = 0

        # Train Model in Batch Mode
        print('-----------------------')
        for t in range(1, self.epochs + 1):
            total_cost = self.batch_train(Data_train, batches)
            print('Epoch', t, ' Total Cost:', total_cost)
            
            # Evaluate Model
            if t % self.evaluation_interval == 0:
                train_acc_old, train_acc_new = self.batch_predict(Data_train, n_train)
                val_acc_old, val_acc_new = self.batch_predict(Data_val, n_val)
                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('Training Accuracy Score:', train_acc_old, train_acc_new)
                print('Validation Accuracy Score:', val_acc_old, val_acc_new)
                print('-----------------------')
                sys.stdout.flush()
                
                # Save best model
                if val_acc_old >= best_validation_accuracy:
                    best_validation_accuracy = val_acc_old
                    self.saver.save(self.sess, self.model_dir + 'model.ckpt', global_step=t)
                if val_acc_new >= best_validation_accuracy:
                    best_validation_accuracy = val_acc_new
                    self.saver.save(self.sess, self.model_dir + 'model.ckpt', global_step=t)

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
            Data_test = Data(self.testData, self.word_idx, self.sentence_size, 
                             self.batch_size, self.memory_size, 
                             self.decoder_vocab_to_index, self.candidate_sentence_size)
            n_test = len(Data_test.stories)
            test_acc_old, test_acc_new = self.batch_predict(Data_test, n_test)

            print("Test Size      : ", n_test)
            print("Test Accuracy  : ", test_acc_old, test_acc_new)      
            print("------------------------")

    def interactive(self):
        print("To Be Implemented")

    def batch_train(self, data, batches):
        '''
            Train Model for a Batch of Input Data
        '''
        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            cost_t, logits = self.model.batch_fit(Batch(data, start, end, self.unk_size, self.word_drop))
            total_cost += cost_t
        return total_cost

    def batch_predict(self, data, n):
        '''
            Get Predictions for Input Data batchwise
        '''
        old_preds = []
        new_preds = []
        count = 0
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            count += 1
            if count >= n / self.batch_size:
                break
            old_pred, new_pred = self.model.predict(Batch(data, start, end))
            old_preds += pad_to_answer_size(list(old_pred), self.candidate_sentence_size)
            new_preds += pad_to_answer_size(list(new_pred), self.candidate_sentence_size)
        return substring_accuracy_score(old_preds, data.answers), substring_accuracy_score(new_preds, data.answers,word_map=self.decoder_index_to_vocab,isTrain=self.is_train)

    def close_session(self):
        self.sess.close()

''' Main Function '''
if __name__ == '__main__': 

    chatbot = chatBot()
    
    print("CHATBOT READY")

    if FLAGS.train: chatbot.train()
    else: chatbot.test()

    chatbot.close_session()
