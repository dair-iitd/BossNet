from __future__ import absolute_import
from __future__ import print_function

from data_utils import get_decoder_vocab, load_dialog_task, vectorize_data, load_candidates, vectorize_candidates, vectorize_data_with_surface_form, vectorize_candidates_sparse, tokenize, restaurant_reco_evluation
from sklearn import metrics
from memn2n.memn2n_dialog_generator import MemN2NGeneratorDialog
from itertools import chain
from six.moves import range, reduce
from operator import itemgetter
import sys
import tensorflow as tf
import numpy as np
import os
import pdb
import json

tf.flags.DEFINE_float("learning_rate", 0.01,
                      "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10,
                        "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 400, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 32,
                        "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 3, "bAbI task id, 1 <= id <= 6")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "../data/dialog-bAbI-tasks/",
                       "Directory containing bAbI tasks")
tf.flags.DEFINE_string("logs_dir", "logs/",
                       "Directory containing bAbI tasks")
tf.flags.DEFINE_string("model_dir", "model/",
                       "Directory containing memn2n model checkpoints")
tf.flags.DEFINE_string('loss_type', 'uniform', 'If weighted, use weighted loss function')
tf.flags.DEFINE_boolean('train', False, 'if True, begin to train')
tf.flags.DEFINE_boolean('interactive', False, 'if True, interactive')
tf.flags.DEFINE_boolean('OOV', False, 'if True, use OOV test set')
tf.flags.DEFINE_float('loss_weight', 2.0, 'used when loss type is weighted, to bias the restaurant recommendation')

FLAGS = tf.flags.FLAGS

class chatBot(object):
    def __init__(self, data_dir, model_dir, logs_dir, task_id, isInteractive=True, OOV=False, memory_size=50, random_state=None, batch_size=32, learning_rate=0.01, epsilon=1e-8, max_grad_norm=40.0, evaluation_interval=10, hops=3, epochs=200, embedding_size=20, loss_type='uniform', loss_weight=2.0, is_train=False):
        self.data_dir = data_dir
        self.task_id = task_id
        self.model_dir = model_dir
        self.logs_dir = logs_dir
        # self.isTrain=isTrain
        self.isInteractive = isInteractive
        self.OOV = OOV
        self.memory_size = memory_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.evaluation_interval = evaluation_interval
        self.hops = hops
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.is_train = is_train
        
        if OOV:
            print("Task ", task_id, " learning rate : ", learning_rate, " with OOV")
        else:
            print("Task ", task_id, " learning rate : ", learning_rate)
        print("")
        
        candidates, self.candid2indx = load_candidates(
            self.data_dir, self.task_id)
        self.n_cand = len(candidates)
        print("Candidate Size : ", self.n_cand)
        self.indx2candid = dict(
            (self.candid2indx[key], key) for key in self.candid2indx)
        
        self.decoder_vocab_to_index, self.decoder_index_to_vocab = get_decoder_vocab(self.data_dir, self.task_id)
        print("Decoder Vocab Size : ", len(self.decoder_vocab_to_index))

        # task data
        self.trainData, self.testData, self.valData = load_dialog_task(
            self.data_dir, self.task_id, self.candid2indx, self.OOV)
        self.trainDataVocab, self.testDataVocab, self.valDataVocab = load_dialog_task(
            self.data_dir, self.task_id, self.candid2indx, False)
        dataVocab = self.trainDataVocab + self.testDataVocab + self.valDataVocab
        self.build_vocab(dataVocab, candidates)

        # self.candidates_vec=vectorize_candidates_sparse(candidates,self.word_idx)
        # self.candidates_vec = vectorize_candidates(
        #    candidates, self.word_idx, self.candidate_sentence_size)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.sess = tf.Session()
        self.model = MemN2NGeneratorDialog(self.batch_size, self.vocab_size, self.n_cand, self.sentence_size, self.embedding_size, self.decoder_vocab_to_index, self.candidate_sentence_size, session=self.sess,
                                  hops=self.hops, max_grad_norm=self.max_grad_norm, optimizer=optimizer, task_id=task_id, is_train=self.is_train)
        self.saver = tf.train.Saver(max_to_keep=50)

    def build_vocab(self, data, candidates):
        vocab = reduce(lambda x, y: x | y, (set(
            list(chain.from_iterable(s)) + q) for s, q, a, start in data))
        vocab = sorted(vocab)
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        max_story_size = max(map(len, (s for s, _, _, _ in data)))
        mean_story_size = int(np.mean([len(s) for s, _, _, _ in data]))
        self.sentence_size = max(
            map(len, chain.from_iterable(s for s, _, _, _ in data)))
        self.candidate_sentence_size = max(map(len, candidates))
        query_size = max(map(len, (q for _, q, _, _ in data)))
        self.memory_size = min(self.memory_size, max_story_size)
        self.vocab_size = len(self.word_idx) + 1  # +1 for nil word
        self.sentence_size = max(
            query_size, self.sentence_size)  # for the position
        # params
        print("Vocab Size     : ", self.vocab_size)

    def train(self):
        trainS, trainQ, trainA, trainSZ, trainQZ, trainCZ = vectorize_data(self.trainData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size, self.decoder_vocab_to_index, self.candidate_sentence_size)
        valS, valQ, valA, valSZ, valQZ, valCZ = vectorize_data(self.valData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size, self.decoder_vocab_to_index, self.candidate_sentence_size)
        n_train = len(trainS)
        n_val = len(valS)
        print("Training Size", n_train)
        print("Validation Size", n_val)
        tf.set_random_seed(self.random_state)
        batches = zip(range(0, n_train - self.batch_size, self.batch_size),
                      range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]
        best_validation_accuracy = 0

        for t in range(1, self.epochs + 1):
            np.random.shuffle(batches)
            total_cost = 0.0
            for start, end in batches:
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                c = self.classify_dialogs(a)
                sizes = trainSZ[start:end]
                qsize = trainQZ[start:end]
                asize = trainCZ[start:end]
                cost_t = self.model.batch_fit(s, q, a, c, sizes, qsize, asize)
                total_cost += cost_t
            
            if t % self.evaluation_interval == 0:
                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('-----------------------')
                sys.stdout.flush()
                
                # TODO : Implement validation set accuracy and early stop
                #if val_acc > best_validation_accuracy:
                #    best_validation_accuracy = val_acc
                self.saver.save(self.sess, self.model_dir + 'model.ckpt', global_step=t)

    def test(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
        if self.isInteractive:
            self.interactive()
        else:
            testS, testQ, testA, testSZ, testQZ, testCZ, S_in_readable_form, Q_in_readable_form, last_db_results, dialogIDs  = vectorize_data_with_surface_form(
                self.testData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size, self.decoder_vocab_to_index, self.candidate_sentence_size)
            n_test = len(testS)
            test_preds = self.batch_predict(testS, testQ, testSZ, testQZ, n_test)
            test_acc = metrics.accuracy_score(test_preds, testA)
            match=0
            total=0
            match_acc=0
            total_acc=0
            all_data_points=[]

            #attn_arry_size = self.hops*3
            attn_arry_size = self.hops
            
            for idx, val in enumerate(test_preds):
                answer = self.indx2candid[testA[idx].item(0)]
                data_point={}
                context=[]
                for _, element in enumerate(S_in_readable_form[idx]):
                    context.append(element)
                data_point['context']=context
                data_point['query']=Q_in_readable_form[idx]
                data_point['answer']=answer
                data_point['prediction']=self.indx2candid[val]
                data_point['dialog-id']=dialogIDs[idx]
                if data_point['prediction'] == answer:
                    data_point['matched']=True
                else:
                    data_point['matched']=False
                data_point['context-length']=len(S_in_readable_form[idx])
                    
                if len(S_in_readable_form[idx]) <= 500:
                    for hop_index in range(0, attn_arry_size):
                        attn_tuples_list = []
                        attn_arr = attn_weights[hop_index][idx]
                        for mem_index in range(0, len(attn_arr)):
                            if(mem_index > len(S_in_readable_form[idx])-1):
                                attn_tuples_list.append((mem_index, attn_arr[mem_index], "NONE"))
                            else:
                                if(len(S_in_readable_form[idx]) > 50):
                                    attn_tuples_list.append((mem_index, attn_arr[mem_index], S_in_readable_form[idx][len(S_in_readable_form[idx])-50+mem_index]))
                                else:
                                    attn_tuples_list.append((mem_index, attn_arr[mem_index], S_in_readable_form[idx][mem_index]))
                        sorted_tuple = sorted(attn_tuples_list, key=itemgetter(1), reverse=True)
                        attn_list=[]
                        for tuple_idx in range(0, 10):
                            if len(sorted_tuple) > tuple_idx and sorted_tuple[tuple_idx][1] > 0.001:
                                attn_list.append(str(sorted_tuple[tuple_idx][1]) + ' : ' + sorted_tuple[tuple_idx][2])
                        data_point['attn-hop-' + str(hop_index)]=attn_list
                
                # for hop_index in range(0, attn_arry_size):
                #     print(beta_arr[hop_index][idx][0])
                #     data_point['beta-' + str(hop_index)]=str(beta_arr[hop_index][idx][0])
                
                ranked_candidateids = ranked_candidates_list[idx]
                ranked_candidates = []
                for i in range(len(ranked_candidateids)):
                    ranked_candidates.append(self.indx2candid[ranked_candidateids[i]])
                #data_point['ranked-candidates']=ranked_candidates
                all_data_points.append(data_point)

                if (self.task_id==3 or self.task_id==5) and "what do you think of this option:" in answer and 'dialog-template' in self.data_dir:
                    dbset=set()
                    for counter_temp, element in enumerate(S_in_readable_form[idx]):
                        if counter_temp%8 == 0 and '\t' not in element :
                           dbset.add('loc_' + str(counter_temp+1))
                            
                    total = total+1
                    pred_str=self.indx2candid[val]
                    if "what do you think of this option:" in pred_str:
                        pred_restaurant=pred_str[34:].strip()
                        if pred_restaurant in dbset:
                            match=match+1
                        if data_point['prediction'] == answer:
                            match_acc = match_acc+1

                if (self.task_id==3 or self.task_id==5) and "what do you think of this option:" in answer and 'dialog-template' not in self.data_dir:
                    dbset=set()
                    splitstr=last_db_results[idx].split( )
                    for i in range(2, len(splitstr)):
                        dbset.add(splitstr[i][:splitstr[i].index('(')])
                    
                    total = total+1
                    pred_str=self.indx2candid[val]
                    if "what do you think of this option:" in pred_str:
                        pred_restaurant=pred_str[34:].strip()
                        if pred_restaurant in dbset:
                            match=match+1
                        if data_point['prediction'] == answer:
                            match_acc = match_acc+1
            
            file_prefix = self.logs_dir + "task" + str(FLAGS.task_id) + "_data-dir-" + filter(None, FLAGS.data_dir.split('/'))[-1] + "_lr-" + str(FLAGS.learning_rate) + "_hops-" + str(FLAGS.hops)
            file_to_dump_json=  file_prefix + '.json'
            if self.OOV:
                file_to_dump_json= file_prefix + '_oov.json'
            
            with open(file_to_dump_json, 'w') as f:
                json.dump(all_data_points, f, indent=4)

            print("Test Size      : ", n_test)
            print("Test Accuracy  : ", test_acc)

            if (self.task_id==3 or self.task_id==5) and 'dialog-template' in self.data_dir:
                print('Restaurant Recommendation Accuracy : ' + str(match_acc/float(total)) +  " (" +  str(match_acc) +  "/" + str(total) + ")")
                print('Restaurant Recommendation from Correct Location Accuracy : ' + str(match/float(total)) +  " (" +  str(match) +  "/" + str(total) + ")")
            if (self.task_id==3 or self.task_id==5) and 'dialog-template' not in self.data_dir:
                print('Restaurant Recommendation Accuracy : ' + str(match_acc/float(total)) +  " (" +  str(match_acc) +  "/" + str(total) + ")")
                print('Restaurant Recommendation from DB Accuracy : ' + str(match/float(total)) +  " (" +  str(match) +  "/" + str(total) + ")")
            
            '''
            if self.task_id==3:
                counter = []
                query1 = ['no', 'this', 'does', 'not', 'work', 'for', 'me', '$u', '#0']
                query2 = ['sure', 'let', 'me', 'find', 'other', 'option', 'for', 'you', '$r', '#0']
                for idx in range(0, n_test):
                    answer = self.indx2candid[testA[idx].item(0)]
                    if len(answer) > 0:
                        last = str(answer)
                        if 'what do you think of this option' in last:
                            count = 1
                            s = testS[idx:idx+1]
                            q = testQ[idx:idx+1]
                            a = testA[idx]
                            pred, _, _ = self.model.predict(s, q)
                            turn = S_in_readable_form[idx][-1]
                            turn_no = int(turn.split('#')[1].split(' ')[0])
                            while pred.item(0) != a and count < 100:
                                turn_no += 1
                                turn_element = "#" + str(turn_no)
                                query1[-1] = (turn_element)
                                query2[-1] = (turn_element)
                                query1_hash = [self.word_idx[w] if w in self.word_idx else 0 for w in query1] + [0] * (self.sentence_size - len(query1))
                                query2_hash = [self.word_idx[w] if w in self.word_idx else 0 for w in query2] + [0] * (self.sentence_size - len(query2))
                                request_query_append = np.array([query1_hash, query2_hash])
                                s = [np.concatenate((s[0], request_query_append))]
                                count += 1                   
                                pred, _, _ = self.model.predict(s, q)
                            counter.append(count)
                correct_predictions = [x for x in counter if x != 100]
                print("Suggestion Game Accuracy :", str(float(len(correct_predictions))/len(counter)) +  " (" +  str(len(correct_predictions)) +  "/" + str(len(counter)) + ")" )
                if len(correct_predictions) > 0:
                    print("Suggestion Game Mean     :", float(sum(correct_predictions))/len(correct_predictions))
            '''

            print("------------------------")

    def batch_predict(self, S, Q, SZ, QZ, n):
        preds = []
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            s = S[start:end]
            q = Q[start:end]
            sizes = SZ[start:end]
            qsize = QZ[start:end]
            pred = self.model.predict(s, q, sizes, qsize)
            preds += list(pred)
            
        return preds

    def classify_dialogs(self, A):
        answer_type = []
        for idx in range(len(A)):
            answer = self.indx2candid[A[idx].item(0)]
            if "what do you think of this option:" in answer:
                answer_type.append(1)
            else:
                answer_type.append(0)
        return answer_type

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':
    
    model_dir = FLAGS.model_dir + "task" + str(FLAGS.task_id) + "_" + FLAGS.data_dir.split('/')[-2] + "_lr-" + str(FLAGS.learning_rate) + "_hops-" + str(FLAGS.hops) + "_model/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    chatbot = chatBot(FLAGS.data_dir, model_dir, FLAGS.logs_dir, FLAGS.task_id, OOV=FLAGS.OOV,
                      isInteractive=FLAGS.interactive, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs,
                      learning_rate = FLAGS.learning_rate, hops = FLAGS.hops, embedding_size = FLAGS.embedding_size,
                      loss_type = FLAGS.loss_type, loss_weight=FLAGS.loss_weight, is_train=FLAGS.train)
    
    if FLAGS.train:
        chatbot.train()
    else:
        chatbot.test()
    chatbot.close_session()
