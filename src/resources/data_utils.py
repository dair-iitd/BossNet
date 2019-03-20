import numpy as np
import os
import re
import tensorflow as tf

__all__ = ["get_decoder_vocab",
           "load_dialog_task",
           "tokenize",
           "pad_to_answer_size",
           "create_batches"]

###################################################################################################
#########                                  Global Variables                              ##########
###################################################################################################

stop_words = set(["a", "an", "the"])
PAD_INDEX = 0
UNK_INDEX = 1
GO_SYMBOL_INDEX = 2
EOS_INDEX = 3

###################################################################################################
#########                                 Dialog Manipulators                            ##########
###################################################################################################


def get_decoder_vocab(data_dir, task_id):
    '''
        Load Vocabulary Space for Response-Decoder
    '''
    assert task_id > 0 and task_id < 9
    decoder_vocab_to_index = {}
    decoder_vocab_to_index['PAD'] = PAD_INDEX             # Pad Symbol
    decoder_vocab_to_index['UNK'] = UNK_INDEX             # Unknown Symbol
    decoder_vocab_to_index['GO_SYMBOL'] = GO_SYMBOL_INDEX  # Start Symbol
    decoder_vocab_to_index['EOS'] = EOS_INDEX             # End Symbol

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    candidate_sentence_size = 0
    responses = get_responses(train_file)
    for response in responses:
        line = tokenize(response.strip())
        candidate_sentence_size = max(len(line), candidate_sentence_size)
        for word in line:
            if word not in decoder_vocab_to_index:
                index = len(decoder_vocab_to_index)
                decoder_vocab_to_index[word] = index
    decoder_index_to_vocab = {v: k for k, v in decoder_vocab_to_index.items()}
    return decoder_vocab_to_index, decoder_index_to_vocab, candidate_sentence_size + 1  # (EOS)


def get_responses(file):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    responses = []
    with open(file) as f:
        for line in f.readlines():
            line = line.strip()
            if line and '\t' in line:
                u, r = line.split('\t')
                responses.append(r)
    return responses


def load_dialog_task(data_dir, task_id):
    '''
        Load Train, Test, Validation Dialogs
    '''
    assert task_id > 0 and task_id < 9
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    test_file = [f for f in files if s in f and 'tst' in f and 'OOV' not in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = parse_dialogs(train_file)
    test_data = parse_dialogs(test_file)
    val_data = parse_dialogs(val_file)
    if task_id < 6:
        oov_file = [f for f in files if s in f and 'tst-OOV' in f][0]
        oov_data = parse_dialogs(oov_file)
    else:
        oov_data = None
    return train_data, test_data, val_data, oov_data


def tokenize(sent):
    '''
        Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent = sent.lower()
    if sent == '<silence>':
        return [sent]
    result = [x.strip() for x in re.split(r'(\W+)?', sent) if x.strip()]
    if not result:
        result = ['<silence>']
    if result[-1] == '.' or result[-1] == '?' or result[-1] == '!':
        result = result[:-1]
    return result


def parse_dialogs(file):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data = []
    context = []
    dialog_id = 1
    turn_id = 1
    for line in open(file).readlines():
        line = line.strip()
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if '\t' in line:
                u, r = map(tokenize, line.split('\t'))
                data.append((context[:], u[:], r[:], dialog_id, turn_id))
                u.extend(['$u', '#{}'.format(nid)])
                r.extend(['$r', '#{}'.format(nid)])
                context.append(u)
                context.append(r)
                turn_id += 1
            else:
                r = tokenize(line)
                r.extend(['$db', '#{}'.format(nid)])
                context.append(r)
        else:
            # clear context / start of new dialog
            dialog_id += 1
            turn_id = 1
            context = []
    return data


###################################################################################################
#########                           Evaluation Metrics & Helpers                         ##########
###################################################################################################

def create_batches(data, batch_size):
    '''
        Helps to partition the dialog into three groups
        1) Dialogs occuring before an API call
        2) Dialogs that have an API call
        3) Dialogs that occur after and API call
    '''
    size = len(data.stories)
    batches = zip(range(0, size - batch_size, batch_size),
                  range(batch_size, size, batch_size))
    batches = [(start, end) for start, end in batches]
    # fix to include last batch
    if batches[-1][1] < size:
        batches.append((batches[-1][1], size))
    return batches


def pad_to_answer_size(pred, size):
    for i, list in enumerate(pred):
        sz = len(list)
        if sz >= size:
            pred[i] = list[:size]
        else:
            pred[i] = np.append(list, np.array([PAD_INDEX] * (size - sz)))
    return pred
