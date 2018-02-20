from __future__ import absolute_import

import os
import re
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from string import punctuation
import random

def load_dialog_task(data_dir, task_id, isOOV):
    ''' 
        Load Train, Test, Validation Dialogs 
    '''
    assert task_id > 0 and task_id < 7
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    if isOOV:
        test_file = [f for f in files if s in f and 'tst-OOV' in f][0]
    else: 
        test_file = [f for f in files if s in f and 'tst' in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = get_dialogs(train_file)
    test_data = get_dialogs(test_file)
    val_data = get_dialogs(val_file)
    return train_data, val_data, test_data

def get_dialogs(f):
    '''
        Given a file name, read the file, retrieve the dialogs, 
        and then convert the sentences into a single dialog.
    '''
    with open(f) as f:
        return parse_dialogs_per_response(f.readlines())

def parse_dialogs_per_response(lines):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data=[]
    context=[]
    for line in lines:
        line=line.strip()
        if line:
            context.append(line)
        else:
            data.append(context)
            context=[] # clear context
    return data

def reverse_word(word):
    return word[::-1]

def modify_task1(dialogs):
    cuisine = set()
    location = set()
    for dialog in dialogs:
        for line in dialog:
            if '\t' in line:
                line = line.split('\t')
                user = line[0]
                response = line[1]
                response = response.split()
                if response[0] == 'api_call':
                    print response
                    if response[1] not in cuisine:
                        cuisine.add(response[1])
                    if response[2] not in location:
                        location.add(response[2])
    cuisine_dict = {}
    location_dict = {}
    print cuisine
    print location
    for item in cuisine:
        cuisine_dict[item] = reverse_word(item)
    for item in location:
        location_dict[item] = reverse_word(item)
    context = []
    for k, dialog in enumerate(dialogs):
        # print k
        data = []
        for line in dialog:
            if '\t' in line:
                user, response = line.split('\t')
                user = user.split()
                response = response.split()
                for i, word in enumerate(user):
                    if word in cuisine_dict:
                        user[i] = cuisine_dict[word]
                    elif word in location_dict:
                        user[i] = location_dict[word]
                for i, word in enumerate(response):
                    if word in cuisine_dict:
                        response[i] = cuisine_dict[word]
                    elif word in location_dict:
                        response[i] = location_dict[word]
                if 'resto' in response[-1]:
                    tokens = response[-1].split('_')
                    tokens[1] = reverse_word(tokens[1])
                    tokens[3] = reverse_word(tokens[3])
                    response[-1] = "_".join(tokens)
                user = " ".join(user)
                response = " ".join(response)
                line = user + '\t' + response
            else:
                words = line.split()
                for i, word in enumerate(words):
                    if 'resto' in word:
                        tokens = word.split('_')
                        tokens[1] = reverse_word(tokens[1])
                        tokens[3] = reverse_word(tokens[3])
                        words[i] = "_".join(tokens)
                    if word in cuisine_dict:
                        words[i] = cuisine_dict[word]
                    if word in location_dict:
                        words[i] = location_dict[word]
                line = " ".join(words)
            data.append(line)
        context.append(data)
    return context

def create_dev(val, oov):
    # 20% of val
    val_items = random.sample(val, len(val)/5)
    # 80% of oov
    oov_items = random.sample(oov, len(oov)*4/5)
    dev = val_items + oov_items
    return dev
    # return random.sample(dev, len(dev))

train, val, test = load_dialog_task("../data/dialog-bAbI-tasks/", 5, True)
new_val = modify_task1(val)

# dev = create_dev(val, test)
with open('task5_dev.txt', 'a') as the_file:
    for story in new_val:
        for line in story:
            the_file.write(line + '\n')
        the_file.write('\n')

# train, val, test = load_dialog_task("../data/dialog-bAbI-tasks/", 2, True)
# dev = create_dev(val, test)
# with open('task2_dev.txt', 'a') as the_file:
#     for story in dev:
#         for line in story:
#             the_file.write(line + '\n')
#         the_file.write('\n')

# train, val, test = load_dialog_task("../data/dialog-bAbI-tasks/", 5, True)
# dev = create_dev(val, test)
# with open('task5_dev.txt', 'a') as the_file:
#     for story in dev:
#         for line in story:
#             the_file.write(line + '\n')
#         the_file.write('\n')


