import numpy as np
import random
from itertools import chain
import pdb

PAD_INDEX = 0
UNK_INDEX = 1
GO_SYMBOL_INDEX = 2
EOS_INDEX = 3


class Data(object):

    def __init__(self, data, args, glob):
        self._db_vocab_id = glob['word_idx'].get('$db', -1)
        self._decode_vocab_size = len(glob['decode_idx'])
        self._char_set = [] # For tokenize function

        ## Sort Dialogs based on turn_id
        self._extract_data_items(data)
        
        ## Process Stories
        self._vectorize_stories(self._stories_ext, args, glob)

        ## Process Queries
        self._vectorize_queries(self._queries_ext, glob)
        
        ## Process Answers
        self._vectorize_answers(self._answers_ext, glob)
        
        ## Create DB word mappings to Vocab
        self._entity_set = self._populate_entity_set(self._stories_ext, self._answers_ext)
        
        ## Get indicies where copying must take place
        self._intersection_set = self._intersection_set_mask(self._answers, self._entity_set, glob)
        
        ## Get entities at response level
        self._entities = self._get_entity_indecies(self._read_answers, self._entity_set)

        
    ## Dialogs ##
    @property
    def stories(self):
        return self._stories

    @property
    def queries(self):
        return self._queries

    @property
    def answers(self):
        return self._answers

    ## Sizes ##
    @property
    def story_lengths(self):
        return self._story_lengths

    @property
    def story_sizes(self):
        return self._story_sizes

    @property
    def query_sizes(self):
        return self._query_sizes

    @property
    def answer_sizes(self):
        return self._answer_sizes

    ## Read Dialogs ##
    @property
    def readable_stories(self):
        return self._read_stories

    @property
    def readable_queries(self):
        return self._read_queries

    @property
    def readable_answers(self):
        return self._read_answers

    ## Char Tokens ##
    @property
    def story_tokens(self):
        return self._story_tokens

    @property
    def query_tokens(self):
        return self._query_tokens

    ## Char Tokens Sizes ##
    @property
    def story_word_sizes(self):
        return self._story_word_sizes

    @property
    def query_word_sizes(self):
        return self._query_word_sizes

    @property
    def token_size(self):
        return self._token_size

    @property
    def char_set(self):
        return self._char_set

    ## OOV ##
    @property
    def oov_ids(self):
        return self._oov_ids

    @property
    def oov_sizes(self):
        return self._oov_sizes

    @property
    def oov_words(self):
        return self._oov_words

    ## Dialog Info ##
    @property
    def dialog_ids(self):
        return self._dialog_ids

    @property
    def turn_ids(self):
        return self._turn_ids

    @property
    def db_vocab_id(self):
        return self._db_vocab_id

    ## Decode Variables ##
    @property
    def answers_emb_lookup(self):
        return self._answers_emb_lookup

    ## DB(entity) Words and Vocab Maps ##
    @property
    def entity_set(self):
        return self._entity_set

    @property
    def entities(self):
        return self._entities

    @property
    def responses(self):
        return self._responses
    
    ## PGen Mask
    @property
    def intersection_set(self):
        return self._intersection_set

    def _extract_data_items(self, data):
        '''
            Sorts the dialogs and seperates into respective lists
        '''
        data.sort(key=lambda x: len(x[0]), reverse=True)    # Sort based on dialog size
        self._stories_ext, self._queries_ext, self._answers_ext, self._dialog_ids, self._turn_ids = zip(*data)

    def _index(self, token):
        '''
            Maintains a list of seen charater tokens and assigns corresponding token ids
        '''
        if token in self._char_set:
            return self._char_set.index(token) + 1
        else:
            self._char_set.append(token)
            return len(self._char_set)

    def _tokenize(self, word, size=1):
        '''
            Breaks each sentence into a list of tokens of given size
        '''
        tokens = []
        start = 0; end = size
        while len(word) < size: word += " " 
        while end <= len(word):
            tokens.append(self._index(word[start:end]))
            start += 1; end += 1;
        else: return tokens

    def _vectorize_stories(self, stories, args, glob):     
        '''
            Maps each story into word and character tokens and assigns them ids
        '''   
        self._stories = []              # Encoded Stories (using word_idx)
        self._story_lengths = []        # Story Lengths
        self._story_sizes = []          # Story sentence sizes
        self._story_word_sizes = []     # Story word sizes
        self._story_tokens = []         # Character Tokens of Stories
        self._read_stories = []         # Readable Stories
        self._oov_ids = []              # The index of words for copy in Response-Decoder
        self._oov_sizes = []            # The size of OOV words set in Response-Decoder
        self._oov_words = []            # The OOV words in the Stories
        self._responses = {}

        for i, story in enumerate(stories):
            if i % args.batch_size == 0:
                memory_size = max(1, min(args.memory_size, len(story)))
            story_sentences = []    # Encoded Sentences of Single Story
            sentence_sizes = []     # List of lengths of each sentence of a Single Story
            word_sizes = []         # 2D List of word lengths of sentences of a Single Story
            tokens = []             # 3D list of character tokens of words of a Single Story
            story_string = []       # Readable Sentences of a Single Story
            oov_ids = []            # The ids of words in OOV index for copy
            oov_words = []          # The OOV words in a Single Story

            self._responses[i] = []
            for sentence in story:
                pad = max(0, glob['sentence_size'] - len(sentence))
                story_sentences.append([glob['word_idx'][w] if w in glob['word_idx'] else UNK_INDEX for w in sentence] + [0] * pad)
                sentence_sizes.append(len(sentence))
                word_tokens = [self._tokenize(w) for w in sentence] + [[]] * pad
                word_sizes.append([len(w) for w in word_tokens])
                tokens.append(word_tokens)
                story_string.append([str(x) for x in sentence] + [''] * pad)

                oov_sentence_ids = []
                for w in sentence:
                    if w not in glob['decode_idx']:
                        if w not in oov_words:
                            oov_sentence_ids.append(self._decode_vocab_size + len(oov_words))
                            oov_words.append(w)
                        else:
                            oov_sentence_ids.append(self._decode_vocab_size + oov_words.index(w))
                    else:
                        oov_sentence_ids.append(glob['decode_idx'][w])
                oov_sentence_ids = oov_sentence_ids + [PAD_INDEX] * pad
                oov_ids.append(oov_sentence_ids)

            # take only the most recent sentences that fit in memory
            if len(story_sentences) > args.memory_size:
                story_sentences = story_sentences[::-1][:args.memory_size][::-1]
                sentence_sizes = sentence_sizes[::-1][:args.memory_size][::-1]
                word_sizes = word_sizes[::-1][:args.memory_size][::-1]
                tokens = tokens[::-1][:args.memory_size][::-1]
                story_string = story_string[::-1][:args.memory_size][::-1]
                oov_ids = oov_ids[::-1][:args.memory_size][::-1]
            else: # pad to memory_size
                mem_pad = max(0, memory_size - len(story_sentences))
                for _ in range(mem_pad):
                    story_sentences.append([0] * glob['sentence_size'])
                    sentence_sizes.append(0)
                    word_sizes.append([0] * glob['sentence_size'])
                    tokens.append([[]] * glob['sentence_size'])
                    story_string.append([''] * glob['sentence_size'])
                    oov_ids.append([0] * glob['sentence_size'])

            self._stories.append(np.array(story_sentences))
            self._story_lengths.append(len(story))
            self._story_sizes.append(np.array(sentence_sizes))
            self._story_word_sizes.append(np.array(word_sizes))
            self._story_tokens.append(tokens)
            self._read_stories.append(np.array(story_string))
            self._oov_ids.append(np.array(oov_ids))
            self._oov_sizes.append(np.array(len(oov_words)))
            self._oov_words.append(oov_words)
            
        self._token_size = 0
        for size in self._story_word_sizes: 
            self._token_size = max(np.amax(np.amax(size)), self._token_size)
        
        Padded_Word_Tokens = []
        for story in self._story_tokens:
            pad_stories = []
            for token in story:
                pad_token = []
                for token_list in token:
                    token_list = token_list + [0]*(self._token_size - len(token_list))
                    pad_token.append(token_list)
                pad_stories.append(pad_token)
            Padded_Word_Tokens.append(np.array(pad_stories))
        self._story_tokens = Padded_Word_Tokens

    def _vectorize_queries(self, queries, glob):
        '''
            Maps each query into word and character tokens and assigns them ids
        '''  
        self._queries = [] 
        self._query_sizes = []
        self._query_word_sizes = []
        self._query_tokens = []
        self._read_queries = []

        for i, query in enumerate(queries):
            pad = max(0, glob['sentence_size'] - len(query))
            query_sentence = [glob['word_idx'][w] if w in glob['word_idx'] else UNK_INDEX for w in query] + [0] * pad
            tokens = [self._tokenize(w) for w in query] + [[]] * pad
            token_size = [len(w) for w in tokens]

            self._queries.append(np.array(query_sentence))
            self._query_sizes.append(np.array([len(query)]))
            self._query_word_sizes.append(np.array(token_size))
            self._query_tokens.append(tokens)
            self._read_queries.append(' '.join([str(x) for x in query]))

        Padded_Word_Tokens = []
        for token_list in self._query_tokens:
            Padded_Word_Tokens.append(np.array([token + [0]*(self._token_size - len(token)) for token in token_list]))
        self._query_tokens = Padded_Word_Tokens

    def _vectorize_answers(self, answers, glob):
        '''
            Maps each story into word tokens and assigns them ids
        '''   
        self._answers = []
        self._answer_sizes = []
        self._read_answers = []
        self._answers_emb_lookup = []

        for i, answer in enumerate(answers):
            pad = max(0, glob['candidate_sentence_size'] - len(answer) - 1)
            answer_sentence = []
            a_emb_lookup = []
            for w in answer:
                if w in glob['decode_idx']:
                    answer_sentence.append(glob['decode_idx'][w])
                    a_emb_lookup.append(glob['decode_idx'][w])
                elif w in self._oov_words[i]:
                    answer_sentence.append(self._decode_vocab_size + self._oov_words[i].index(w))
                    a_emb_lookup.append(UNK_INDEX)
                else:
                    answer_sentence.append(UNK_INDEX)
                    a_emb_lookup.append(UNK_INDEX)
            answer_sentence = answer_sentence + [EOS_INDEX] + [PAD_INDEX] * pad
            a_emb_lookup = a_emb_lookup + [EOS_INDEX] + [PAD_INDEX] * pad
            self._answers.append(np.array(answer_sentence))
            self._answer_sizes.append(np.array([len(answer)+1]))
            self._read_answers.append(' '.join([str(x) for x in answer]))
            self._answers_emb_lookup.append(np.array(a_emb_lookup))

    def _populate_entity_set(self, stories, answers):
        '''
            Create a set of all entity words seen
        '''
        entity_set = set()                  # Maintain a set of entities seen
        for story in stories:
            for sentence in story:
                if '$db' in sentence:
                    for w in sentence[:-2]:
                        if w not in entity_set:
                            entity_set.add(w)
                                
        for answer in answers:
            if 'api_call' in answer:
                for w in answer[1:]:
                    if w not in entity_set:
                        entity_set.add(w)
        return entity_set

    def _intersection_set_mask(self, answers, entity_set, glob):
        '''
            Create a mask which tracks the postions to copy a DB word
        '''
        mask = []
        for i, answer in enumerate(answers):
            vocab = set(answer).intersection(entity_set)
            dialog_mask = [0.0 if (x in vocab or x not in glob['idx_decode']) else 1.0 for x in answer]
            mask.append(np.array(dialog_mask))
        return mask

    def _get_entity_indecies(self, read_answers, entity_set):
        '''
            Get list of entity indecies in each Dialog Response
        '''
        return [np.array([i for i, word in enumerate(ans.split()) if word in entity_set ]) for ans in read_answers]
    

class Batch(Data):

    def __init__(self, data, start, end, args):

        self._stories = data.stories[start:end]

        self._queries = data.queries[start:end]

        self._answers = data.answers[start:end]

        self._answers_emb_lookup = data.answers_emb_lookup[start:end]

        self._story_sizes = data.story_sizes[start:end]

        self._query_sizes = data.query_sizes[start:end]

        self._answer_sizes = data.answer_sizes[start:end]

        self._story_tokens = data.story_tokens[start:end]
        
        self._token_size = data.token_size

        self._query_tokens = data.query_tokens[start:end]

        self._story_word_sizes = data.story_word_sizes[start:end]

        self._query_word_sizes = data.query_word_sizes[start:end]

        self._read_stories = data.readable_stories[start:end]

        self._read_queries = data.readable_queries[start:end]

        self._read_answers = data.readable_answers[start:end]

        self._oov_ids = data.oov_ids[start:end]

        self._oov_sizes = data.oov_sizes[start:end]

        self._oov_words = data.oov_words[start:end]

        self._dialog_ids = data.dialog_ids[start:end]

        self._intersection_set = data.intersection_set[start:end]

        self._entities = data.entities[start:end]

        self._entity_set = data.entity_set

        self._char_set = data.char_set

        if args.word_drop:
            self._stories, self._story_tokens = self._all_db_to_unk(self._stories, self._story_tokens, data.db_vocab_id, args.word_drop_prob)

    def _all_db_to_unk(self, stories, story_tokens, db_vocab_id, word_drop_prob):
        '''
            Perform Entity-Dropout on stories and story tokens
        '''
        new_stories = []
        new_stories_tokens = []
        replace_index = dict()
        
        for k, story in enumerate(stories):
            new_story = story.copy()
            for i in range(new_story.shape[0]):
                if db_vocab_id not in new_story[i]:
                    for j in range(new_story.shape[1]):
                        if new_story[i][j] in self._entity_set:
                            sample = random.uniform(0,1)
                            if sample < word_drop_prob:
                                new_story[i][j] = UNK_INDEX
                                if k in replace_index:
                                    replace_index[k].append((i,j))
                                else:
                                    replace_index[k] = [(i,j)]
            new_stories.append(new_story)
        UNK_token = self._tokenize('UNK') 
        UNK_token += [0]*(self._token_size - len(UNK_token))
        for i, story_token in enumerate(story_tokens):
            if i in replace_index:
                new_story_token = story_token.copy()
                for x,y in replace_index[i]:
                    new_stories_tokens[x][y] = UNK_token
                new_stories_tokens.append(new_story_token) 
            else:
                new_stories_tokens.append(story_token) 
        return new_stories, new_stories_tokens
    
    