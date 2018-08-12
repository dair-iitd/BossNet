import numpy as np
import random
from itertools import chain

PAD_INDEX = 0
UNK_INDEX = 1
GO_SYMBOL_INDEX = 2
EOS_INDEX = 3


class Data(object):

    def __init__(self,
                 data, 
                 word_idx, 
                 idx_word,
                 sentence_size, 
                 batch_size,
                 max_memory_size, 
                 decoder_vocab, 
                 candidate_sentence_size,
                 char_emb_length,
                 char_overlap,
                 copy_first):

        self._decode_vocab_size = len(decoder_vocab)
        self._encoder_vocab = word_idx.keys()
        self._idx_word = idx_word
        if '$db' in word_idx:
            self._db_vocab_id = word_idx['$db']
        else:
            self._db_vocab_id = -1
        
        self._stories_ext, self._queries_ext, self._answers_ext, self._dialog_ids = \
            self._extract_data_items(data)
        self._stories, self._story_sizes, self._story_tokens, self._story_word_sizes, self._read_stories, self._oov_ids, self._oov_sizes, self._oov_words, self._token_size, self._story_positions, self._story_vocabs = \
            self._vectorize_stories(self._stories_ext, word_idx, sentence_size, batch_size, self._decode_vocab_size, max_memory_size, decoder_vocab, char_emb_length, char_overlap, copy_first)
        self._queries, self._query_sizes, self._query_tokens, self._query_word_sizes, self._read_queries = \
            self._vectorize_queries(self._queries_ext, word_idx, sentence_size, char_emb_length, char_overlap)
        # Jan 6 : added answers with UNKs
        self._answers, self._answer_sizes, self._read_answers, self._answers_emb_lookup = \
            self._vectorize_answers(self._answers_ext, decoder_vocab, candidate_sentence_size, self._oov_words, self._decode_vocab_size, copy_first)
        self._decode_to_encode_db_vocab_map, self._db_words_in_decoder_vocab, self._db_words_in_encoder_vocab, self._entity_map = self._populate_db_vocab_structures(self._stories_ext, self._answers_ext, word_idx, decoder_vocab)
        self._intersection_set = self._intersection_set_mask(self._answers, decoder_vocab)
        self._entities = self._get_entity_indecies(self._entity_map, self._read_answers)

        
    @property
    def stories(self):
        return self._stories

    @property
    def story_positions(self):
        return self._story_positions

    @property
    def queries(self):
        return self._queries

    @property
    def answers(self):
        return self._answers

    @property
    def story_sizes(self):
        return self._story_sizes

    @property
    def query_sizes(self):
        return self._query_sizes

    @property
    def answers_emb_lookup(self):
        return self._answers_emb_lookup

    @property
    def answer_sizes(self):
        return self._answer_sizes

    @property
    def story_tokens(self):
        return self._story_tokens

    @property
    def query_tokens(self):
        return self._query_tokens

    @property
    def story_word_sizes(self):
        return self._story_word_sizes

    @property
    def query_word_sizes(self):
        return self._query_word_sizes

    @property
    def readable_stories(self):
        return self._read_stories

    @property
    def readable_queries(self):
        return self._read_queries

    @property
    def readable_answers(self):
        return self._read_answers

    @property
    def oov_ids(self):
        return self._oov_ids

    @property
    def oov_sizes(self):
        return self._oov_sizes

    @property
    def oov_words(self):
        return self._oov_words

    @property
    def dialog_ids(self):
        return self._dialog_ids

    @property
    def decode_vocab_size(self):
        return self._decode_vocab_size

    @property
    def token_size(self):
        return self._token_size

    @property
    def intersection_set(self):
        return self._intersection_set

    @property
    def decode_to_encode_db_vocab_map(self):
        return self._decode_to_encode_db_vocab_map
        
    @property
    def db_words_in_decoder_vocab(self):
        return self._db_words_in_decoder_vocab

    @property
    def db_words_in_encoder_vocab(self):
        return self._db_words_in_encoder_vocab

    @property
    def entity_words(self):
        return self._entity_map

    @property
    def entities(self):
        return self._entities

    @property
    def db_vocab_id(self):
        return self._db_vocab_id

    @property
    def encoder_vocab(self):
        return self._encoder_vocab

    @property
    def idx2word(self):
        return self._idx_word
    
    def _populate_db_vocab_structures(self, stories, answers, word_idx, decoder_vocab):
        decode_to_encode_db_vocab_map = {}
        db_words_in_decoder_vocab = {}
        db_words_in_encoder_vocab = {}
        entity_map = []
        for _, story in enumerate(stories):
            for _, sentence in enumerate(story, 1):
                if '$db' in sentence:
                    for w in sentence[:-2]:
                        if w not in entity_map:
                            entity_map.append(w)
                        if not w.startswith('r_'):
                            if w in word_idx and w in decoder_vocab and decoder_vocab[w] not in decode_to_encode_db_vocab_map:
                                decode_to_encode_db_vocab_map[decoder_vocab[w]]=word_idx[w]
                                #print(w + " " + str(word_idx[w]) + " " + str(decoder_vocab[w]))
                            if w in word_idx:
                                db_words_in_encoder_vocab[word_idx[w]] = w
                            if w in decoder_vocab:
                                db_words_in_decoder_vocab[decoder_vocab[w]] = w
                                
        for _,answer in enumerate(answers):
            if 'api_call' in answer:
                for w in answer[1:]:
                    if w not in entity_map:
                        entity_map.append(w)
                    if w in word_idx and w in decoder_vocab and decoder_vocab[w] not in decode_to_encode_db_vocab_map:
                        decode_to_encode_db_vocab_map[decoder_vocab[w]]=word_idx[w]
                        #print(w + " " + str(word_idx[w]) + " " + str(decoder_vocab[w]))
                    if w in word_idx:
                        db_words_in_encoder_vocab[word_idx[w]] = w
                    if w in decoder_vocab:
                        db_words_in_decoder_vocab[decoder_vocab[w]] = w
                        
        #print(db_words_in_decoder_vocab)
        #print(db_words_in_encoder_vocab)
        return decode_to_encode_db_vocab_map, db_words_in_decoder_vocab, db_words_in_encoder_vocab, entity_map


    def _extract_data_items(self, data):
        data.sort(key=lambda x:len(x[0]),reverse=True)
        stories = [x[0] for x in data]
        queries = [x[1] for x in data]
        answers = [x[2] for x in data]
        dialog_id = [x[3] for x in data]
        return stories, queries, answers, dialog_id

    def _index(self, token, size):
        # if token in self._token_list:
        #     return self._token_list.index(token)
        # else:
        #     self._token_list += [token]
        #     return len(self._token_list)

        index_list = [ord(c) for c in token]
        index = 0
        for i in range(size):
            index = index*(256) + index_list[i]
        return index

    def _tokenize(self, word, size, overlap):
        tokens = []
        start = 0
        end = size
        if overlap:
            while len(word) < size:
                word += " " 
        else:
            while (len(word) % size) > 0:
                word += " "
            
        while end <= len(word):
            token = self._index(word[start:end], size)
            tokens.append(token)
            if overlap:
                start += 1; end += 1
            else:
                start += size; end += size

        return tokens

    def _vectorize_stories(self, stories, word_idx, sentence_size, batch_size, decode_vocab_size, max_memory_size, decoder_vocab, char_emb_length, char_overlap, copy_first):
        S = []
        SP = []
        SZ = []
        Word_tokens = []
        SWZ = []
        S_in_readable_form = []
        OOV_ids = []
        OOV_size = []
        OOV_words = []
        S_VOCAB = []

        for i, story in enumerate(stories):
            if i % batch_size == 0:
                memory_size = max(1, min(max_memory_size, len(story)))
            ss = []
            sps = []
            sizes = []
            tokens = []
            word_sizes = []
            story_string = []
            oov_ids = []
            oov_words = []
            vocab = set()

            # Jan 6 : changed index to k
            for k, sentence in enumerate(story, 1):
                ls = max(0, sentence_size - len(sentence))
                # Jan 6 : words not in vocab are changed from NIL to UNK
                ss.append([word_idx[w] if w in word_idx else UNK_INDEX for w in sentence] + [0] * ls)
                sps.append(list(np.arange(1,len(sentence)+1)) + [0] * ls)
                sizes.append(len(sentence))
                word_tokens = [self._tokenize(w, char_emb_length, char_overlap) for w in sentence] + [[]] * ls
                tokens.append(word_tokens)
                word_sizes.append([len(w) for w in word_tokens])
                story_element = [str(x) for x in sentence] + [''] * ls
                story_string.append(story_element)

                oov_sentence_ids = []
                
                #if copy_first:
                #   oov_sentence_ids.append(decode_vocab_size + len(oov_words))
                #   oov_words.append(SENTINAL_SURFACE_FORM)

                for w in sentence:
                    vocab.add(w)
                    if w not in decoder_vocab:
                        if w not in oov_words:
                            oov_sentence_ids.append(decode_vocab_size + len(oov_words))
                            oov_words.append(w)
                        else:
                            oov_sentence_ids.append(decode_vocab_size + oov_words.index(w))
                    else:
                        oov_sentence_ids.append(decoder_vocab[w])
                oov_sentence_ids = oov_sentence_ids + [PAD_INDEX] * ls
                oov_ids.append(oov_sentence_ids)


            # take only the most recent sentences that fit in memory
            ss = ss[::-1][:memory_size][::-1]
            sps = sps[::-1][:memory_size][::-1]
            oov_ids = oov_ids[::-1][:memory_size][::-1]
            sizes = sizes[::-1][:memory_size][::-1]
            tokens = tokens[::-1][:memory_size][::-1]
            word_sizes = word_sizes[::-1][:memory_size][::-1]
            story_string = story_string[::-1][:memory_size][::-1]

            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * sentence_size)
                sps.append([0] * sentence_size)
                oov_ids.append([0] * sentence_size)
                sizes.append(0)
                word_sizes.append([0] * sentence_size)
                tokens.append([[]] * sentence_size)
                story_string.append([''] * sentence_size)

            S.append(np.array(ss))
            SP.append(np.array(sps))
            SZ.append(np.array(sizes))
            Word_tokens.append(tokens)
            SWZ.append(np.array(word_sizes))
            S_in_readable_form.append(np.array(story_string))
            OOV_ids.append(np.array(oov_ids))
            OOV_size.append(np.array(len(oov_words)))
            OOV_words.append(np.array(oov_words))
            S_VOCAB.append(vocab)
            
        max_token_size = 0
        for size in SWZ:
            token_size = np.amax(np.amax(size))
            if token_size > max_token_size:
                max_token_size = token_size
        padded_tokens = []
        for story in Word_tokens:
            pad_stories = []
            for token in story:
                pad_token = []
                for token_list in token:
                    token_list = token_list + [0]*(max_token_size - len(token_list))
                    pad_token.append(token_list)
                pad_stories.append(pad_token)
            padded_tokens.append(np.array(pad_stories))

        return S, SZ, padded_tokens, SWZ, S_in_readable_form, OOV_ids, OOV_size, OOV_words, max_token_size, SP, S_VOCAB

    def _vectorize_queries(self, queries, word_idx, sentence_size, char_emb_length, char_overlap):
        Q = []
        QZ = []
        Word_tokens = []
        QWZ = []
        Q_in_readable_form = []

        for i, query in enumerate(queries):
            lq = max(0, sentence_size - len(query))
            # Jan 6 : words not in vocab are changed from NIL to UNK
            q = [word_idx[w] if w in word_idx else UNK_INDEX for w in query] + [0] * lq
            tokens = [self._tokenize(w, char_emb_length, char_overlap) for w in query] + [[]] * lq
            qw = [len(w) for w in tokens]

            Q.append(np.array(q))
            QZ.append(np.array([len(query)]))
            Word_tokens.append(tokens)
            QWZ.append(np.array(qw))
            Q_in_readable_form.append(' '.join([str(x) for x in query]))

        max_token_size = self._token_size
        padded_tokens = []
        for token in Word_tokens:
            pad_token = []
            for token_list in token:
                token_list = token_list + [0]*(max_token_size - len(token_list))
                pad_token.append(token_list)
            padded_tokens.append(np.array(pad_token))

        return Q, QZ, padded_tokens, QWZ, Q_in_readable_form

    def _vectorize_answers(self, answers, decoder_vocab, candidate_sentence_size, OOV_words, decode_vocab_size, copy_first):
        A = []
        AZ = []
        # Jan 6 : added answers with UNKs
        A_for_embeddding_lookup = []
        A_in_readable_form = []

        for i, answer in enumerate(answers):
            aq = max(0, candidate_sentence_size - len(answer) - 1)
            a = []
            a_emb_lookup = []
            for w in answer:
                if w in decoder_vocab:
                    a.append(decoder_vocab[w])
                    a_emb_lookup.append(decoder_vocab[w])
                elif w in OOV_words[i]:
                    a.append(decode_vocab_size + OOV_words[i].tolist().index(w))
                    a_emb_lookup.append(UNK_INDEX)
                else:
                    a.append(UNK_INDEX)
                    a_emb_lookup.append(UNK_INDEX)
            a = a + [EOS_INDEX] + [PAD_INDEX] * aq
            a_emb_lookup = a_emb_lookup + [EOS_INDEX] + [PAD_INDEX] * aq

            A.append(np.array(a))
            A_for_embeddding_lookup.append(np.array(a_emb_lookup))
            AZ.append(np.array([len(answer)+1]))
            A_in_readable_form.append(' '.join([str(x) for x in answer]))

        return A, AZ, A_in_readable_form, A_for_embeddding_lookup

    def _intersection_set_mask(self, answers, decoder_vocab):
        mask = []
        index = 0

        inv_index = {}
        for word, idx in decoder_vocab.items():
            inv_index[idx]=word

        db_vocab = set(self._db_words_in_decoder_vocab.keys())
        for answer in answers:
            '''
            story_words = self._story_vocabs[index]
            vocab = set()
            for word in story_words:
                if word in decoder_vocab:
                    vocab.add(decoder_vocab[word])
            '''
            vocab = set(answer).intersection(db_vocab)
            #print(answer)
            #for v in vocab:
            #    print(self._db_words_in_decoder_vocab[v])
            #print("------")

            dialog_mask = [0.0 if (x in vocab or x not in inv_index) else 1.0 for x in answer]
            mask.append(np.array(dialog_mask))
            index+=1
        return mask
    
    # the vocab returned is in encoder vocab space
    def _get_db_output_intersection_vocab(self, answer, decode_to_encode_db_vocab_map):
        output_decode_vocab = set(answer.tolist())
        vocab = set()
        for w in output_decode_vocab:
            if w in decode_to_encode_db_vocab_map:
                vocab.add(decode_to_encode_db_vocab_map[w])
        return vocab
    
    def _get_entity_indecies(self, entity_map, read_answers):
        entities = []
        for ans in read_answers:
            ent = []
            for i, word in enumerate(ans.split()):
                if word in entity_map:
                    ent.append(i)
            entities.append(ent)
        return entities

class Batch(Data):

    def __init__(self, data, start, end, unk_size=0, word_drop=False, word_drop_prob=0.0):

        self._unk_size = unk_size

        self._stories = data.stories[start:end]

        self._story_positions = data.story_positions[start:end]

        self._queries = data.queries[start:end]

        self._answers = data.answers[start:end]

        # Jan 6 : added answers with UNKs
        self._answers_emb_lookup = data.answers_emb_lookup[start:end]

        if word_drop:
            #self._stories, self._queries = self._random_unk(self._stories, self._queries, self._answers, data.encoder_vocab)
            self._stories, self._queries = self._all_db_to_unk(self._stories, self._queries, data.db_vocab_id, data.db_words_in_encoder_vocab, word_drop_prob)

        self._story_sizes = data.story_sizes[start:end]

        self._query_sizes = data.query_sizes[start:end]

        self._answer_sizes = data.answer_sizes[start:end]

        self._story_tokens = data.story_tokens[start:end]
        
        self._token_size = data.token_size

        if word_drop:
            self._story_tokens = self._get_char_drop_tokens(self._stories, data.idx2word)

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

    def _get_char_drop_tokens(self, stories, idx2word):
        Word_tokens = []
        for i, story in enumerate(stories):
            tokens = []
            for k, sentence in enumerate(story, 1):
                word_tokens = [self._tokenize(idx2word[w], 1, True) for w in sentence]
                tokens.append(word_tokens)
            Word_tokens.append(tokens)

        padded_tokens = []
        for story in Word_tokens:
            pad_stories = []
            for token in story:
                pad_token = []
                for token_list in token:
                    token_list = token_list + [0]*(self._token_size - len(token_list))
                    pad_token.append(token_list)
                pad_stories.append(pad_token)
            padded_tokens.append(np.array(pad_stories))
        return padded_tokens

    # Jan 8 : randomly make a few words in the input as UNK
    def _random_unk(self, stories, queries, answers, encoder_vocab):

        new_stories = []
        new_queries = []
        
        for story, query, answer in zip(stories, queries, answers):
            coin_toss = random.randint(0,1)
            if coin_toss == 0:
                sampled_words = []
            else:
                sampled_words = list(encoder_vocab)
            for element in sampled_words:
                story[story == element] = UNK_INDEX
                query[query == element] = UNK_INDEX
            new_stories.append(story)
            new_queries.append(query)

        return new_stories, new_queries
    
    def _all_db_to_unk(self, stories, queries, db_vocab_id, db_words_in_encoder_vocab,word_drop_prob):

        new_stories = []
        new_queries = []
        
        db_words = list(db_words_in_encoder_vocab.keys())
        
        for story, query in zip(stories, queries):
            new_story = story.copy()
            for i in range(new_story.shape[0]):
                if db_vocab_id not in new_story[i]:
                    for j in range(new_story.shape[1]):
                        if new_story[i][j] in db_words:
                            sample = random.uniform(0,1)
                            if sample < word_drop_prob:
                                new_story[i][j] = UNK_INDEX
            new_stories.append(new_story)
            new_queries.append(query)

        return new_stories, new_queries