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
                 sentence_size, 
                 batch_size,
                 max_memory_size, 
                 decoder_vocab, 
                 candidate_sentence_size,
                 char_emb_length,
                 char_overlap):

        self._decode_vocab_size = len(decoder_vocab)
        self._stories_ext, self._queries_ext, self._answers_ext, self._dialog_ids = \
            self._extract_data_items(data)
        self._db_values_set = self._get_db_values_set(self._stories_ext, word_idx)
        self._stories, self._story_sizes, self._story_tokens, self._story_word_sizes, self._read_stories, self._oov_ids, self._oov_sizes, self._oov_words = \
            self._vectorize_stories(self._stories_ext, word_idx, sentence_size, batch_size, self._decode_vocab_size, max_memory_size, decoder_vocab, char_emb_length, char_overlap)
        self._queries, self._query_sizes, self._query_tokens, self._query_word_sizes, self._read_queries = \
            self._vectorize_queries(self._queries_ext, word_idx, sentence_size, char_emb_length, char_overlap)
        # Jan 6 : added answers with UNKs
        self._answers, self._answer_sizes, self._read_answers, self._answers_emb_lookup = \
            self._vectorize_answers(self._answers_ext, decoder_vocab, candidate_sentence_size, self._oov_words, self._decode_vocab_size)

    @property
    def stories(self):
        return self._stories

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
    def dialog_ids(self):
        return self._dialog_ids

    @property
    def decode_vocab_size(self):
        return self._decode_vocab_size

    @property
    def db_values_set(self):
        return self._db_values_set
    
    def _get_db_values_set(self, stories, word_idx):
        inv_db_values_idx = set()
        # for i, story in enumerate(stories):
        #     for k, sentence in enumerate(story, 1):
        #         if '$db' in sentence:
        #             for w in sentence[:-2]:
        #                 if not w.startswith('r_'):
        #                     inv_db_values_idx.add(word_idx[w])
        return inv_db_values_idx

    def _extract_data_items(self, data):
        data.sort(key=lambda x:len(x[0]),reverse=True)
        stories = [x[0] for x in data]
        queries = [x[1] for x in data]
        answers = [x[2] for x in data]
        dialog_id = [x[3] for x in data]
        return stories, queries, answers, dialog_id

    def _index(token, size):
        index_list = [ord(c) for c in s]
        index = 0
        for i in range(size):
            index = index*(10*size) + index_list[i]
        return index

    def _tokenize(word, size, overlap):
        tokens = []
        start = 0
        end = size-1
        if overlap:
            while word < size:
                word += " " 
        else:
            while (word % size) < size:
                word += " "
            
        while end < len(word):
            token = self._index(word[start:end])
            tokens.append(token)
            if overlap:
                start += 1; end += 1
            else:
                start += size; end += size

        return tokens


    def _vectorize_stories(self, stories, word_idx, sentence_size, batch_size, decode_vocab_size, max_memory_size, decoder_vocab, char_emb_length, char_overlap):
        S = []
        SZ = []
        Word_tokens = []
        SWZ = []
        S_in_readable_form = []
        OOV_ids = []
        OOV_size = []
        OOV_words = []

        for i, story in enumerate(stories):
            if i % batch_size == 0:
                memory_size = max(1, min(max_memory_size, len(story)))
            ss = []
            sizes = []
            tokens = []
            word_sizes = []
            story_string = []
            oov_ids = []
            oov_words = []

            # Jan 6 : changed index to k
            for k, sentence in enumerate(story, 1):
                ls = max(0, sentence_size - len(sentence))
                # Jan 6 : words not in vocab are changed from NIL to UNK
                ss.append([word_idx[w] if w in word_idx else UNK_INDEX for w in sentence] + [0] * ls)
                sizes.append(len(sentence))
                word_tokens = [self._tokenize(w) for w in sentence] + [] * ls
                tokens.append(word_tokens)
                word_sizes.append([len(w) for w in word_tokens])
                story_element = ' '.join([str(x) for x in sentence[:-2]])
                story_string.append(' '.join([str(x) for x in sentence[-2:]]) + ' : ' + story_element)

                oov_sentence_ids = []
                for w in sentence:
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
            oov_ids = oov_ids[::-1][:memory_size][::-1]
            sizes = sizes[::-1][:memory_size][::-1]
            tokens = tokens[::-1][:memory_size][::-1]
            word_sizes = word_sizes[::-1][:memory_size][::-1]

            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * sentence_size)
                oov_ids.append([0] * sentence_size)
                sizes.append(0)
                word_sizes.append([0] * sentence_size)
                tokens.append([] * sentence_size)

            S.append(np.array(ss))
            SZ.append(np.array(sizes))
            Word_tokens.append(tokens)
            SWZ.append(np.array(word_sizes))
            S_in_readable_form.append(story_string)
            OOV_ids.append(np.array(oov_ids))
            OOV_size.append(np.array(len(oov_words)))
            OOV_words.append(np.array(oov_words))

        return S, SZ, Word_tokens, SWZ, S_in_readable_form, OOV_ids, OOV_size, OOV_words

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
            tokens = [self._tokenize(w) for w in query] + [] * lq
            qw = [len(w) for w in tokens]

            Q.append(np.array(q))
            QZ.append(np.array([len(query)]))
            Word_tokens.append(tokens)
            QWZ.append(np.array(qw))
            Q_in_readable_form.append(' '.join([str(x) for x in query]))

        return Q, QZ, Word_tokens, QWZ, Q_in_readable_form

    def _vectorize_answers(self, answers, decoder_vocab, candidate_sentence_size, OOV_words, decode_vocab_size):
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

class Batch(Data):

    def __init__(self, data, start, end, unk_size=0, word_drop=False):

        self._unk_size = unk_size

        self._stories = data.stories[start:end]

        self._queries = data.queries[start:end]

        self._answers = data.answers[start:end]

        # Jan 6 : added answers with UNKs
        self._answers_emb_lookup = data.answers_emb_lookup[start:end]

        if word_drop:
            self._stories, self._queries = self._random_unk(self._stories, self._queries, self._answers, data.db_values_set)

        self._story_sizes = data.story_sizes[start:end]

        self._query_sizes = data.query_sizes[start:end]

        self._answer_sizes = data.answer_sizes[start:end]

        self._story_word_sizes = data.story_word_sizes[start:end]

        self._query_word_sizes = data.query_word_sizes[start:end]

        self._story_tokens = self._pad_tokens_story(data.story_tokens[start:end], self._story_word_sizes)

        self._query_tokens = self._pad_tokens_query(data.query_tokens[start:end], self._query_word_sizes)

        self._read_stories = data.readable_stories[start:end]

        self._read_queries = data.readable_queries[start:end]

        self._read_answers = data.readable_answers[start:end]

        self._oov_ids = data.oov_ids[start:end]

        self._oov_sizes = data.oov_sizes[start:end]

        self._dialog_ids = data.dialog_ids[start:end]

    # Jan 8 : randomly make a few words in the input as UNK
    def _random_unk(self, stories, queries, answers, db_values_set):

        new_stories = []
        new_queries = []
        
        for story, query, answer in zip(stories, queries, answers):
            vocab = self._get_input_output_intersection_vocab(story, query, answer)
            coin_toss = random.randint(0,1)
            if coin_toss == 0 or len(vocab) == 0:
                sampled_words = []
            else:
                sampled_words = list(map(lambda _: random.choice(list(vocab)), range(min(self._unk_size,len(vocab)))))
            for element in sampled_words:
                story[story == element] = UNK_INDEX
                query[query == element] = UNK_INDEX
            new_stories.append(story)
            new_queries.append(query)

        return new_stories, new_queries

    def _get_input_output_intersection_vocab(self, story, query, answer):
        
        input_vocab = set(list(chain.from_iterable(story.tolist())) + query.tolist())
        output_vocab = set(answer.tolist())
        vocab = input_vocab.intersection(output_vocab)
        if 0 in vocab: vocab.remove(0)
        return vocab

    def _pad_tokens_story(tokens, token_sizes):
        max_token_size = 0
        for size in token_sizes:
            token_size = np.amax(size)
            if token_size > max_token_size:
                max_token_size = token_size
        padded_tokens = []
        for story in tokens:
            pad_stories = []
            for token in tokens:
                pad_token = []
                for token_list in token:
                    token_list = token_list + [0]*(max_token_size - len(token_list))
                    pad_token.append(token_list)
                pad_stories.append(pad_token)
            padded_tokens.append(np.array(pad_stories))
        return padded_tokens

    def _pad_tokens_query(tokens, token_sizes):
        max_token_size = 0
        for size in token_sizes:
            token_size = np.amax(size)
            if token_size > max_token_size:
                max_token_size = token_size
        padded_tokens = []
        for token in tokens:
            pad_token = []
            for token_list in token:
                token_list = token_list + [0]*(max_token_size - len(token_list))
                pad_token.append(token_list)
            padded_tokens.append(np.array(pad_token))
        return padded_tokens

