import numpy as np

UNK_INDEX = 0
GO_SYMBOL_INDEX = 1
EOS_INDEX = 2

class Data(object):

    def __init__(self,
                 data, 
                 word_idx, 
                 sentence_size, 
                 batch_size, 
                 candidates_size, 
                 max_memory_size, 
                 decoder_vocab, 
                 candidate_sentence_size):

        self._stories_ext, self._queries_ext, self._answers_ext, self._dialog_ids = self._extract_data_items(data)
        self._stories, self._story_sizes, self._read_stories, self._oov_ids, self._oov_sizes = self._vectorize_stories(self._stories_ext, word_idx, sentence_size, batch_size, candidates_size, max_memory_size, decoder_vocab)
        self._queries, self._query_sizes, self._read_queries = self._vectorize_queries(self._queries_ext, word_idx, sentence_size)
        self._answers, self._answer_sizes, self._read_answers = self._vectorize_answers(self._answers_ext, decoder_vocab, candidate_sentence_size)

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
    def answer_sizes(self):
        return self._answer_sizes

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

    def _extract_data_items(self, data):
        data.sort(key=lambda x:len(x[0]),reverse=True)
        stories = [x[0] for x in data]
        queries = [x[1] for x in data]
        answers = [x[2] for x in data]
        dialog_id = [x[3] for x in data]
        return stories, queries, answers, dialog_id

    def _vectorize_stories(self, stories, word_idx, sentence_size, batch_size, candidates_size, max_memory_size, decoder_vocab):
        S = []
        SZ = []
        S_in_readable_form = []
        OOV_ids = []
        OOV_size = []

        for i, story in enumerate(stories):
            if i % batch_size == 0:
                memory_size = max(1, min(max_memory_size, len(story)))
            ss = []
            sizes = []
            story_string = []
            oov_ids = []
            oov_words = []

            for i, sentence in enumerate(story, 1):
                ls = max(0, sentence_size - len(sentence))
                ss.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)
                sizes.append(len(sentence))

                story_element = ' '.join([str(x) for x in sentence[:-2]])
                story_string.append(' '.join([str(x) for x in sentence[-2:]]) + ' : ' + story_element)

                for w in sentence:
                    if w not in word_idx:
                        if w not in oov_words:
                            oov_ids.append(candidates_size + len(oov_words))
                            oov_words.append(w)
                        else:
                            oov_ids.append(candidates_size + oov_words.index(w))

            # take only the most recent sentences that fit in memory
            ss = ss[::-1][:memory_size][::-1]
            sizes = sizes[::-1][:memory_size][::-1]

            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * sentence_size)
                sizes.append(0)

            S.append(np.array(ss))
            SZ.append(np.array(sizes))
            S_in_readable_form.append(story_string)
            OOV_ids.append(np.array(oov_ids))
            OOV_size.append(np.array(len(oov_words)))

        return S, SZ, S_in_readable_form, OOV_ids, OOV_size

    def _vectorize_queries(self, queries, word_idx, sentence_size):
        Q = []
        QZ = []
        Q_in_readable_form = []

        for i, query in enumerate(queries):
            lq = max(0, sentence_size - len(query))
            q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq

            Q.append(np.array(q))
            QZ.append(np.array([len(query)]))
            Q_in_readable_form.append(' '.join([str(x) for x in query]))

        return Q, QZ, Q_in_readable_form

    def _vectorize_answers(self, answers, decoder_vocab, candidate_sentence_size):
        A = []
        AZ = []
        A_in_readable_form = []

        for i, answer in enumerate(answers):
            aq = max(0, candidate_sentence_size - len(answer) - 1)
            a = [decoder_vocab[w] if w in decoder_vocab else 0 for w in answer] + [EOS_INDEX] + [0] * aq

            A.append(np.array(a))
            AZ.append(np.array([len(answer)+1]))
            A_in_readable_form.append(' '.join([str(x) for x in answer]))

        return A, AZ, A_in_readable_form 
