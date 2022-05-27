from submodulos.tokenizer.custom_tokenizer import SpacyCustomTokenizer, get_progressbar
from math import log


class ProbabilisticModel:
    def __init__(self) -> None:
        self.corpus = []
        self.inverted_index = {}
        self.term_to_index = {}
        self.N = 0
        self.pi = []

    def __add_ii__(self, dj, term):
        try:
            self.inverted_index[term].add(dj)
        except KeyError:
            self.inverted_index[term] = set([dj])

    def computing_independent_values(self):
        self.document_w_vector = []
        bar = get_progressbar(self.N, ' precomputing weights ')
        bar.start()
        for i in range(self.N):
            wj = [0] * len(self.inverted_index)
            for key in self.inverted_index:
                pi, ri = self.pi[self.term_to_index[key]], log(
                    self.N/len(self.inverted_index[key]))

                wj[self.term_to_index[key]] = log((pi*(1-ri) + 1)/(ri*(1-pi) + 1)) * \
                    1 if i in self.inverted_index[key] else 0
            self.document_w_vector.append(wj)
            bar.update(i + 1)
        bar.finish()

    def fit(self, text_list):
        self.corpus = text_list
        self.N = len(self.corpus)
        bar = get_progressbar(len(text_list), ' precomputing all values ')
        bar.start()
        nlp = SpacyCustomTokenizer()
        for i, text in enumerate(text_list):
            for token in nlp(text):
                if token.is_stop:
                    continue
                self.__add_ii__(i, token.lemma.lower(
                ) if not token.lemma is None else token.text.lower())
            bar.update(i + 1)
        bar.finish()

        bar = get_progressbar(len(text_list), ' find named entities ')
        bar.start()
        for i, text in enumerate(text_list):
            for token in nlp.__ents__(text):
                self.__add_ii__(i, token.text.lower())
            bar.update(i + 1)
        bar.finish()

        bar = get_progressbar(len(self.inverted_index),
                              ' numerate terms ')
        bar.start()
        for i, key in enumerate(self.inverted_index):
            self.term_to_index[key] = i
            bar.update(i + 1)
        bar.finish()

        self.pi = [0.5] * len(self.inverted_index)

        self.computing_independent_values()

    def sorted_and_find(self, query, recover_len=10):
        nlp = SpacyCustomTokenizer()
        query_term = set()
        for token in nlp(query):
            if token.is_stop:
                continue
            text = token.lemma if not token.lemma is None else token.text
            try: query_term.add(self.term_to_index[text.lower()])
            except: pass

        for token in nlp.__ents__(query):
            try: query_term.add(self.term_to_index[token.text.lower()])
            except: pass

        sim_result = []
        for i in range(self.N):
            sim_result.append((i, sum([self.document_w_vector[i][j] for j in query_term])))
        sim_result.sort(key=lambda x: x[1], reverse=True)

        _len_ = recover_len if self.N > recover_len else self.N
        return [self.corpus[sim_result[i][0]] for i in range(_len_)]
