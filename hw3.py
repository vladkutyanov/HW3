from math import log


class CountVectorizer:
    def __init__(self):
        pass

    def __fill_hash(self, corpus):
        """Заполняем временный словарь: {'слово': индекс}"""
        feature_hash = dict()
        index_counter = 0
        for text in corpus:
            for word in text.lower().split(' '):
                if word not in feature_hash:
                    feature_hash[word] = index_counter
                    index_counter += 1
        return feature_hash

    def fit_transform(self, corpus):
        """Вычисляем матрицу"""
        text_counter = 0
        self._feature_hash = self.__fill_hash(corpus)
        count_matrix_temp = [[0 for i in range(len(self._feature_hash))]
                             for i in range(len(corpus))]
        for text in corpus:
            for word in text.lower().split(' '):
                if word in self._feature_hash:
                    count_matrix_temp[text_counter][self._feature_hash[word]] += 1
            text_counter += 1
        return count_matrix_temp

    def get_feature_names(self):
        """Возвращаем список уникальных слов"""
        return list(self._feature_hash.keys())

    def term_freq(self, corpus):
        length = 0
        frequency_list = []
        matrix = self.fit_transform(corpus)
        for row in matrix:
            temp_frequency = []
            length = sum(row)
            for term in row:
                temp_frequency.append(round(term / length, 3))
            frequency_list.append(temp_frequency)
        return frequency_list

    def idf_transform(self, corpus):
        matrix = self.fit_transform(corpus)
        number_of_docs = len(corpus)
        idf_matrix = []
        idf_abs = [0 for i in range(len(matrix[0]))]
        for row in range(len(matrix)):
            for term in range(len(matrix[row])):
                if matrix[row][term] > 0:
                    idf_abs[term] += 1
        for item in idf_abs:
            idf_matrix.append(round(log((number_of_docs + 1)/(item + 1)) + 1, 3))
        return idf_matrix


class TfidfTransformer():
    def __init__(self):
        pass

    def fit_transform(self, tf_matrix, idf_matrix):
        tf_idf_matrix = []
        for row in tf_matrix:
            tf_idf_matrix.append([round(x * y, 2) for x, y in zip(row, idf_matrix)])
        return tf_idf_matrix


class TfidVectorizer(CountVectorizer):
    def __init__(self):
        super.__init__()
        self.tf_idf = TfidfTransformer()


corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste',
    ]
vect = CountVectorizer()
vect2 = TfidfTransformer()
count_matrix = vect.fit_transform(corpus)
print(vect.get_feature_names())
print(vect.term_freq(corpus))
print(vect.idf_transform(corpus))
