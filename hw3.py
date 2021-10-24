class CountVectorizer:
    def __init__(self):
        pass

    def __fill_hash(self, corpus):
        """Заполняем временный словарь: {'слово': индекс}"""
        feature_hash = dict()
        counter = 0
        for text in corpus:
            for word in text.lower().split(' '):
                if word not in feature_hash:
                    feature_hash[word] = counter
                    counter += 1
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


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste',
        'Pasta is a life'
    ]
vect = CountVectorizer()
count_matrix = vect.fit_transform(corpus)
assert vect.get_feature_names() == ['crock', 'pot', 'pasta', 'never', 'boil', 'again', 'pomodoro', 'fresh',
                                    'ingredients', 'parmesan', 'to', 'taste', 'is', 'a', 'life']
assert count_matrix == [[1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
print(vect.get_feature_names())
print(count_matrix)
