import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from fs.dataset.data import get_data

russian_stop_words = ['и', 'в', 'на', 'для', 'с', 'по', 'о', 'как', 'к', 'что', 'это', 'не', 'да', 'все']


class Vectorize:
    __data = None
    __vectorizer = None
    __X = None
    __Y = None

    @classmethod
    def __init(cls):
        cls.__data = get_data()
        cls.__vectorizer = CountVectorizer(stop_words='english', max_features=1000)

    @classmethod
    def __init_x(cls):
        if cls.__vectorizer is None or cls.__data is None:
            cls.__init()
        x = cls.__vectorizer.fit_transform(cls.__data['Text']).toarray().astype(np.float32)
        cls.__X = x

    @classmethod
    def __init_y(cls):
        if cls.__vectorizer is None or cls.__data is None:
            cls.__init()
        cls.__Y = cls.__data['Label']

    @classmethod
    def get_x(cls):
        if cls.__X is None:
            cls.__init_x()
        return cls.__X

    @classmethod
    def get_y(cls):
        if cls.__Y is None:
            cls.__init_y()
        return cls.__Y
