from sklearn.model_selection import train_test_split

from fs.vector.vector import Vectorize


class DataProcessor:
    def __init__(self, data, attribute: str):
        self.data = data
        self.X = self.data.drop([attribute], axis=1)
        self.y = self.data[attribute]
        x = Vectorize.get_x()
        y = Vectorize.get_y()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.3, random_state=42
        )

    def get(self, axis: str, data_type: str):
        if axis.lower() == "x":
            return self.__get_x(data_type)
        return self.__get_y(data_type)

    def __get_x(self, data_type):
        if data_type.lower() == "train":
            return self.X_train
        return self.X_test

    def __get_y(self, data_type):
        if data_type.lower() == "train":
            return self.y_train
        return self.y_test
