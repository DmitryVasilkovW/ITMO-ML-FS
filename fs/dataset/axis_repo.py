from fs.dataset.data import get_data
from fs.dataset.select_data import DataProcessor


class DataRepo:
    __x_train = None
    __y_train = None
    __x_test = None
    __y_test = None
    __data = None
    __repo = None

    @classmethod
    def __get_x(cls, data_type: str):
        assert cls.__repo is not None and isinstance(cls.__repo, DataProcessor)

        if data_type == 'train':
            if cls.__x_train is None:
                cls.__x_train = cls.__repo.get("x", data_type)
            return cls.__x_train

        if data_type == 'test':
            if cls.__x_test is None:
                cls.__x_test = cls.__repo.get("x", data_type)
            return cls.__x_test

        return None

    @classmethod
    def __get_y(cls, data_type: str):
        assert cls.__repo is not None and isinstance(cls.__repo, DataProcessor)

        if data_type == 'train':
            if cls.__y_train is None:
                cls.__y_train = cls.__repo.get("y", data_type)
            return cls.__y_train

        if data_type == 'test':
            if cls.__y_test is None:
                cls.__y_test = cls.__repo.get("y", data_type)
            return cls.__y_test

        return None

    @classmethod
    def get_axis(cls, axis: str, data_type: str, attribute="Label"):
        if cls.__data is None:
            cls.__data = get_data()
        if cls.__repo is None:
            cls.__repo = DataProcessor(data=cls.__data, attribute=attribute)

        if axis == 'x':
            return cls.__get_x(data_type)
        if axis == 'y':
            return cls.__get_y(data_type)

        return None
