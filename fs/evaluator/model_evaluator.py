from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from fs.dataset.axis_repo import DataRepo
from fs.params.feature_selection_and_modeling import FeatureSelectionAndModeling


class ModelEvaluator:
    __models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(kernel='linear', random_state=42),
        'Naive Bayes': MultinomialNB(),
    }

    __results = None

    __x_train = DataRepo.get_axis('x', 'train')
    __y_train = DataRepo.get_axis('y', 'train')
    __x_test = DataRepo.get_axis('x', 'test')
    __y_test = DataRepo.get_axis('y', 'test')

    __x_train_filter = FeatureSelectionAndModeling.get_x_train_filter()
    __x_test_filter = FeatureSelectionAndModeling.get_x_test_filter()

    __x_train_embedded = FeatureSelectionAndModeling.get_x_train_embedded()
    __x_test_embedded = FeatureSelectionAndModeling.get_x_test_embedded()

    __x_train_wrapper = FeatureSelectionAndModeling.get_x_train_wrapper()
    __x_test_wrapper = FeatureSelectionAndModeling.get_x_test_wrapper()

    @classmethod
    def __try_init_res(cls):
        if cls.__results is None:
            cls.__results = {}
            for name, model in cls.__models.items():
                model.fit(cls.__x_train, cls.__y_train)
                y_pred = model.predict(cls.__x_test)
                acc = accuracy_score(cls.__y_test, y_pred)
                cls.__results[name] = {'Without Feature Selection': acc}

                model.fit(cls.__x_train_filter, cls.__y_train)
                y_pred_filter = model.predict(cls.__x_test_filter)
                acc_filter = accuracy_score(cls.__y_test, y_pred_filter)
                cls.__results[name]['Filter'] = acc_filter

                model.fit(cls.__x_train_embedded, cls.__y_train)
                y_pred_embedded = model.predict(cls.__x_test_embedded)
                acc_embedded = accuracy_score(cls.__y_test, y_pred_embedded)
                cls.__results[name]['Embedded'] = acc_embedded

                model.fit(cls.__x_train_wrapper, cls.__y_train)
                y_pred_wrapper = model.predict(cls.__x_test_wrapper)
                acc_wrapper = accuracy_score(cls.__y_test, y_pred_wrapper)
                cls.__results[name]['Wrapper'] = acc_wrapper

    @classmethod
    def __init_results(cls):
        cls.__try_init_res()
        return cls.__results

    @classmethod
    def get_results(cls, models=None):
        if cls.__results is not None:
            old_models = cls.__models
            cls.__models = models
            cls.__results = None
            res = cls.__init_results()
            cls.__models = old_models
            return res
        return cls.__init_results()
