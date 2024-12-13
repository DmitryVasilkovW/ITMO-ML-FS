from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.svm import SVC

from fs.dataset.axis_repo import DataRepo


class FeatureSelectionAndModeling:
    __x_train = None
    __y_train = None
    __x_test = None
    __y_test = None

    __x_train_filter = None
    __x_test_filter = None

    __x_train_embedded = None
    __x_test_embedded = None

    __x_train_wrapper = None
    __x_test_wrapper = None

    @classmethod
    def __init(cls):
        cls.__x_train = DataRepo.get_axis('x', 'train')
        cls.__y_train = DataRepo.get_axis('y', 'train')
        cls.__x_test = DataRepo.get_axis('x', 'test')
        cls.__y_test = DataRepo.get_axis('y', 'test')

    @classmethod
    def __try_to_init(cls):
        if (
                cls.__x_train is None
                or cls.__y_train is None
                or cls.__x_test is None
                or cls.__y_test is None):
            cls.__init()

    @classmethod
    def __try_init_filter(cls):
        cls.__try_to_init()
        if cls.__x_train_filter is None or cls.__x_test_filter is None:
            filter_selector = SelectKBest(chi2, k=30)
            cls.__x_train_filter = filter_selector.fit_transform(cls.__x_train, cls.__y_train)
            cls.__x_test_filter = filter_selector.transform(cls.__x_test)

    @classmethod
    def get_x_train_filter(cls):
        cls.__try_init_filter()
        return cls.__x_train_filter

    @classmethod
    def get_x_test_filter(cls):
        cls.__try_init_filter()
        return cls.__x_test_filter

    @classmethod
    def __try_init_embedded(cls):
        cls.__try_to_init()
        if cls.__x_train_embedded is None or cls.__x_test_embedded is None:
            rf = RandomForestClassifier(random_state=42)
            rf.fit(cls.__x_train, cls.__y_train)
            importance = rf.feature_importances_
            top_features_rf = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:30]
            cls.__x_train_embedded = cls.__x_train[:, top_features_rf]
            cls.__x_test_embedded = cls.__x_test[:, top_features_rf]

    @classmethod
    def get_x_train_embedded(cls):
        cls.__try_init_embedded()
        return cls.__x_train_embedded

    @classmethod
    def get_x_test_embedded(cls):
        cls.__try_init_embedded()
        return cls.__x_test_embedded

    @classmethod
    def __try_init_wrapper(cls):
        cls.__try_to_init()
        if cls.__x_train_wrapper is None or cls.__x_test_wrapper is None:
            svc = SVC(kernel='linear', random_state=42)
            rfe_selector = RFE(estimator=svc, n_features_to_select=30)
            cls.__x_train_wrapper = rfe_selector.fit_transform(cls.__x_train, cls.__y_train)
            cls.__x_test_wrapper = rfe_selector.transform(cls.__x_test)

    @classmethod
    def get_x_test_wrapper(cls):
        cls.__try_init_wrapper()
        return cls.__x_test_wrapper

    @classmethod
    def get_x_train_wrapper(cls):
        cls.__try_init_wrapper()
        return cls.__x_train_wrapper
