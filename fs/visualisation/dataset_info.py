from fs.dataset.axis_repo import DataRepo
from fs.dataset.data import get_data
from fs.vector.vector import Vectorize


def show_dataset_info():
    data = get_data()

    print('Dataset content')
    print(data.head())
    print()

    x_train = DataRepo.get_axis('x', 'train')
    n_samples, n_features = x_train.shape
    print('training sample size')
    print(f'Number of samples: {n_samples}, number of features: {n_features}')
    print()

    x_test = DataRepo.get_axis('x', 'test')
    n_samples, n_features = x_test.shape
    print('testing sample size')
    print(f'Number of samples: {n_samples}, number of features: {n_features}')
    print()

    n_samples, n_features = Vectorize.get_x().shape
    print('vector data size')
    print(f'Number of samples: {n_samples}, number of features: {n_features}')
    print()

