import pandas as pd

from fs.evaluator.model_evaluator import ModelEvaluator
from fs.visualisation.plot import show_plot


def main():
    show_plot()

    res = ModelEvaluator.get_results()
    print(pd.DataFrame(res))


if __name__ == '__main__':
    main()
