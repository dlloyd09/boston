import numpy
import pandas
import matplotlib
import scipy
import sklearn

from matplotlib import pyplot
from sklearn.datasets import load_boston

def main():
    boston = load_boston()

    # create a pandas dataframe of the boston data
    frame = pandas.DataFrame(boston.data)
    print(frame)

    # we could instead just print the first few rows
    print(frame.head())

    # we could also give the columns names instead of numbers
    frame.columns = boston.feature_names
    print(frame.head())

    # and we can generate some summary data!
    print(frame.describe())

    # or we can isolate just a few columns that we care about, rather than all 13
    truncated = frame[frame.columns[8:10]]
    print(truncated.describe())

if __name__ == "__main__":
    main()
