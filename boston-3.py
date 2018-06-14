import numpy
import pandas
import matplotlib
import scipy
import sklearn

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def main():
    boston = load_boston()

    # create a pandas dataframe of the boston data
    frame = pandas.DataFrame(boston.data)

    # create a set of predictors (first 13 cols) and a target (last col)
    predictors = frame
    mhv = boston.target

    # create from our large dataset a randomly-generated training set and a randomly-generated test set
    pr_train, pr_test, mhv_train, mhv_test = sklearn.model_selection.train_test_split(predictors, mhv, test_size = 0.25, random_state = 50)

    # see how large our data is
    print(pr_train.shape)
    print(pr_test.shape)
    print(mhv_train.shape)
    print(mhv_test.shape)

if __name__ == "__main__":
    main()
