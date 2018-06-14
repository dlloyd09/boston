import numpy
import pandas
import matplotlib
import scipy
import sklearn

from matplotlib import pyplot
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    boston = load_boston()

    # create a pandas dataframe of the boston data
    frame = pandas.DataFrame(boston.data)

    # drop some columns that we suspect won't be as relevant
    ## CHAS, INDUS, NOX, AGE, B, PTRATIO
    predictors = frame.drop(labels = [2, 3, 4, 6, 10, 11], axis = 1)
    mhv = boston.target

    # create from our large dataset a randomly-generated training set and a randomly-generated test set
    pr_train, pr_test, mhv_train, mhv_test = sklearn.model_selection.train_test_split(predictors, mhv, test_size = 0.33, random_state = 50)

    # regression time!
    reg = LinearRegression()

    # fit our data and execute a prediction
    reg.fit(pr_train, mhv_train)
    mhv_pred = reg.predict(pr_test)

    # print out the r^2 value
    print(f"r^2 value: {reg.score(pr_test, mhv_test)}")

    # and plot it out!
    pyplot.scatter(mhv_test, mhv_pred)
    pyplot.xlabel("Actual listed value (in thousands)")
    pyplot.ylabel("Predicted value based on data (in thousands)")

    # and see how far off we were as mean squared error
    err = numpy.around(sklearn.metrics.mean_squared_error(mhv_test, mhv_pred), 3)
    pyplot.title(f"Predicted versus actual listed value\nfewer columns, 33% test set (MSE = {err})")

    # determine the trendline
    tr = numpy.polyfit(mhv_test, mhv_pred, 1)
    trendline = numpy.poly1d(tr)

    # plot the line, coloring it magenta
    pyplot.plot(mhv_test, trendline(mhv_test), "m-")
    pyplot.show()

if __name__ == "__main__":
    main()
