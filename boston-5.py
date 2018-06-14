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

    # create a set of predictors (first 13 cols) and a target (last col)
    predictors = frame
    mhv = boston.target

    # create from our large dataset a randomly-generated training set and a randomly-generated test set
    pr_train, pr_test, mhv_train, mhv_test = sklearn.model_selection.train_test_split(predictors, mhv, test_size = 0.25, random_state = 50)

    # regression time!
    reg = LinearRegression()

    # fit our data and execute a prediction
    reg.fit(pr_train, mhv_train)
    mhv_pred = reg.predict(pr_test)

    # and plot it out!
    pyplot.scatter(mhv_test, mhv_pred)
    pyplot.xlabel("Actual listed value (in thousands)")
    pyplot.ylabel("Predicted value based on data (in thousands)")

    # and see how far off we were as mean squared error
    err = numpy.around(sklearn.metrics.mean_squared_error(mhv_test, mhv_pred), 3)
    pyplot.title(f"Predicted versus actual listed value (MSE = {err})")

    # determine the trendline
    tr = numpy.polyfit(mhv_test, mhv_pred, 1)
    trendline = numpy.poly1d(tr)

    # plot the line, coloring it magenta
    pyplot.plot(mhv_test, trendline(mhv_test), "m-")
    pyplot.show()

if __name__ == "__main__":
    main()
