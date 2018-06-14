import numpy
import pandas
import matplotlib
import scipy
import sklearn

from sklearn.datasets import load_boston

def main():
    boston = load_boston()

    # examine the data value using its native "shape" method
    print(boston.data.shape)

if __name__ == "__main__":
    main()
