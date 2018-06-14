import numpy
import pandas
import matplotlib
import scipy
import sklearn

from matplotlib import pyplot
from sklearn.datasets import load_boston

def main():
    boston = load_boston()
    print(boston)

if __name__ == "__main__":
    main()
