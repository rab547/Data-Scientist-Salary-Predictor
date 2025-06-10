import os
import sys
import random
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataAnalytics')))
from dataAnalysis import *
from dataVisualizer import *

file_path = 'Data/ds_salaries.csv'

def loadDataFrame(filePath):
    return pd.read_csv(filePath)

def test_manipulation():
    # PART A
    # df of only nan
    df = pd.DataFrame(np.nan, index = [0,1,2,3], columns=["a","b"])
    assert removeNullsWithoutReplacement(df).equals(pd.DataFrame({"a":[],"b":[]}))

    # df of some nan
    cols = {'Name': ["Jack", "Jill", np.nan, "Jane"],
            'Number': [10.0, 4.0, 25.0, np.nan]}
    df = pd.DataFrame(cols)
    rem_cols = {'Name': ["Jack", "Jill"],
                'Number': [10.0, 4.0]}
    assert removeNullsWithoutReplacement(df).equals(pd.DataFrame(rem_cols))

    # random df with no nan
    df = pd.DataFrame(np.random.randint(0,100,size=(10, 10)), columns=list('abcdefghij'))
    assert removeNullsWithoutReplacement(df).equals(df)

    # PART B
    # df of only nan
    df = pd.DataFrame(np.nan, index = [0,1,2,3], columns=["a","b"])
    zero_df = pd.DataFrame(np.nan, index = [0,1,2,3], columns=["a","b"])
    assert replaceNullsWithMedian(df).equals(zero_df)

    # df of some numerical nan
    cols = {'Number1': [5.5, 2.5, np.nan, 10.0],
            'Number2': [10.0, 4.0, 25.0, np.nan]}
    df = pd.DataFrame(cols)
    med_cols = {'Number1': [5.5, 2.5, 5.5, 10.0],
                'Number2': [10.0, 4.0, 25.0, 10.0]}
    assert replaceNullsWithMedian(df).equals(pd.DataFrame(med_cols))

    # random df with no nan
    df = pd.DataFrame(np.random.randint(0,100,size=(10, 10)), columns=list('abcdefghij'))
    assert replaceNullsWithMedian(df).equals(df)

    # PART C
    # column of unique values
    col = ['b','a','c']
    lbls = ['a', 'b', 'c']
    df = columnOneHotEncoding(col, lbls)
    onehot = {'a': [0,1,0],
              'b': [1,0,0],
              'c': [0,0,1]}
    assert df.equals(pd.DataFrame(onehot))
    
    # column of same value
    col = ['a','a','a','a']
    lbls = ['a']
    df = columnOneHotEncoding(col, lbls)
    assert df.equals(pd.DataFrame({'a': [1,1,1,1]}))
    
    # column of random values
    lbls = ['a','b', 'c', 'd', 'e', 'f', 'g']
    col = [random.choice(lbls) for _ in range(20)]
    df = columnOneHotEncoding(col, lbls)
    assert len(col) == df.sum().sum()

    # PART D
    df = loadDataFrame(file_path)
    stats = cleanDataAndReturnSummaryStatistics(df)

    # no nan values
    assert 0 == df.isna().sum().sum()
    # same number of columns
    assert len(df.columns) == len(stats)
    # each column has 2 or 5 statistics
    for name, col in stats.items():
        assert len(col) == 2 or len(col) == 5

# def test_visualization():
#     try:
#         show_plots()
#     except:
#         assert False # error occurred
#     assert True # all visualizations successful"