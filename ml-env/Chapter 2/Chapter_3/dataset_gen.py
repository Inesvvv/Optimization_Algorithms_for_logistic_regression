"""
File created 19/03/2026
This file aims to extract the data on marriage matching and create the features. 
I will later apply SISTA with newton method to determine the relevant features/parameters in marriage matching

Data obtained from Alfred Galichon, 'math+econ+code' masterclass on optimal transport and economic applications, January 2022. https://github.com/math-econ-code/mec_optim


"""

import pandas as pd
import numpy as np
from scipy import optimize
import scipy.sparse as spr
from sklearn import linear_model


thepath = 'https://raw.githubusercontent.com/math-econ-code/mec_optim_2021-01/master/data_mec_optim/marriage-ChooSiow/'

n_singles = pd.read_csv(thepath + 'n_singles.txt', sep='\t', header=None)
marr = pd.read_csv(thepath + 'marr.txt', sep='\t', header=None)
n_avail = pd.read_csv(thepath + 'n_avail.txt', sep='\t', header=None)

print("n_singles shape:", n_singles.shape)
print("marr shape:", marr.shape)
print("n_avail shape:", n_avail.shape)
print(marr.head())

