import pandas as pd
import seaborn as sns
import logging
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.feature_selection import RFE

from sklearn.metrics import root_mean_squared_error

from sklearn.preprocessing import MinMaxScaler