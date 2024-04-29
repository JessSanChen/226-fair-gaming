import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from aif360.datasets import GermanDataset # what we're using
import numpy as np

import pickle
import numpy as np
import time

from load_data import *

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier
from aif360.algorithms.inprocessing.celisMeta.utils import getStats
from aif360.algorithms.inprocessing import PrejudiceRemover

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler

# import matplotlib.pyplot as plt

#  DATA IMPORT
np.random.seed(226)

