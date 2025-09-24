import nfl_data_py as nfl
import pandas as pd
import os
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

data = nfl.import_pbp_data([2024])

print(data.shape)

print(data.head())

print(data.columns)