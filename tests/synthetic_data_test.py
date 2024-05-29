from src.data_process.synthetic_data import synData

import numpy as np
import matplotlib.pyplot as plt

# Create synthetic data
data = synData(100, 100, 2, 0.1, 0.1, 0.1, ictype='random')
indata ,out ,xr,yr= data.generate_training_data()
data.plot_data(xr, yr, out)