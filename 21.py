import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

df = pd.read_csv("./suri_jikken6_data/mmse_kadai14.txt",header=None)
data = np.array(df)

x = np.array(data[:,0])
y = np.array(data[:,1])

