import math
import lir
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Read LR Data into Dataframe
df = pd.read_excel('table_data_appendix.xlsx')
x_2d = df['Automated_2D'].replace(['Inf'], [np.inf]).astype(float).values
x_3d = df['Automated_3D'].replace(['Inf'], [np.inf]).astype(float).values
y = df['Ground_Truth'].replace(['SS','DS'], [1, 0]).values

# Calculate cllr and cllrmin and print
print('The 2D log likelihood ratio cost is', lir.metrics.cllr(x_2d, y), '(lower is better)')
print('The 2D discriminative power is', lir.metrics.cllr_min(x_2d, y), '(lower is better)')

print('The 3D log likelihood ratio cost is', lir.metrics.cllr(x_3d, y), '(lower is better)')
print('The 3D discriminative power is', lir.metrics.cllr_min(x_3d, y), '(lower is better)')
