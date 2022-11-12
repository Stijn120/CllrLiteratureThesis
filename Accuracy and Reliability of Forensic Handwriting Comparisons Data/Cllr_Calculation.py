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

# Read Data into Dataframe
df = pd.read_excel('pnas.2119944119.sd02.xlsx')

# For each QKset, calculate probabilities of each conclusion
dummies = pd.get_dummies(df['Conclusion'])
dummies['QKset'] = df['QKset']
qkset_probs = dummies.groupby('QKset').mean()

# Transform probabilities into LRs
x = qkset_probs.drop(['NoConc', 'ProbNot', 'ProbWritten'], axis=1).values
x = np.array([x[1]/x[0] for x in x])
y = df.groupby('QKset')['Mating'].first().replace(['M', 'N'], [1, 0]).values

# Calculate and print Cllr and Cllrmin
print('The log likelihood ratio cost is', lir.metrics.cllr(x, y), '(lower is better)')
print('The discriminative power is', lir.metrics.cllr_min(x, y), '(lower is better)')