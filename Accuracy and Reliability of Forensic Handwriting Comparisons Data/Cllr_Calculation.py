import lir
import pandas as pd
import numpy as np

# Read Data into Dataframe
df = pd.read_excel('pnas.2119944119.sd02.xlsx')[['Mating', 'Conclusion', 'Outcome']]

# Get conclusions for all QKsets belonging to each hypothesis
h1s = df[df['Mating'] == 'M']['Conclusion']
h2s = df[df['Mating'] == 'N']['Conclusion']

# Calculate LRs for every conclusion category (i.e. how more likely is it that a category is concluded when h1 is true as opposed to when h2 is true)
LR_map = {kw: (h1s == kw).sum() / (h2s == kw).sum() for kw in df['Conclusion'].unique()}
print(LR_map)

# For every QKset, now replace discrete conclusion with continuous LR
h1s = np.array(h1s.map(LR_map))
h2s = np.array(h2s.map(LR_map))

# Calculate
stats = lir.calculate_lr_statistics(h2s, h1s)

# Calculate and print Cllr and Cllrmin
print('The log likelihood ratio cost is', stats.cllr, '(lower is better)')
print('The discriminative power is', stats.cllr_min, '(lower is better)')