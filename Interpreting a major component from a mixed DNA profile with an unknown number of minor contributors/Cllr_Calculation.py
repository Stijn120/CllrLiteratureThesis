import lir
import pandas as pd
import numpy as np

# Read LR Data into Dataframe
df = pd.read_excel('STRMix_LRs.xlsx')
x = lir.util.to_odds(lir.util.from_log_odds_to_probability(df['Log10LR'].values))
y = df['Contributor'].replace(['True','False'], [1, 0]).values

# Calculate cllr and cllrmin and print
print('The log likelihood ratio cost is', lir.metrics.cllr(x, y), '(lower is better)')
print('The discriminative power is', lir.metrics.cllr_min(x, y), '(lower is better)')
