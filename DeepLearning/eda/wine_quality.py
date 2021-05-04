# -*- coding: utf-8 -*-
print(__doc__)

import matplotlib.pyplot as plt
import pandas as pd
import rhplib.datasets_pandas
import seaborn as sns

# load Winequality dataset using pandas
dataset = pd.read_csv('../datasets/winequality-red.csv',delimiter=',',header=0)
print(dataset.head())

# Histogram of features
rhplib.datasets_pandas.plot_hist_dataset(dataset,100,False)

# Violin
sns.set_theme(style="whitegrid")
for c in dataset.columns:
    print(c)
    if c not in ['quality']:
        fig = plt.figure()
        ax = sns.violinplot(y=c,x='quality',data=dataset)
        ax.set_title('Violin Plot '+c)
plt.show()