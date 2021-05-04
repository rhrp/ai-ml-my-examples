# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
print(__doc__)

import matplotlib.pyplot as plt
import pandas as pd
import rhplib.datasets_pandas
import seaborn as sns

#load data from a CSV
dataset = pd.read_csv('../datasets/animals.csv', header=0, sep=';')
print(dataset.head())


# Histogram of features
rhplib.datasets_pandas.plot_hist_dataset(dataset,100,False)

# Violin
sns.set_theme(style="whitegrid")
for c in dataset.columns:
    if c not in ['name','lives in the water','Has legs']:
        fig = plt.figure()
        ax = sns.violinplot(x=c,y='lives in the water',data=dataset,hue='Has legs')
        
        
plt.show()