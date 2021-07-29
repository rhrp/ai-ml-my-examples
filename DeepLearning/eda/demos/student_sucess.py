#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd

#load data from a CSV
dataset = pd.read_csv('../../../datasets/rhp/students_sucess.csv', header=0, sep=';')
dataset.head()


# Two variables analises
plt.figure(figsize=(12,8))
ds_tmp=dataset[dataset.Passed==1]
plt.plot(ds_tmp['AttendHours'],ds_tmp['StudyHours'],'*',label='Passed',color='Green')
ds_tmp=dataset[dataset.Passed==0]
plt.plot(ds_tmp['AttendHours'],ds_tmp['StudyHours'],'o',label='Reproved',color='Red')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Students' Success")
plt.xlabel("Attended hours in classes")
plt.ylabel("Hours of study at home")
plt.show()

# Three variables analises
fig = plt.figure(2, figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

ds_tmp=dataset[dataset.Passed==1]
ax.scatter(ds_tmp['AttendHours'],ds_tmp['StudyHours'],ds_tmp['PraticeEx'], c='Green',cmap=plt.cm.Set1, edgecolor='k', s=40)

ds_tmp=dataset[dataset.Passed==0]
ax.scatter(ds_tmp['AttendHours'],ds_tmp['StudyHours'],ds_tmp['PraticeEx'], c='Red',cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("Students' Success")
ax.set_xlabel("Attended hours in classes")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Hours of study at home")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Practical exercices")
ax.w_zaxis.set_ticklabels([])


plt.show()
# In[ ]:




