# -*- coding: utf-8 -*-
"""
Covariância é uma medida descritiva de associação linear entre duas variáveis.
Linear, porque tende a aproximar-se de uma recta quando esta é forte


Correlação (Pearson)
"""
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def cov(a, b):
    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a)-1)


# Negative correlation  (Perfect)
x=[5,4,3,2,1]
y=[1,2,3,4,5]

# Positive correlation (Perfect)
x=[1,2,3,4,5]
y=[1,2,3,4,5]


# Covariance == 0
x=[2,2,2,2,2]
y=[2,2,2,2,2]

# Negative correlation  (Perfect)
#x=range(1,100,1)
#y=range(100,1,-1)

# Positive correlation  (Perfect)
x=np.arange(1,100,1)
y=np.arange(1,50.5,0.5)

#Random
#x = np.random.normal(0,10,100)
#y = np.random.normal(0,10,100)

# weak (noise_level=10), strong (noise_level=1) or Perfect (noise_level=0, depending on the adder noise
noise_level=10
if noise_level>0:
    noise_x = np.random.normal(0,noise_level,len(x))
    noise_y = np.random.normal(0,noise_level,len(y))
    x=x+noise_x
    y=y+noise_y


model = LinearRegression()
model.fit(x.reshape((-1,1)), y)
r_sq = model.score(x.reshape((-1,1)), y)
print('coefficient of determination:', r_sq)
m=model.coef_
b=model.intercept_
print('m (model.coef_)=',model.coef_)
print('b (model.intercept_)=',model.intercept_)



print('x        y')
print('-------  ----------')
for vx,vy in zip(x,y):
    print('%.3f     %.3f' %(vx,vy))
c=cov(x,y)
print('cov=',c)



covMatrix=np.cov(x,y)
c=covMatrix[0][1]
corrMatrix=np.corrcoef(x, y)
r=corrMatrix[0][1]
print('np.cov=',c)
print('np.corrcoef=',r)

#sn.heatmap(covMatrix, annot=True, fmt='g')

plt.plot(x,y,'.')
plt.plot(x,x*m+b,label=(str(m)+'*x+'+str(b)))
plt.legend(loc='upper left')
plt.show()
