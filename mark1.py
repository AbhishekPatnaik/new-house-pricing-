
#------------prediction model on house pricing for various features ----------------------

import pandas as pd
from scipy.stats import skew
import numpy as np

data = pd.read_csv('/home/abhishek/Downloads/DATASETS/kc_house_data.csv')
data['id'] = data.id.astype('category').cat.codes
data['price'] = np.log(data['price'])

#data['date'] = data.loc[str(data['date'].split('T'))[0]]
print(data['date'])

for i in data.index:
    if data['id'].loc[i] == 17927:
        print(data.loc[i])

data = data.drop('date',axis=1)
print(skew(data['price']))

y = data['price']

data = data.drop(['price'],axis=1)


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data,y,test_size=0.25,random_state=42)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
print('training the LinearRegression model.......')
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
print('training RandomForestRegressor.....')
rf.fit(X_train,y_train)
print(rf.score(X_test,y_test))
print(X_test.head())
print(rf.predict([17927 ,5 ,3.00 ,2900 ,6730 , 1.0, 0,0 , 5 ,8 ,1830,1070  ,1977,0,98115 , 47.6784 ,-122.285 ,2370 ,6283  ]))

