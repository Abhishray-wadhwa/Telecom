import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('churn-bigml-20.csv')
dataset.sample(5)
X = dataset.drop(columns=['Churn'])
y = dataset['Churn']


le = LabelEncoder()
y = le.fit_transform(y)
x = le.fit_transform(X['International plan'])
x_1 = le.fit_transform(X['Voice mail plan'])

X = dataset.drop(columns=['Churn']).values
y = dataset['Churn'].values

X[:,3] = x
X[:,4] = x_1
x = X[:,1:]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)



import pickle
pickle.dump(classifier, open('model.pkl', 'wb'))

