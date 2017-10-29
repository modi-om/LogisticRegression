import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_dataset  = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

X_train = train_dataset.iloc[:, : -1].values
y_train = train_dataset.iloc[:, -1].values

X_test = test_dataset.iloc[:, : -1].values
y_test = test_dataset.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
acc = 100 * (1-(1.0 * (cm[0][1]+cm[1][0])/(cm[0][0]+cm[1][1])))
print "Accuracy of model is :--"
print(acc)