#1. logistic Regression
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


model=LogisticRegression()

df=pd.read_csv("C:/Users/gauta/OneDrive/Desktop/SEM_5/DS_Internship/Iris.csv")

x = df.drop('Id', axis=1)
x = x.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=1,test_size=0.3)

#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

model = LogisticRegression(max_iter=200)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

#print(train.coef_)
#print(train.intercept_)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

#2. Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score

nb=GaussianNB()

X2_train, X2_test, y2_train, y2_test = train_test_split(x, y, random_state=0,test_size=0.4)

nb.fit(X2_train,y2_train)

y_pred1=nb.predict(X2_test)

print("Naive Bayes: ",accuracy_score(y2_test,y_pred1))


# 3. KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn=KNeighborsClassifier(n_neighbors=5)

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0)

train=knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print("KNN: ",accuracy_score(y_test,y_pred))


#4. Decision Tree

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dt=tree.DecisionTreeClassifier()

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

train=dt.fit(X_train,y_train)

y_pred=dt.predict(X_test)

print("Decision Tree :",accuracy_score(y_test,y_pred))


# 5. Random Forest

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

train=rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

print("Random Forest: ",accuracy_score(y_test,y_pred))


# 6. Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gbm=GradientBoostingClassifier(n_estimators=10)

X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=0,test_size=0.2)

gbm.fit(X_train,Y_train)

y_pred=gbm.predict(X_test)

print("GBM: ",accuracy_score(Y_test,y_pred))