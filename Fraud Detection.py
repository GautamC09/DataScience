import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/Users/gauta/OneDrive/Documents/DS/DataSet/Dataset(Data Science Internship)-20240619T055654Z-001/Dataset(Data Science Internship)/Final Project - Fraud Detection/creditcard.csv")
pd.set_option('display.max_columns', None)
print(df.info())

#Dealing with Missing Values
print(df.isnull().sum())

#Data Visualization
'''
for column in df.columns:
    if df[column].dtype in [np.float64, np.int64]:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.show()
'''
sns.countplot(x='Class', data=df)
plt.title('Distribution of Target Variable (Class)')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Count')
plt.show()

plt.hist(df["Amount"])
plt.show()

X = df.drop(['Time','Class'], axis=1)
Y=df['Class']

#Dealing with Imbalance
from imblearn.over_sampling import SMOTE
from collections import Counter#Synthetic Minority Oversampling (SMOTE) oversampling
sms=SMOTE(random_state=0)
X, Y=sms.fit_resample(X,Y)
print(Counter(Y))

'''
#Feature selection
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(4).plot(kind='barh')
plt.show()
'''

#1. logistic Regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr=LogisticRegression()
pca=PCA(n_components=2)

pca.fit(X)
X=pca.transform(X)

#print(X)

X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state=0,test_size=0.2)

logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
print("logistic Regression: ",accuracy_score(y_test,y_pred))

#2. Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score

nb=GaussianNB()

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0,test_size=0.4)

nb.fit(X_train,y_train)

y_pred1=nb.predict(X_test)

print("Naive Bayes: ",accuracy_score(y_test,y_pred1))

# 3. KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn=KNeighborsClassifier(n_neighbors=5)

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)

train=knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print("KNN: ",accuracy_score(y_test,y_pred))

#4. Decision Tree

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dt=tree.DecisionTreeClassifier()

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.3)

train=dt.fit(X_train,y_train)

y_pred=dt.predict(X_test)

print("Decision Tree: ",accuracy_score(y_test,y_pred))

# 5. Random Forest

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.3)

train=rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

print("Random Forest: ",accuracy_score(y_test,y_pred))

# 6. Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gbm=GradientBoostingClassifier(n_estimators=10)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.2)

gbm.fit(X_train,Y_train)

y_pred=gbm.predict(X_test)

print("GBM: ",accuracy_score(Y_test,y_pred))
