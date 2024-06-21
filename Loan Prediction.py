import pandas as pd

df=pd.read_csv("C:/Users/gauta/OneDrive/Documents/DS/DataSet/Dataset(Data Science Internship)-20240619T055654Z-001/Dataset(Data Science Internship)/Loan Prediction/train_u6lujuX_CVtuZ9i.csv")
df.info()

#Dealing with Missing Values
print(df.isnull().sum())
df['Gender'].fillna((df['Gender'].mode()[0]), inplace=True)
df['Married'].fillna((df['Married'].mode()[0]), inplace=True)
df['Dependents'].fillna((df['Dependents'].mode()[0]), inplace=True)
df['Self_Employed'].fillna((df['Self_Employed'].mode()[0]), inplace=True)
df['LoanAmount'].fillna((df['LoanAmount'].mean()), inplace=True)
df['Loan_Amount_Term'].fillna((df['Loan_Amount_Term'].mean()), inplace=True)
df['Credit_History'].fillna((df['Credit_History'].mean()), inplace=True)
print(df.isnull().sum())
'''
#Data Visualisation
from matplotlib import pyplot as plt
import seaborn as sns
sns.histplot(df['ApplicantIncome'])
plt.show()

sns.histplot(df['CoapplicantIncome'])
plt.show()

sns.histplot(df['LoanAmount'])
plt.show()

sns.countplot(x='Property_Area', data = df)
plt.show()

sns.countplot(x='Gender', data = df)
plt.show()

sns.countplot(x='Married', data = df)
plt.show()

sns.countplot(x='Dependents', data = df)
plt.show()

sns.countplot(x='Education', data = df)
plt.show()

sns.countplot(x='Self_Employed', data = df)
plt.show()
'''

#Dealing with outliers
import numpy as np
"""
df['ApplicantIncome'] = np.log1p(df['ApplicantIncome'])
df['CoapplicantIncome'] = np.log1p(df['CoapplicantIncome'])
df['LoanAmount'] = np.log1p(df['LoanAmount'])
"""
pd.set_option('display.max_columns', None)
print(df.head())

#Converting into categorical or numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Married'] = le.fit_transform(df['Married'])
df['Education'] = le.fit_transform(df['Education'])
df['Dependents'] = le.fit_transform(df['Dependents'])
df['Married'] = le.fit_transform(df['Married'])
df['Self_Employed'] = le.fit_transform(df['Self_Employed'])
df['Property_Area'] = le.fit_transform(df['Property_Area'])
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])


print(df.head())

X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
Y=df['Loan_Status']


#Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScores = pd.concat([dfcolumns, dfscores], axis=1)
featuresScores.columns = ['Specs', 'Score']

print(featuresScores)

X = df.drop(['Loan_Status', 'Loan_ID', 'Gender','Married','Dependents', 'Self_Employed', 'Property_Area','Education'], axis=1)

print(X)
print(Y)


#1. logistic Regression

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr=LogisticRegression()
pca=PCA(n_components=2)


pca.fit(X)
X=pca.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state=0,test_size=0.3)

logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
print('logistic Regression',accuracy_score(y_test,y_pred))

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










