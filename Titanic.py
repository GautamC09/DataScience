import pandas as pd

df=pd.read_csv("C:/Users/gauta/OneDrive/Desktop/SEM_5/DS_Internship/Titanic/tested.csv")
print(df.info())

#Dealing with Missing Values
print(df.isnull().sum())
df['Age'].fillna((df['Age'].mean()), inplace=True)
df['Fare'].fillna((df['Fare'].mean()), inplace=True)

#Converting into categorical or numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

#Preparing X and Y
X = df.drop(['PassengerId', 'Name', 'Cabin', 'Survived','Ticket','Age','Embarked','Pclass','Fare'], axis=1)
Y=df['Survived']


#Data Visualization

import matplotlib.pyplot as plt
ages = df['Age']
surv = df['Survived']

plt.hist(ages, bins=10, color="green", histtype='bar', edgecolor='black', alpha=0.5, weights=surv)
plt.xlabel("Ages")
plt.ylabel("Count of Survived")
plt.title("Age Distribution by Survival")
plt.show()

plt.hist(ages, bins=10, color="green", histtype='bar', edgecolor='black', alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.title("Number of People in Each Age Group")
plt.show()

plt.hist(df["Sex"])
plt.show()


#Dealing with Imbalance
a = (df['Survived'] == 1).sum()
print(a)
from collections import Counter
print(Counter(Y))
from imblearn.over_sampling import RandomOverSampler     #Random OverSampling
ros=RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X,Y)
print(Counter(Y))


#Feature selection 1
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScores = pd.concat([dfcolumns, dfscores], axis=1)
featuresScores.columns = ['Specs', 'Score']
print(featuresScores)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



'''
#Feature selection 2
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