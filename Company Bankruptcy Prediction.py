import pandas as pd

df=pd.read_csv("C:/Users/gauta/OneDrive/Documents/DS/DataSet/Dataset(Data Science Internship)-20240619T055654Z-001/Dataset(Data Science Internship)/Company Bankruptcy Prediction/data.csv")
pd.set_option('display.max_rows', None)
print(df.info())

#Dealing with Missing Values
print(df.isnull().sum())

#Preparing X and Y
X = df.drop(['Bankrupt?'], axis=1)
Y=df['Bankrupt?']

#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Bankrupt?', data=df)
plt.title('Class Distribution of Bankrupt and Non-Bankrupt Companies')
plt.xlabel('Bankrupt?')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#Dealing with Imbalance
a = (df['Bankrupt?'] == 1).sum()
print(a)
from collections import Counter
print(Counter(Y))
from imblearn.over_sampling import SMOTE                 #Synthetic Minority Oversampling (SMOTE) oversampling
sms=SMOTE(random_state=0)
X, Y=sms.fit_resample(X,Y)
print(Counter(Y))

# Feature Selection 1

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScores = pd.concat([dfcolumns, dfscores], axis=1)
featuresScores.columns = ['Specs', 'Score']
print(featuresScores)

#KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('KNN :',accuracy)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Decision Tree :',accuracy )
