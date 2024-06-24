import pandas as pd
import numpy as np

df=pd.read_csv("C:/Users/gauta/OneDrive/Documents/DS/DataSet/Dataset(Data Science Internship)-20240619T055654Z-001/Dataset(Data Science Internship)/Black Friday Sales/train.csv")
print(df.info())
print(df['Purchase'].describe())

#Dealing with Missing Values
print(df.isnull().sum())
df['Product_Category_2'].fillna((df['Product_Category_2'].mean()), inplace=True)
df['Product_Category_3'].fillna((df['Product_Category_3'].mean()), inplace=True)
print(df.isnull().sum())

#Converting into categorical or numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Product_ID'] = le.fit_transform(df['Product_ID'])
df['Gender'] = le.fit_transform(df['Gender'])
df['City_Category'] = le.fit_transform(df['City_Category'])
df['Stay_In_Current_City_Years'] = le.fit_transform(df['Stay_In_Current_City_Years'])
df['Age'] = le.fit_transform(df['Age'])

'''
bins = [0, 5823, 8047, 12054, float('inf')]
labels = [0, 1, 2, 3]
df['Purchase'] = pd.cut(df['Purchase'], bins=bins, labels=labels, right=False)
'''

#Preparing X and Y
X = df.drop(['Purchase','Stay_In_Current_City_Years','Marital_Status'], axis=1)
Y=df['Purchase']
'''
#Data Visualization
import  seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['Purchase'], bins=25)
plt.show()

sns.violinplot(x='City_Category',y='Purchase',hue='Marital_Status',data=df)
plt.show()

age_plot = df.pivot_table(index='Age', values='Purchase', aggfunc=np.mean)
age_plot.plot(kind='bar', figsize=(13, 7))
plt.title("Age and Purchase Analysis")
plt.xticks(rotation=0)
plt.show()
'''

#Dealing with Imbalance
a = (df['Purchase'] == 1).sum()
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



from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
model = LinearRegression()


x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.25)
model.fit(x_train, y_train)

# predict the results
pred = model.predict(x_test)


cv_score = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=5)
cv_score = np.abs(np.mean(cv_score))

print("Results")
print("MSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("CV Score:", np.sqrt(cv_score))


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.25)
model.fit(x_train, y_train)

# predict the results
pred = model.predict(x_test)

cv_score = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=5)
cv_score = np.abs(np.mean(cv_score))

print("Results")
print("MSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("CV Score:", np.sqrt(cv_score))



from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.25)
model.fit(x_train, y_train)

# predict the results
pred = model.predict(x_test)

cv_score = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=5)
cv_score = np.abs(np.mean(cv_score))

print("Results")
print("MSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("CV Score:", np.sqrt(cv_score))