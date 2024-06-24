import pandas as pd

df=pd.read_csv("C:/Users/gauta/OneDrive/Documents/DS/DataSet/Dataset(Data Science Internship)-20240619T055654Z-001/Dataset(Data Science Internship)/housing.csv", delim_whitespace=True, header=None)
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = column_names
print(df.info())

#Dealing with Missing Values
print(df.isnull().sum())

X = df.drop(df.columns[13], axis=1)
Y = df[df.columns[13]]

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RM', y='MEDV', data=df)
plt.title('Rooms vs. Median Value of Homes')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Median Value of Homes (MEDV)')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='CRIM', data=df)
plt.title('Distribution of Crime Rate')
plt.xlabel('Crime Rate (CRIM)')
plt.show()

sns.pairplot(df[['RM', 'AGE', 'DIS', 'MEDV']])
plt.show()

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
model = XGBRegressor()
model.fit(X_train, Y_train)
pred = model.predict(X_train)
print(pred)

score_1 = metrics.r2_score(Y_train, pred)

score_2 = metrics.mean_absolute_error(Y_train, pred)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)


plt.scatter(Y_train, pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()
