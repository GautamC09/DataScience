import pandas as pd

df=pd.read_csv("C:/Users/gauta/OneDrive/Documents/DS/DataSet/Dataset(Data Science Internship)-20240619T055654Z-001/Dataset(Data Science Internship)/50_Startups.csv")
print(df.info())

#Dealing with Missing Values
print(df.isnull().sum())

#Converting into categorical or numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['State'] = le.fit_transform(df['State'])

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df)
plt.show()

X = df.drop('Profit', axis=1)
Y = df['Profit']

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x= ss.fit_transform(X)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(lr.score(x_test,y_test))


