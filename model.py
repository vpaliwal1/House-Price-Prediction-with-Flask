import pandas as pd
from sklearn.datasets import load_boston
boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
# print(boston.head())
# print(boston.isnull().sum())
# boston.drop(columns=[])
boston= boston.drop(columns=['CRIM', 'ZN', 'CHAS', 'NOX', 'AGE', 'DIS', 'RAD', 'B'])
boston["Price"]=boston_dataset.target
print(boston.columns)

X = boston.iloc[:, :5]
y = boston.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

import pickle
# Saving model to disk
pickle.dump(regressor, open('model_house.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model_house.pkl','rb'))
print(model.predict([[2.31, 6.421, 222.0,18.9,5.53]]))