from sklearn.tree import DecisionTreeRegressor
import pandas as pd

melb_data_path = 'melb_data.csv'
melb_data = pd.read_csv(melb_data_path) # reading CSV data

# print(melb_data.describe())
# print(melb_data.columns)

# print(melb_data)
melb_data = melb_data.dropna(axis = 0) # removes all rows with missing values
# print(melb_data)

y = melb_data.Price # dot notation to select the column we want to predict
# print(y)

melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'] # columns we are going to use to predict our target Price
X = melb_data[melb_features]
# print(X.describe())
# print(X.head()) # inspecting data is an important part

melb_model = DecisionTreeRegressor(random_state=1) # choosing a number means we get the same results in every run
melb_model.fit(X, y) # Fitting features and target

print("Initial Predictions")
print(melb_model.predict(X.head()))