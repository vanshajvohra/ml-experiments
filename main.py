from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

predicted_prices = melb_model.predict(X)
# print("Initial Predictions")
# print(predicted_prices)

mae = mean_absolute_error(y, predicted_prices) # Measure of model quality
# print(mae)

train_x, val_x, train_y, val_y = train_test_split(X, y, random_state=0) # splitting data into training and validation sets
new_model = DecisionTreeRegressor(random_state=0)
new_model.fit(train_x, train_y) # Fitting or training is always done on training data
new_predictions = new_model.predict(val_x) # Predictions are done on validation data
# print(mean_absolute_error(val_y, new_predictions)) # predictions are compared with unseen data

candidates_max_leaf_nodes = [5, 10, 25, 100, 500, 1000]
for mlf in candidates_max_leaf_nodes:
    newest_model = DecisionTreeRegressor(max_leaf_nodes=mlf, random_state=0)
    newest_model.fit(train_x, train_y)
    newest_predictions = newest_model.predict(val_x)
    print(mean_absolute_error(val_y, newest_predictions))
optimal_leaves = 500

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_x, train_y)
forest_model_predictions = forest_model.predict(val_x)
forest_model_mae = mean_absolute_error(val_y, forest_model_predictions)
print(forest_model_mae) # significant lower mae even after optimizing max leaf nodes