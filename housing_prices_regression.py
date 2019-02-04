import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#Load Data
data = pd.read_csv('housing_prices.csv')
data = data.dropna()
#data.info()

#One Hot encoding of categorical variable
data['ocean_proximity'] = data['ocean_proximity'].astype("category")
proximity_dummies=pd.get_dummies(data['ocean_proximity'])

#Selection of columns
target_name = data.columns[8]
training_names = data.columns 
training_names = training_names.drop(["ocean_proximity", target_name])


#Split into input and output matrices
y = data[target_name]
X = data.loc[:, training_names]
X = X.join(proximity_dummies)



#Split into test and train sets
X_test = X[0:5000]
y_test = y[0:5000]
X_train = X[5000:20433]
y_train = y[5000:20433]


#Fit the Model
regr = RandomForestRegressor(n_estimators=100)
regr.fit(X_train, y_train)


#Test the Model
print(regr.score(X_test, y_test))