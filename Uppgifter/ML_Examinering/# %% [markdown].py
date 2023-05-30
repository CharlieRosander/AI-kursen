import nbformat as nbf
ntbk = nbf.read("Markdown.ipynb", nbf.NO_CONVERT)
new_ntbk = ntbk
new_ntbk.cells = [cell for cell in ntbk.cells if cell.cell_type != "markdown"]
nbf.write(new_ntbk, "MarkdownDeleted.ipynb", version=nbf.NO_CONVERT)

# %% [markdown]
# # Machine Learning - Examinering
# #### Charlie Rosander
# 
# 
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score

# %% [markdown]
#  ### In this notebook we will do some EDA/ETL, preprocessing of the data, and then train a model and see what we can do with it.
#  We will be using the Happiness dataset from kaggle, and try to predict the happiness score of a country based on the other features in the dataset.
#  (https://www.kaggle.com/datasets/sougatapramanick/happiness-index-2018-2019)
# 
# Some of the steps we might take are:
# 
# - EDA
#     - Visualizations
#     - Correlations
# <br>
# <br>
# - Preprocessing
#     - Missing values
#     - Outliers
#     - Scaling
#     - Encoding
#     - Train test split
# <br>
# <br>
# - Model
#     - Train
#     - Evaluate
#     - Predict
# <br>
# <br>
# - Evaluation
#     - Metrics
#     - Visualizations
# <br>
# <br>
# - Conclusion
#     - What did we learn?
#     - What could we have done better?
#     - What could we do next?
# 
#     

# %%
# We start by loading the data from the csv file
data_2018 = pd.read_csv('./data/happiness/2018.csv')
data_2019 = pd.read_csv('./data/happiness/2019.csv')


# %%
# Inspect the data.
# We see that we have a column containing the country name or region, and as they are letters we might need to encode them later on.
# We also see that the dataset doesn't really need any comprehensive cleaning aside from.
data_2018

# We start by creating a copy of the data, so we can modify it without losing the original data.
data_2018_copy = data_2018.copy()
data_2019_copy = data_2019.copy()

# %% [markdown]
# ### To more easily understand the dataset and its features I will put the description of the data here:
# 
# ### Overview of the Data (2018 & 2019)
# 
# - Overall rank: List of ranks of different countries from 1 to 156
# - Country or region: List of the names of different countries.
# - Score: List of happiness scores of different countries.
# - GDP per capita: The GDP per capita score of different countries.
# - Social support: The social support of different countries.
# - Healthy life expectancy: The healthy life expectancy of different countries.
# - Freedom to make life choices: The score of perception of freedom of different countries.
# - Generosity: Generosity (the quality of being kind and generous) score of different countries.
# - Perceptions of corruption: The score of the perception of corruption in different countries.

# %%
# We use the describe function to get a quick overview of the data.
data_2018.describe()

# %%
# Quick check for nan values, there are none.
data_2018.isnull().sum()

# %%
# Here we check the correlation between the different columns, but for this I will drop the country name column.
# We will be using a heatmap to visualize the correlation.

# We drop the country name column.
data_2018_copy.drop(['Country or region'], axis=1, inplace=True)

# As well as the Overall rank column, as it doesn't really tell us anything here.
data_2018_copy.drop(['Overall rank'], axis=1, inplace=True)

# We create a correlation matrix.
corr_2018 = data_2018_copy.corr(method="pearson")

# We create a heatmap to visualize the correlation.
sns.heatmap(corr_2018, annot=True)
plt.title('Correlation Matrix - Dataset 2018')
plt.show()

# We will also quickly examine the dataset from 2019, to see if there are any major differences.
data_2019 = pd.read_csv('./data/happiness/2019.csv')

data_2019_copy.drop(['Country or region'], axis=1)

data_2019_copy.drop(['Overall rank'], axis=1)

corr_2019 = data_2019_copy.corr(method="pearson")

sns.heatmap(corr_2019, annot=True)
plt.title('Correlation Matrix - Dataset 2019')
plt.show()


# %% [markdown]
# From this we can see that there is a strong correlation between:
# - GDP per capita and score
# - Social support and score
# - Healthy life expectancy and score
# 
# As well as:
# - GDP per capita and social support
# - GDP per capita and healthy life expectancy
# - Social support and healthy life expectancy
# - Social support and healy life expectancy
# 
# Which is both interesting and expected as people who live in countries with a high GDP per capita, social support and healthy life expectancy are more likely to be happy, and that a higher social support leads to a higher healthy life expectancy. 
# 
# 
# We can also see that there is a weak correlation between:
# - Freedom to make life choices and score
# - Generosity and score
# - Perceptions of corruption and score
# 
# This is also quite interesting as this tells me that for example the freedom to make life choices is not as important as the other features when it comes to happiness, the correlation is quite low.
# 
# Generosity is even more interesting as the correlation there is almost non-existant, which at first thought would seem strange as you would think that countries who are more generous would have a happier population, but judging from this dataset that is not necessarily the case, which is something to think about. 

# %% [markdown]
# Let's start testing  different models by trying to predict the happiness score of a country based on some of the more interesting features in the dataset.
# 
# We have chosen to use the following features based on the correlations we found above:
# - GDP per capita
# - Social support
# - Healthy life expectancy
# - Freedom to make life choices
# - Generosity
# - Perceptions of corruption

# %% [markdown]
# ### Creating, training and testing the models
# 
# We will create and train the models in order, using GridSearchCV where possible to find the best parameters for each model.
# 
# We will then test the models and see how they perform, using different scoring methods, as well as plotting the results so we can see how they compare.

# %%
# Here we create a simple list to store the different results from the models to easily compare them later on.
model_results = []

# %% [markdown]
# **Linear Regression**

# %%
# Select the features and target variable
features = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices',
            'Generosity', 'Perceptions of corruption']
target = 'Score'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_2018[features], data_2018[target], test_size=0.2, random_state=42)

# We will now scale the data.
# We will scale the data using the MinMaxScaler, as we want to keep the data in the range of 0-1.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg_model.predict(X_test)



# %% [markdown]
# **Testing of the LinearRegression model**
# 
# - MSE - Mean squared error, the lower the better
# - R2 - R squared, the higher the better
# - CV - Cross validation (K-fold), the higher the better
# - Plotting - Plotting the predicted values against the actual values, the closer to a straight line the better

# %%
# Test the models accuracy with different methods.
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

cv_scores = cross_val_score(reg_model, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Average cross-validation score: {:.2f}".format(cv_scores.mean()))

# We plot the actual vs predicted happiness scores to visualize the results.
plt.scatter(y_test, y_pred, color='blue', label='Predicted Scores')
plt.scatter(y_test, y_test, color='red', label='Actual Scores')
plt.xlabel('Actual Happiness Score')
plt.ylabel('Predicted Happiness Score')
plt.title('Actual vs Predicted Happiness Scores, Linear Regression')
plt.legend()
plt.show()

# We add the results to the list.
model_results.append(['Linear Regression', mse, r2, cv_scores.mean()])

# %% [markdown]
# We can see the scores of the regression model is okay, but we will see if we can get better scores with other models.
# 
# ### These are the models we will be testing next:
# - Random Forest
# - Decision Tree
# - KNN
# - Stacking

# %% [markdown]
# **Random Forest**

# %% [markdown]
# GridSearchCV

# %%
# We will use GridSearchCV to find the best parameters for the model.

# Define the parameter grid
params_rf = {'n_estimators': [int(x) for x in np.linspace(start=100, stop=3000, num=20)]}

# Create a base model
rf = RandomForestRegressor(random_state=42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=params_rf, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters:", grid_search.best_params_)

# %% [markdown]
# Training of the Random Forest model

# %%
# Extract the best parameters
best_params_rf = grid_search.best_params_

# We will now use the best parameters to create a new model.
rf_model = RandomForestRegressor(n_estimators=best_params_rf["n_estimators"], random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# %% [markdown]
# **Testing of the RandomForest model**
# - MSE - Mean squared error, the lower the better
# - R2 - R squared, the higher the better
# - CV - Cross validation (K-fold), the higher the better
# - Plotting - Plotting the predicted values against the actual values, the closer to a straight line the better
# - Feature importance - Plotting the feature importance of the model

# %%
# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate the R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Perform cross-validation and calculate the average score
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score: {:.2f}".format(cv_scores.mean()))

# We plot the actual vs predicted happiness scores to visualize the results.
plt.scatter(y_test, y_pred, color='blue', label='Predicted Scores')
plt.scatter(y_test, y_test, color='red', label='Actual Scores')
plt.xlabel('Actual Happiness Score')
plt.ylabel('Predicted Happiness Score')
plt.title('Actual vs Predicted Happiness Scores, Random Forest Regression')
plt.legend()
plt.show()

# Get the feature importances
importances = rf_model.feature_importances_

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.barh(range(X_train.shape[1]), importances, align='center')
plt.yticks(range(X_train.shape[1]), features)
plt.show()

# We add the results to the list.
model_results.append(['Random Forest Regression', mse, r2, cv_scores.mean()])

# %% [markdown]
# From the testing of the RandomForest model we can see that it performs better than the LinearRegression model.
# It's also interesting to see the feature importance of the model, as we can see and compare with what we saw in the correlation matrix, that GDP per capita, social support and healthy life expectancy are the most important features when it comes to predicting the happiness score of a country. 
# 
# However, from this models testing we see that social support is the most important feature, which is interesting as we saw in the correlation matrix that GDP per capita was the most important feature for 2018.

# %% [markdown]
# **Decision Tree**

# %% [markdown]
# GridSearchCV

# %%
# We will now use a Decision Tree Regressor model.
# We will once again use GridSearch to find the best parameters for the model.

# Define the parameters to tune
params_dt = {
    'max_depth': range(1, 10),
    'min_samples_split': range(2, 10),
    'min_samples_leaf': range(1, 10)
}

# Initialize the Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=42)

# Initialize the GridSearchCV
grid_search = GridSearchCV(tree_model, params_dt, cv=5, scoring='neg_mean_squared_error')

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
print('Best parameters:', grid_search.best_params_)


# %% [markdown]
# Training of the DecisionTree model

# %%
# Extract the best parameters
best_params_dt = grid_search.best_params_

# We will now use the best parameters to create a new model.
tree_model = DecisionTreeRegressor(max_depth=best_params_dt['max_depth'], 
                                   min_samples_leaf=best_params_dt['min_samples_leaf'], 
                                   min_samples_split=best_params_dt['min_samples_split'], 
                                   random_state=42)

# Fit the model to the training data
tree_model.fit(X_train, y_train)

# %% [markdown]
# **Testing of the DecisionTree model**
# - MSE - Mean squared error, the lower the better
# - R2 - R squared, the higher the better
# - CV - Cross validation (K-fold), the higher the better
# - Plotting - Plotting the predicted values against the actual values, the closer to a straight line the better

# %%
# Make predictions
y_pred = tree_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate the R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Perform cross-validation and calculate the average score
cv_scores = cross_val_score(tree_model, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Average cross-validation score: {:.2f}".format(cv_scores.mean()))

plt.scatter(y_test, y_pred, color='blue', label='Predicted Scores')
plt.scatter(y_test, y_test, color='red', label='Actual Scores')
plt.xlabel('Actual Happiness Score')
plt.ylabel('Predicted Happiness Score')
plt.title('Actual vs Predicted Happiness Scores, Decision Tree Regression')
plt.legend()
plt.show()

# Get the feature importances
importances = tree_model.feature_importances_

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.barh(range(X_train.shape[1]), importances, align='center') 
plt.yticks(range(X_train.shape[1]), features) 
plt.show()

# We add the results to the list.
model_results.append(['Decision Tree Regression', mse, r2, cv_scores.mean()])

# %% [markdown]
# We can see that the DecisionTree model performs worse than both of the other models, but also how the feature-importance differs from the RandomForest model, as we can see that the most important feature is GDP per capita by a big margin, and that Freedom to make life choices makes a comeback as the third most important feature.

# %% [markdown]
# **KNN**

# %% [markdown]
# GridSearchCV

# %%
# We will now use KNN regression.
# We will once again use GridSearch to find the best parameters for the model.

# Define the parameters
params_knn = {
    'n_neighbors': range(1, 21), # Number of neighbors to use
    'weights': ['uniform', 'distance'], # Uniform weights or distance weights
    'p': [1, 2] # 1: Manhattan distance, 2: Euclidean distance
}

# Initialize the KNN Regressor
knn_model = KNeighborsRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(knn_model, params_knn, cv=5, scoring='neg_mean_squared_error')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
print('Best parameters:', grid_search.best_params_)


# %% [markdown]
# Training of the KNN model

# %%
# We will now use the best parameters to create a new model.
# Extract best parameters
best_params_knn = grid_search.best_params_

# Create a new KNN Regressor with best parameters
knn_model_best = KNeighborsRegressor(n_neighbors=best_params_knn["n_neighbors"], 
                                     weights=best_params_knn["weights"], 
                                     p=best_params_knn["p"])

# Fit the model
knn_model_best.fit(X_train, y_train)

# %% [markdown]
# **Testing of the KNN model**

# %%
# Predict
y_pred = knn_model_best.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate the R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Perform cross-validation and calculate the average score
cv_scores = cross_val_score(knn_model_best, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Average cross-validation score: {:.2f}".format(cv_scores.mean()))

# Plot the predicted vs actual happiness scores
plt.scatter(y_test, y_pred, color='blue', label='Predicted Scores')
plt.scatter(y_test, y_test, color='red', label='Actual Scores')
plt.xlabel('Actual Happiness Score')
plt.ylabel('Predicted Happiness Score')
plt.title('Actual vs Predicted Happiness Scores, KNN Regression')
plt.legend()
plt.show()

# We add the results to the list.
model_results.append(['KNN Regression', mse, r2, cv_scores.mean()])

# %% [markdown]
# We see here that KNN performed pretty well, last but not least we will try to stack the models and see if we can get even better results.

# %% [markdown]
# **Stacking**

# %%
# We will now use stacking to combine the models and see if we can improve the results.
# We will use the best parameters for each model.

# Define the base models; Random Forest, Decision Tree and KNN
base_models = [
    ('random_forest', RandomForestRegressor(n_estimators=best_params_rf["n_estimators"], 
                                            max_depth=best_params_rf["max_depth"], 
                                            random_state=42)),
    ('decision_tree', DecisionTreeRegressor(max_depth=best_params_dt["max_depth"], 
                                            min_samples_split=best_params_dt["min_samples_split"], 
                                            min_samples_leaf=best_params_dt["min_samples_leaf"], 
                                            random_state=42)),
    ('knn', KNeighborsRegressor(n_neighbors=best_params_knn["n_neighbors"], 
                                     weights=best_params_knn["weights"], 
                                     p=best_params_knn["p"]))
]

# Initialize the stacking model
stacking_model = StackingRegressor(estimators=base_models, 
                                   final_estimator=LinearRegression(), 
                                   cv=5)

# Fit the model
stacking_model.fit(X_train, y_train)

# Predict
y_pred = stacking_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate the R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Perform cross-validation and calculate the average score
cv_scores = cross_val_score(stacking_model, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Average cross-validation score: {:.2f}".format(cv_scores.mean()))

# Plot the predicted vs actual happiness scores
plt.scatter(y_test, y_pred, color='blue', label='Predicted Scores')
plt.scatter(y_test, y_test, color='red', label='Actual Scores')
plt.xlabel('Actual Happiness Score')
plt.ylabel('Predicted Happiness Score')
plt.title('Actual vs Predicted Happiness Scores, Stacking Regressor')
plt.legend()
plt.show()


# %%



