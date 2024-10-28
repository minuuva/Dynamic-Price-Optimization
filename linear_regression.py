# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
jj = pd.read_csv('jj_translated.csv')

# Display the first few rows of the dataframe to understand its structure
print(jj.head())

# Define the relevant features and target columns
features = ['Date', 'Day of Week']  # Adjust based on your dataset

# %%
# Convert the 'Date' column to datetime format
jj['Date'] = pd.to_datetime(jj['Date'], errors='coerce')

# Drop rows where 'Date' could not be converted (if any)
jj = jj.dropna(subset=['Date'])

# Extract the day of the week from the 'Date' column if 'Day of Week' is not already present
if 'Day of Week' not in jj.columns:
    jj['Day of Week'] = jj['Date'].dt.day_name()

# Define the features as 'Date' and 'Day of Week'
features = ['Date']  # We'll add 'Day of Week' dummy variables later

# Convert 'Date' to numerical values (e.g., timestamp) and 'Day of Week' to dummy/indicator variables
jj['Date'] = jj['Date'].map(pd.Timestamp.toordinal)
#jj = pd.get_dummies(jj, columns=['Day of Week'], drop_first=True, errors='ignore')

# Identify unique menu items
menu_items = jj['Order'].unique()

# Dictionary to store models for each menu item
models = {}

# Loop through each menu item and train a model
for menu_item in menu_items:
    print(f'Training model for: {menu_item}')
    
    # Filter data for the specific menu item
    jj_item = jj[jj['Order'] == menu_item].copy()
    
    # Ensure there are sufficient data points
    if jj_item.shape[0] < 10:
        print(f'Not enough data to train model for {menu_item}. Skipping...')
        continue
    
    # Separate the features (X) and the target variable (y)
    X = jj_item[features + list(jj.columns[jj.columns.str.startswith('Day of Week_')])]
    y = jj_item['Price']  # The price for this specific menu item
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Print actual vs. predicted prices
    print(f"\nActual vs Predicted Prices for {menu_item}:")
    for actual, predicted in zip(y_test, y_pred):
        print(f"Actual: {actual}, Predicted: {predicted}")
    
    # Save the model in the dictionary
    models[menu_item] = model

# %%
menu_items = jj['Order'].unique()
print(menu_items)

# %%
for menu_item in menu_items:
    jj_item = jj[jj['Order'] == menu_item].copy()
    print(jj_item.shape[0])

# %%
# Dictionary to store models for each menu item
models = {}

# Loop through each menu item and train a model
for menu_item in menu_items:
    print(f'Training model for: {menu_item}')

    # Filter data for the specific menu item
    jj_item = jj[jj['Order'] == menu_item].copy()
    
    # Ensure there are sufficient data points
    if jj_item.shape[0] < 5:
        print(f'Not enough data to train model for {menu_item}. Skipping...')
        continue
    
    # Separate the features (X) and the target variable (y)
    X = jj_item[features]
    y = jj_item['Price']  # The price for this specific menu item
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error for {menu_item}: {mse}')
    print(f'R^2 Score for {menu_item}: {r2}')
    
    # Save the model in the dictionary
    models[menu_item] = model
    
    # Optionally, save each model to a file
    #model_path = f'/mnt/data/{menu_item.replace(" ", "_")}_price_model.pkl'
    #joblib.dump(model, model_path)
    #print(f'Trained model for {menu_item} saved to {model_path}')

# Example: Access the model for 'Strawberry Joyful' later
# strawberry_model = models.get('Strawberry Joyful')

# You now have models for each menu item saved and ready for use in your AI system.

# %%
for menu_item in menu_items:
    jj_item = jj[jj['Order'] == menu_item].copy()
    
    if jj_item.shape[0] < 10:
        print(f'Not enough data to train model for {menu_item}. Skipping...')
    else:
    # Separate the features (X) and the target variable (y)
        X = jj_item[features]
        y = jj_item['Price']  # The price for this specific menu item
    
    # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
    
    # Make predictions on the test set
        y_pred = model.predict(X_test)
    
    # Print actual vs. predicted prices
        print("Actual vs Predicted Prices for Each Menu:")
        for actual, predicted in zip(y_test, y_pred):
            print(f"Actual: {actual}, Predicted: {predicted}")

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load the dataset again
jj = pd.read_csv('jj_translated.csv')

# Select relevant features for the model
features = jj[['Day of Week', 'Payment Type', 'Price', 'Cost', 'Commision', 'Total VAT', 'Total Fee']]
target = jj['Revenue']

# Convert categorical variables to numerical format using one-hot encoding
categorical_features = features[['Day of Week', 'Payment Type']]
one_hot_encoder = OneHotEncoder(sparse=False, drop='first')  # Corrected argument 'sparse'
encoded_features = one_hot_encoder.fit_transform(categorical_features)

# Combine encoded features with the rest of the numerical features
numeric_features = features.drop(['Day of Week', 'Payment Type'], axis=1).values
X = np.hstack((numeric_features, encoded_features))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%

# Step 2: Initialize and train the Linear Regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

# Step 3: Predict revenue for different prices
# Generate a range of possible prices
price_range = np.arange(4000, 13000, 100)  # Prices from 4000 to 13000 in steps of 100

# Prepare the base input template: Keep other features at their average or typical values
average_cost = features['Cost'].mean()
average_commission = features['Commision'].mean()
average_vat = features['Total VAT'].mean()
average_fee = features['Total Fee'].mean()

# Encoding for categorical features (e.g., 'Monday' and 'BANK')
encoded_features = one_hot_encoder.transform([['Monday', 'BANK']])  # Example of fixed values

predicted_revenues = []
predicted_profits = []

for price in price_range:
    # Combine fixed average values with the varying price
    input_features = np.hstack(([[price, average_cost, average_commission, average_vat, average_fee]], encoded_features))
    
    # Predict revenue
    predicted_revenue = linear_regression_model.predict(input_features)
    
    # Calculate corresponding profit
    predicted_profit = predicted_revenue[0] - average_cost  # Simplified profit calculation
    
    predicted_revenues.append(predicted_revenue[0])
    predicted_profits.append(predicted_profit)

# Step 5: Find the optimal price
optimal_price_index = np.argmax(predicted_profits)  # Find index of maximum profit
optimal_price = price_range[optimal_price_index]

print(f"Optimal Price: {optimal_price}")


