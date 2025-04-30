import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers

# 1. Load and clean data
data = pd.read_csv('data/supply_chain_data.csv')
data = data.dropna()

# 2. Date features
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Quarter'] = data['Date'].dt.quarter
    data = data.drop(columns=['Date'])

# 3. One-hot encode categorical features
categorical_columns = [
    'Product type', 'SKU', 'Customer demographics', 'Shipping carriers',
    'Supplier name', 'Location', 'Transportation modes', 'Routes',
    'Inspection results'
]
data = pd.get_dummies(data, columns=categorical_columns)

# 4. Split into features/target
target = 'Number of products sold'
X = data.drop(columns=[target])
y = data[target]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Scale only numeric columns
numeric_columns = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# 7. Build & compile the neural network
model = tf.keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 8. Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# 9. Plot training vs validation loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# 10. Evaluate on test set
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error on Test Set: {mse}')

# 11. Plot True vs Predicted Sales
plt.scatter(y_test, predictions)
plt.xlabel('True Sales')
plt.ylabel('Predicted Sales')
plt.title('True vs Predicted Sales')
plt.show()

# 12. Save the trained model
model.save('models/demand_forecasting_model.h5')
