
#Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Define Dataset
data = {
    'sqft': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'bedrooms': [3, 3, 3, 4, 2, 3, 4, 4, 3, 3],
    'bathrooms': [1, 2, 2, 2, 1, 2, 3, 3, 1, 2],
    'price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}
df = pd.DataFrame(data)

#  Step 3: Feature Selection
X = df[['sqft', 'bedrooms', 'bathrooms']]
y = df['price']

#  Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

#  Step 6: Prediction
y_pred = model.predict(X_test)

#  Step 7: Evaluation
print("✅ Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)  # Order corresponds to ['sqft', 'bedrooms', 'bathrooms']

#  Step 8: Visualization with regplot (for sqft only, 1D)
sns.regplot(x='sqft', y='price', data=df, line_kws={'color': 'red'})
plt.title("House Price vs Square Footage")
plt.xlabel("Square Footage")
plt.ylabel("House Price")
plt.grid(True)
plt.show()
