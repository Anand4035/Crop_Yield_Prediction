import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import pickle

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("yield_df.csv")

# -----------------------------
# Basic Cleaning
# -----------------------------
df.drop('Unnamed: 0', axis=1, inplace=True)

print(df.head())

print(df.info())

print(df.isnull().sum())

print("Duplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Dataset Shape:", df.shape)

print(df.describe())

# -----------------------------
# Correlation (Numeric Columns Only)
# -----------------------------
numeric_df = df.select_dtypes(include=['number'])

print(numeric_df.corr())

# -----------------------------
# Data Visualization
# -----------------------------
plt.figure(figsize=(15,20))
sns.countplot(y=df['Area'])
plt.title("Area Count")
plt.show()

plt.figure(figsize=(15,20))
sns.countplot(y=df['Item'])
plt.title("Crop Count")
plt.show()

# -----------------------------
# Yield Per Country
# -----------------------------
country = df['Area'].unique()

yield_per_country = []

for state in country:
    total = df[df['Area'] == state]['hg/ha_yield'].sum()
    yield_per_country.append(total)

plt.figure(figsize=(15,20))
sns.barplot(y=country, x=yield_per_country)
plt.title("Yield Per Country")
plt.show()

# -----------------------------
# Yield Per Crop
# -----------------------------
crops = df['Item'].unique()

yield_per_crop = []

for crop in crops:
    total = df[df['Item'] == crop]['hg/ha_yield'].sum()
    yield_per_crop.append(total)

plt.figure(figsize=(15,20))
sns.barplot(y=crops, x=yield_per_crop)
plt.title("Yield Per Crop")
plt.show()

# -----------------------------
# Select Required Columns
# -----------------------------
col = [
    'Year',
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp',
    'Area',
    'Item',
    'hg/ha_yield'
]

df = df[col]

print(df.head())

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop('hg/ha_yield', axis=1)

y = df['hg/ha_yield']

print("X Shape:", X.shape)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# -----------------------------
# Preprocessing
# -----------------------------
ohe = OneHotEncoder(drop='first')

scale = StandardScaler()

preprocesser = ColumnTransformer(
    transformers=[
        ('StandardScale', scale, [0,1,2,3]),
        ('OneHotEncode', ohe, [4,5])
    ],
    remainder='passthrough'
)

# IMPORTANT FIX
X_train_dummy = preprocesser.fit_transform(X_train)

X_test_dummy = preprocesser.transform(X_test)

# -----------------------------
# Models
# -----------------------------
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    'KNN': KNeighborsRegressor()
}

print("\nModel Performance:\n")

for name, model in models.items():

    model.fit(X_train_dummy, y_train)

    y_pred = model.predict(X_test_dummy)

    mae = mean_absolute_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    print(f"{name}")
    print(f"MAE : {mae}")
    print(f"R2 Score : {r2}")
    print("-" * 40)

# -----------------------------
# Final Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_dummy, y_train)

# -----------------------------
# Prediction System
# -----------------------------
def prediction(
    Year,
    average_rain_fall_mm_per_year,
    pesticides_tonnes,
    avg_temp,
    Area,
    Item
):

    features = np.array([[
        Year,
        average_rain_fall_mm_per_year,
        pesticides_tonnes,
        avg_temp,
        Area,
        Item
    ]], dtype=object)

    transformed_features = preprocesser.transform(features)

    predicted_yield = model.predict(transformed_features)

    return predicted_yield[0]

# -----------------------------
# Sample Prediction
# -----------------------------
result = prediction(
    1990,
    1485.0,
    121.0,
    16.37,
    'Albania',
    'Maize'
)

print("\nPredicted Yield:")
print(result)

# -----------------------------
# Save Model
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))

pickle.dump(preprocesser, open("preprocesser.pkl", "wb"))

print("\n✅ Model and Preprocessor Saved Successfully")