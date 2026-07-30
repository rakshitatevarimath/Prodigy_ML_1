import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "data", "train.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# Create outputs folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("HOUSE PRICE PREDICTION PROJECT")
print("=" * 60)
print("Dataset Path :", DATA_PATH)
print("Output Folder:", OUTPUT_DIR)

# =====================================================
# LOAD DATASET
# =====================================================

try:
    df = pd.read_csv(DATA_PATH)
    print("\n✅ Dataset loaded successfully!")
except Exception as e:
    print("\n❌ Error loading dataset:")
    print(e)
    exit()

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = df["SalePrice"]

# =====================================================
# SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

model = LinearRegression()
model.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# EVALUATION
# =====================================================

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 30)
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"MSE      : {mse:.2f}")

# =====================================================
# SAVE RESULTS
# =====================================================

results_file = os.path.join(OUTPUT_DIR, "prediction_results.txt")

with open(results_file, "w", encoding="utf-8") as file:
    file.write("HOUSE PRICE PREDICTION RESULTS\n")
    file.write("=" * 40 + "\n\n")
    file.write(f"R² Score : {r2:.4f}\n")
    file.write(f"MAE      : {mae:.2f}\n")
    file.write(f"MSE      : {mse:.2f}\n")

print("\n✅ Results file created:")
print(results_file)

print("Exists:", os.path.exists(results_file))

# =====================================================
# SAVE GRAPH
# =====================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--"
)

plot_file = os.path.join(OUTPUT_DIR, "house_price_plot.png")

plt.savefig(plot_file, dpi=300)

plt.close()

print("\n✅ Plot file created:")
print(plot_file)

print("Exists:", os.path.exists(plot_file))

# =====================================================
# EXAMPLE PREDICTION
# =====================================================

sample_house = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3],
    "FullBath": [2]
})

prediction = model.predict(sample_house)

print("\nExample Prediction")
print("-" * 30)
print(f"Predicted House Price = ${prediction[0]:,.2f}")

print("\nProject completed successfully.")