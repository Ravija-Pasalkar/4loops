import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf

# Load dataset
df = pd.read_csv("datasets/classification_data.csv")

label_col = 'recommended_strategy'
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[label_col])

# Feature Engineering
df["income_to_networth_ratio"] = df["annual_income"] / (df["net_worth"] + 1e-6)
df["adjusted_debt_to_income"] = df["debt_to_income_ratio"] * df["annual_income"]
df["investment_savings_ratio"] = df["portfolio_value"] / (df["savings_rate"] + 1e-6)

df["annual_income"] = df["annual_income"].clip(lower=0)
df["net_worth"] = df["net_worth"].clip(lower=0)

df["age_group"] = pd.cut(df["age"], bins=[18, 35, 55, np.inf], labels=["Young", "Mid-age", "Senior"])
df["income_group"] = pd.cut(df["annual_income"], bins=[0, 50000, 150000, np.inf], labels=["Low", "Medium", "High"])
df["net_worth_level"] = pd.cut(df["net_worth"], bins=[0, 50000, 200000, np.inf], labels=["Poor", "Stable", "Wealthy"])

df["total_financial_score"] = df["financial_knowledge_score"] + df["macroeconomic_score"] + df["sentiment_index"]
df["total_allocation_pct"] = df["equity_allocation_pct"] + df["fixed_income_allocation_pct"]

X = df.drop(columns=['client_id', label_col])

# Date handling
if 'month' in X.columns:
    X['month'] = pd.to_datetime(X['month'], errors='coerce')
    X['month_num'] = X['month'].dt.month
    X['year'] = X['month'].dt.year
    X.drop(columns='month', inplace=True)

# Label encoding
label_encoders = {}
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Fill + scale
X = X.fillna(X.mean(numeric_only=True))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X_reshaped, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Transformer model
def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    pos_embed = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(tf.range(start=0, limit=input_shape[0]))
    x = inputs + pos_embed

    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.LayerNormalization()(x + attention)

    ffn = layers.Dense(64, activation="relu")(x)
    ffn = layers.Dense(input_shape[1])(ffn)
    x = layers.LayerNormalization()(x + ffn)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)

# Train
model = build_model(X_train.shape[1:], len(np.unique(y)))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32,
          callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)], verbose=1)

# Evaluate
_, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")