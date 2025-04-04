import joblib
import numpy as np
import os
from tensorflow.keras.models import save_model

# Import these from train_model.py
from train_model import model, X, target_encoder, label_encoders, scaler

# Create save dir
os.makedirs("saved_pipeline", exist_ok=True)

# Save model
save_model(model, "saved_pipeline/transformer_model.keras")

# Save label encoder (for target)
np.save("saved_pipeline/label_classes.npy", target_encoder.classes_, allow_pickle=True)
joblib.dump(target_encoder, "saved_pipeline/target_encoder.pkl")

# Save feature columns
joblib.dump(X.columns.tolist(), "saved_pipeline/feature_columns.pkl")

# Save all feature encoders
joblib.dump(label_encoders, "saved_pipeline/label_encoders.pkl")

# Save scaler
joblib.dump(scaler, "saved_pipeline/scaler.pkl")

print("All pipeline components saved!")