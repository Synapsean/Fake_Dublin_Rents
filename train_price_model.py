import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib  # <--- NEW: For saving the scalers
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

print("Fetching data...")
response = supabase.table("listings").select("price, beds, baths, property_type").execute()
df = pd.DataFrame(response.data)
df = df.dropna()

# 2. PREPROCESSING
y = df['price'].values
X = df.drop('price', axis=1)

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True, dtype=float)

# *** CRITICAL: Save the column names ***
# If our training data has "House" but our test data only has "Apartment",
# the shapes will break. We save this list to enforce consistency later.
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Saved model_columns.pkl")

# Scaling
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X.values)
joblib.dump(scaler_x, 'scaler_x.pkl') # <--- Save Input Scaler
print("Saved scaler_x.pkl")

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
joblib.dump(scaler_y, 'scaler_y.pkl') # <--- Save Output Scaler
print("Saved scaler_y.pkl")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)

# 3. DEFINE MODEL
class PriceNet(nn.Module):
    def __init__(self, input_features):
        super(PriceNet, self).__init__()
        self.layer1 = nn.Linear(input_features, 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x

model = PriceNet(X_train.shape[1])

# 4. TRAIN
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\nTraining Model...")
epochs = 500
for epoch in range(epochs):
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1} | Loss: {loss.item():.4f}')

# 5. SAVE MODEL
torch.save(model.state_dict(), 'rental_price_model.pth')
print("Saved rental_price_model.pth")
print("\n--- TRAINING COMPLETE ---")