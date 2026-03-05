import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# STEP 1: AUTOMATED DATA GENERATION
# ==========================================
print("Step 1: Generating 1,000 synthetic FEM simulations...")
np.random.seed(42)
num_samples = 1000

# Randomize parameters (Length, Moment of Inertia, Force)
lengths = np.random.uniform(2.0, 10.0, num_samples)
inertias = np.random.uniform(5e-6, 20e-6, num_samples)
forces = np.random.uniform(-10000, -1000, num_samples)
E = 200e9  # Young's Modulus for steel/carbon fiber approximation

# Simulate the FEM calculation for max deflection
# (Using the structural formula for rapid dataset generation)
deflections = (7 * forces * (lengths**3)) / (768 * E * inertias)

# Save to a structured CSV dataset as specified in your report
df = pd.DataFrame({
    'Length_m': lengths,
    'Inertia_m4': inertias,
    'Force_N': forces,
    'Max_Deflection_m': deflections
})
df.to_csv('synthetic_beam_data.csv', index=False)
print("Dataset successfully saved to 'synthetic_beam_data.csv'\n")

# ==========================================
# STEP 2: TRAIN THE PREDICTIVE AI MODEL
# ==========================================
print("Step 2: Training the Machine Learning Model...")
X = df[['Length_m', 'Inertia_m4', 'Force_N']]
y = df['Max_Deflection_m']

# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the AI Model
ai_model = RandomForestRegressor(n_estimators=100, random_state=42)
ai_model.fit(X_train, y_train)

# Evaluate the model's accuracy
accuracy = ai_model.score(X_test, y_test)
print(f"Model Training Complete! AI Accuracy Score (R-squared): {accuracy * 100:.2f}%\n")

# ==========================================
# STEP 3: PREDICT NEW DESIGN INSTANTLY
# ==========================================
print("Step 3: Predicting a new structure without running FEM matrices...")
# Let's test it on a 4-meter beam with the same 5000N force we used in our JSON file
test_design = pd.DataFrame({'Length_m': [4.0], 'Inertia_m4': [8.33e-6], 'Force_N': [-5000]})
predicted_deflection = ai_model.predict(test_design)

print(f"Test Parameters -> Length: 4.0m | Inertia: 8.33e-6 | Force: -5000 N")
print(f"AI Predicted Max Deflection: {predicted_deflection[0] * 1000:.4f} mm")