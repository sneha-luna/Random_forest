import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Step 1: Create the dataset
data = {
    'color_score': [1, 2, 3, 7, 8, 9],
    'texture_score': [1, 1, 2, 8, 9, 10],
    'label': [0, 0, 0, 1, 1, 1]  # 0 = Healthy, 1 = Diseased
}

df = pd.DataFrame(data)

# Step 2: Features and label separation
X = df[['color_score', 'texture_score']]
y = df['label']

# Step 3: Train the Random Forest model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Step 4: Predict on a new sample using a DataFrame (to avoid warning)
new_leaf = pd.DataFrame([[5, 5]], columns=['color_score', 'texture_score'])
prediction = model.predict(new_leaf)

# Step 5: Output result
print("Prediction (0=Healthy, 1=Diseased):", prediction[0])
