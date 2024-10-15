import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
dataset = pd.read_csv('diabetes_prediction_dataset.csv')

# Encode categorical variables
le = LabelEncoder()
columns = ['gender', 'smoking_history']
for col in columns:
    dataset[col] = le.fit_transform(dataset[col])

# Separate features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Balance the dataset
sm = SMOTE(random_state=2)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train_sm, y_train_sm)

# Save the model and scaler
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Model and scaler saved successfully.")
