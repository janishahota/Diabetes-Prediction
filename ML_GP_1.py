import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

# Reading the dataset using the pandas library
dataset = pd.read_csv('diabetes_prediction_dataset.csv')

# Encoding the gender and smoking history columns into categorical variables 
# for the model to understand the string values
le = LabelEncoder()
columns = ['gender', 'smoking_history']
for i in range(len(columns)):
    dataset[columns[i]] = le.fit_transform(dataset[columns[i]])

# Showing the first five entries in the dataset to understand its content
print(f'First five entries in the dataset: - \n{dataset.head()}')
print('\n')
# Checking the datatype of the columns and if there are any null entries present
print('Information about the dataset: -')
print(dataset.info())
print('\n')

# Showing the statistical description of the dataset
print(f'Statistical Description of the dataset: - \n{dataset.describe()}')
print('\n')
# Showing the number of entries with no diabetes and diabetes seperately, 
# helping us understand the distribution of the target variable throughout the dataset
outcome = dataset['diabetes'].value_counts()
print(f'Number of non diabetic and diabetic patients in the dataset: - \n{outcome}')
print('\n')
labels = {0: 'Not Diabetes', 1: 'Diabetes'}
outcome.index = outcome.index.map(labels)
plt.pie(outcome, autopct="%1.1f%%")
plt.title("Diabetes Prediction")
plt.legend(title="Diabetes Prediction", labels=outcome.index)
plt.show()

# Plotting histograms for the selected features in the dataset
db = ["bmi", "HbA1c_level", "blood_glucose_level", "age"]

for i in range(len(db)):
    plt.hist(dataset[db[i]])
    plt.title('Histogram Plot')
    plt.xlabel(db[i])
    plt.ylabel("values")
    plt.show()

# Plotting the heatmap to find correlations in the dataset
plt.figure(figsize=(10, 8))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='magma')
plt.title('Correlation Heatmap')
plt.show()

# Plotting boxplots for the selected features in the dataset
features = ["age", 'bmi', 'HbA1c_level', 'blood_glucose_level']

for feature in features:
    sns.boxplot(x='diabetes', y=feature, data=dataset, palette='coolwarm')
    plt.title(f'Box Plot of {feature} by Outcome')
    plt.show()

# Plotting the age vs blood glucose level line plot
sns.lineplot(x="age", y="blood_glucose_level", hue='diabetes', data=dataset, palette='coolwarm')
plt.title(f'Line Graph of Age by Outcome')
plt.xlabel("Age")
plt.ylabel('Glucose')
plt.show()

# Plotting the outcome vs HbA1c level violin plot
sns.violinplot(x='diabetes', y='HbA1c_level', data=dataset, palette='coolwarm')
plt.title('Violin Plot of Blood Pressure by Outcome')
plt.show()

# Plotting the Kernel Density Estimate plot for selected features in features list by outcome
for feature in features:
    sns.kdeplot(data=dataset, x=feature, hue='diabetes', fill=True, palette='coolwarm')
    plt.title(f'KDE Plot of {feature} by Outcome')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()

# Seperating the features and the target variables from the dataset into two different arrays
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Using standard scale to scale the dataset and equalise the values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Balancing the distribution of the outcomes throughout the data for unbiased results
sm = SMOTE(random_state=2)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Plotting the line chart for outcomes to show the balance in the values after using SMOTE
n = pd.DataFrame(y_train_sm).value_counts()
n.plot(kind='bar')
plt.show()

# Creating a list of all the classification models used in the project
models = {
    # Giving 1000 as max iterations in Logistic Regression to train on the dataset accurately
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Creating a for loop to see the results of the model after training it on the training set
# and evaluating it by calculating accuracy score, confusion matrix, and plotting the probability
# distribution cureve
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy}')
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'Confusion matrix for the predictions of {name}: - \n{conf_matrix}')

    # Code for plotting confusion matrix
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted') 
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Code for plotting probability distribution curve
    plt.figure(figsize=(7.5, 4.5))
    sns.kdeplot(y_test, label='True')
    sns.kdeplot(y_pred, label='Predicted')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution of Original vs Predicted Classifications')
    plt.legend()
    plt.show()

# After checking accuracies of each model, we consider Decision Tree Classifier as the most accurate model
# Now we convert this project into a real time application where doctors can put in the feature values
# and the model predicts whether the person has diabetes or not
model = DecisionTreeClassifier()
model.fit(X_train_sm, y_train_sm)
name = input('Enter the patient\'s name: - ') # Takes in name
# The rest of the input lines take in the feature values in no particular order in the same datatype
# (These feature values are organised later)
age = float(input('Enter the patient\'s age: - '))
gender = int(input('Enter the patient\'s gender as 0. Female, 1. Male: - '))
bgl = int(input('Enter the patient\'s blood glucose level: - '))
hb = float(input('Enter the patient\'s HbA1c_level: - '))
hyper = int(input('Does the patient have hypertension: - '))
hd = int(input('Enter the patient\'s number of heart diseases: - '))
bmi = float(input('Enter the patient\'s BMI: - '))
sm = int(input('Enter the patient\'s smoking history as: - \n0. No Info\n1. current\n2. ever\n3. former\n4. never\n5. not current\n'))
print('\n')
# The values are first put in a dictionary
values = {'gender':[gender], 'age':[age], 'hypertension':[hyper], 'heart_disease':[hd], 'smoking_history':[sm], 'bmi':[bmi], 'HbA1c_level':[hb], 'blood_glucose_level':[bgl]}
# The dictionary is then converted to a dataframe
dataframe = pd.DataFrame(values)

sv = scaler.transform(dataframe)
# The model then predicts based on the input values
n = model.predict(sv)
print(n)
if n == 1:
    print(f'Your patient {name} has diabetes')
else:
    print(f'Your patient {name} does not have diabetes')

print('\n')
print('Thank you for using our diabetes prediction model')
