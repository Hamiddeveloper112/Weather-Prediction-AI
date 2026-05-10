import pandas as pd
import numpy as np
data = pd.read_csv("weather.csv")
print(data.head())
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})
X = data[['Temperature', 'Humidity', 'WindSpeed', 'Pressure']]
y = data['RainTomorrow']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)

print("Naive Bayes Accuracy:", nb_accuracy)
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_accuracy)
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_accuracy)

print("\nModel Accuracy Comparison")
print("-------------------------")
print("Naive Bayes        :", nb_accuracy)
print("Logistic Regression:", lr_accuracy)
print("Decision Tree      :", dt_accuracy)
