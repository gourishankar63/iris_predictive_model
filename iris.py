import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

if df.isnull().sum().sum() > 0:
    df = df.fillna(df.mean())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[data.feature_names] = scaler.fit_transform(df[data.feature_names])

X = df[data.feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

new_data = [[5.1, 3.5, 1.4, 0.2]]
new_data_scaled = scaler.transform(new_data)
new_prediction = model.predict(new_data_scaled)
print(f"Prediction for new data: {data.target_names[new_prediction][0]}")

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

feature_importances = pd.Series(model.feature_importances_, index=data.feature_names).sort_values(ascending=False)
feature_importances.plot(kind='bar', title='Feature Importance')
plt.ylabel('Importance')
plt.show()

def plot_decision_boundary(X, y, model, feature_names):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ['red', 'green', 'blue']

    X = X.iloc[:, :2]
    model.fit(X, y)

    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor='k', s=20, cmap=ListedColormap(cmap_bold))
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Decision Boundary')
    plt.show()

plot_decision_boundary(X, y, model, data.feature_names[:2])

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

X = df[data.feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.random.seed(42)
noise = np.random.normal(0, 0.2, X_train.shape)
X_train_noisy = X_train + noise

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=20, min_samples_leaf=10),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=200, C=93),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=1),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=5, max_depth=3)
}

results = {}
for name, model in models.items():
    model.fit(X_train_noisy, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

for model_name, accuracy in results.items():
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")

model_names = list(results.keys())
accuracies = list(results.values())

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='lightcoral')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim([0, 1])
plt.show()