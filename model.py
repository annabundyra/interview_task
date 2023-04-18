import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# reading csv files
column_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Soil_Type', 'Cover_Type']
data = pd.read_csv('covtype.data', sep=",", names = column_names)
print(data)

def heuristic_classifier(data):
    X = data[['Horizontal_Distance_To_Fire_Points']]
    y = data[['Soil_Type']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    X_train_absence = X_train[y_train == 0]

    avg_distance = np.mean(X_train_absence['Horizontal_Distance_To_Fire_Points'])
    pred = X_test['Horizontal_Distance_To_Fire_Points'] > avg_distance
    heuristic_acc = accuracy_score(y_test, pred)
    print(f"Heuristic accuracy: {heuristic_acc:.4f}")
    return heuristic_acc

def decission_tree(data):
    X = data.drop(['Soil_Type'], axis=1)
    y = data['Soil_Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    tree_acc = tree_clf.score(X_test, y_test)
    print(f"Decision tree accuracy: {tree_acc:.4f}")
    return tree_acc

def logistic_regression(data):
    X = data.drop(['Soil_Type'], axis=1)
    y = data['Soil_Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    log_reg_acc = log_reg.score(X_test, y_test)
    print(f"Logistic regression accuracy: {log_reg_acc:.4f}")
    return log_reg_acc

print(data.columns[data.isnull().any()])
#no missing values so we dont have to input tthem
#checking Soil Type Dessignation to Horz Dist to nearest wildfire ignition points
heur_acc = heuristic_classifier(data)
tree_acc = decission_tree(data)
log_acc = logistic_regression(data)

models = ['Heuristic', 'Decission Tree', 'Logistic Regression']
acc = [heur_acc, tree_acc, log_acc]


plt.bar(models, acc)
plt.title('Accuracy')
plt.show()