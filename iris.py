# Author: Endri Dibra

# importing the required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# loading the dataset to be used
iris = load_iris()

# loading the data
X = iris.data
y = iris.target

f_names = iris.feature_names
t_names = iris.target_names

print("Feature names: ", f_names)
print("Target names: ", t_names)

print("Below are the 5 first rows of the dataset:")
print(X[:5])

# starting to create and train the knn model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# calculating accuracy of knn model
y_predictions = model.predict(X_test)
print("Accuracy of knn model: ", metrics.accuracy_score(y_test, y_predictions))

# making predictions
data = [[5, 2, 4, 3], [3, 5, 2, 4]]

data_predictions = model.predict(data)
species_predictions = [t_names[i] for i in data_predictions]

print("Predictions about species: ", species_predictions)