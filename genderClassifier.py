from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# height, weight, shoe size
X = [[181, 80, 40], [177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,55,38],[171,75,42],[181,85,43]]

y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

prediction_accuracy = []

# DECISION TREE

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

# prediction = clf.predict([[189,100,44]])
prediction = clf.predict(X)
print("Decision tree prediction")
print(prediction)
print("Prediction accuracy")
print(accuracy_score(y,prediction))
prediction_accuracy.append(['DT', accuracy_score(y,prediction)])


# LOGISTIC REGRESSION

lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr = lr.fit(X,y)

prediction = lr.predict(X)
print("Logistic regression prediction")
print(prediction)
print("Prediction accuracy")
print(accuracy_score(y,prediction))
prediction_accuracy.append(['LR', accuracy_score(y,prediction)])


# K-nearest neighbour

knn =  KNeighborsClassifier()
knn = knn.fit(X,y)

prediction = knn.predict(X)
print("KNN prediction")
print(prediction)
print("Prediction accuracy")
print(accuracy_score(y,prediction))
prediction_accuracy.append(['KNN',accuracy_score(y,prediction)])


# SVM

svm = SVC(gamma='auto')
svm = svm.fit(X,y)
prediction = svm.predict(X)

print("SVM prediction")
print(prediction)
print("Prediction accuracy")
print(accuracy_score(y,prediction))
prediction_accuracy.append(['SVM', accuracy_score(y,prediction)])

# Best performing

print("Performance of algorithms")
print(prediction_accuracy)