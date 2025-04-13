# Імпортуємо необхідні бібліотеки
import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Завантаження датасету Iris
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = read_csv(url, names=column_names)

# Перевірка розміру таблиці
print("Dataset dimensions:", data.shape)

# Перегляд перших 20 записів
print("Sample data:\n", data.head(20))

# Статистичний опис кожного атрибуту
print("\nStatistics summary:\n", data.describe())

# Кількість прикладів у кожному класі
print("\nClass distribution:\n", data.groupby('class').size())

# Побудова діаграм розмаху
data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# Побудова гістограм
data.hist()
pyplot.show()

# Побудова матриці розсіювання
scatter_matrix(data)
pyplot.show()

# Розділення даних на ознаки та ціль
X = data.values[:, 0:4]
y = data.values[:, 4]

# Поділ на тренувальні і тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Список моделей для оцінки
model_list = [
    ('LR', OneVsRestClassifier(LogisticRegression())),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

# Крос-валідація моделей
cv_results = []
model_names = []

for name, model in model_list:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    cv_results.append(scores)
    model_names.append(name)
    print(f"{name}: {scores.mean():.6f} ({scores.std():.6f})")

# Візуалізація результатів крос-валідації
pyplot.boxplot(cv_results, labels=model_names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Тестування моделі SVM
final_model = SVC(gamma='auto')
final_model.fit(X_train, y_train)
predictions = final_model.predict(X_test)

# Виведення результатів оцінки
print("\nModel Performance on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, predictions):.6f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))

# Прогноз для нового прикладу
sample_input = np.array([[5.0, 3.0, 1.5, 0.2]])
final_model.fit(X_train, y_train)
sample_prediction = final_model.predict(sample_input)

print("\nPrediction for new sample:")
print("==========================")
print(f"Predicted Class: {sample_prediction[0]}")
print("==========================")
