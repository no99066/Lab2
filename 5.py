import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Завантажуємо датасет Iris
iris_data = load_iris()
features, labels = iris_data.data, iris_data.target

# Розподіл даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)

# Ініціалізація та навчання класифікатора Ridge
ridge_classifier = RidgeClassifier(tol=1e-2, solver="sag")
ridge_classifier.fit(X_train, y_train)

# Прогнозування результатів на тестовому наборі
y_predicted = ridge_classifier.predict(X_test)

# Виведення метрік оцінки
print('Accuracy:', np.round(metrics.accuracy_score(y_test, y_predicted), 4))
print('Precision:', np.round(metrics.precision_score(y_test, y_predicted, average='weighted'), 4))
print('Recall:', np.round(metrics.recall_score(y_test, y_predicted, average='weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(y_test, y_predicted, average='weighted'), 4))
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(y_test, y_predicted), 4))
print('Matthews Correlation Coefficient:', np.round(metrics.matthews_corrcoef(y_test, y_predicted), 4))

# Звіт про класифікацію
print('\t\tClassification Report:\n', metrics.classification_report(y_test, y_predicted))

# Створення матриці плутанини
conf_matrix = confusion_matrix(y_test, y_predicted)

# Візуалізація матриці плутанини
sns.set()
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()



