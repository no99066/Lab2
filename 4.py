from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np

# Завантаження даних з файлу
input_file = 'income_data.txt'
X_raw, y_raw = [], []
count_class1 = count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as file:
    for line in file:
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X_raw.append(data[:-1])
            y_raw.append(data[-1])
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X_raw.append(data[:-1])
            y_raw.append(data[-1])
            count_class2 += 1

X_raw = np.array(X_raw)

# Кодування категоріальних змінних
label_encoders = []
X_encoded = np.empty(X_raw.shape)

for i in range(X_raw.shape[1]):
    if X_raw[0, i].isdigit():
        X_encoded[:, i] = X_raw[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X_raw[:, i])
        label_encoders.append(encoder)

X = X_encoded.astype(int)
y = preprocessing.LabelEncoder().fit_transform(y_raw)

# Розділення даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Ініціалізація моделей
models = []
models.append(('LR', OneVsRestClassifier(LogisticRegression())))  # Логістична регресія
models.append(('LDA', LinearDiscriminantAnalysis()))  # Лінійний дискримінантний аналіз
models.append(('KNN', KNeighborsClassifier()))  # Алгоритм найближчих сусідів
models.append(('CART', DecisionTreeClassifier()))  # Дерево рішень
models.append(('NB', GaussianNB()))  # Байєсівський класифікатор
models.append(('SVM', SVC(gamma='auto')))  # Метод опорних векторів (SVM)

# Оцінка моделей за допомогою крос-валідації
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})')