import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

# Завантаження набору даних
file_path = 'income_data.txt'
raw_features = []
raw_labels = []
limit_per_class = 25000
count_1 = count_2 = 0

# Зчитування та фільтрація даних
with open(file_path, 'r') as file:
    for line in file:
        if count_1 >= limit_per_class and count_2 >= limit_per_class:
            break
        if '?' in line:
            continue
        row = line.strip().split(', ')
        label = row[-1]
        if label == '<=50K' and count_1 < limit_per_class:
            raw_features.append(row[:-1])
            raw_labels.append(label)
            count_1 += 1
        elif label == '>50K' and count_2 < limit_per_class:
            raw_features.append(row[:-1])
            raw_labels.append(label)
            count_2 += 1

# Перетворення у масиви NumPy
X = np.array(raw_features)
y = np.array(raw_labels)

# Кодування категоріальних ознак
X_transformed = np.empty(X.shape)
encoders = []

for idx in range(X.shape[1]):
    if X[0, idx].isdigit():
        X_transformed[:, idx] = X[:, idx].astype(int)
    else:
        encoder = preprocessing.LabelEncoder()
        X_transformed[:, idx] = encoder.fit_transform(X[:, idx])
        encoders.append(encoder)

# Поділ на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Функція для оцінки моделі та виводу результатів
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted')
    rec = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print("\n Model Evaluation Results:")
    print(f"- Accuracy: {acc * 100:.2f}%")
    print(f"- Precision: {prec * 100:.2f}%")
    print(f"- Recall: {rec * 100:.2f}%")
    print(f"- F1 Score: {f1 * 100:.2f}%\n")

# Функція для тестування SVM з різними ядрами
def test_svm_with_kernel(kernel, degree=3):
    print(f"Testing SVM with kernel: {kernel}")
    if kernel == 'poly':
        model = SVC(kernel='poly', degree=degree)
    elif kernel == 'rbf':
        model = SVC(kernel='rbf')
    elif kernel == 'sigmoid':
        model = SVC(kernel='sigmoid')
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    evaluate_model(model, X_train, X_test, y_train, y_test)

# Запуск оцінки для різних ядер
test_svm_with_kernel('poly', degree=2)
test_svm_with_kernel('rbf')
test_svm_with_kernel('sigmoid')