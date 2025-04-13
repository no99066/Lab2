import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Зчитування та попередня обробка даних
input_file = 'income_data.txt'
X_raw, y_raw = [], []
count_class1 = count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f:
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

# Розподіл даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Навчання моделі класифікації
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)

# Прогнозування результатів для тестової вибірки
y_pred = classifier.predict(X_test)

# Оцінювання точності моделі
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred, average='weighted') * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred, average='weighted') * 100, 2), "%")
print("F1 Score:", round(f1_score(y_test, y_pred, average='weighted') * 100, 2), "%")

# Приклад передбачення доходу для нових даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

input_data_encoded = []
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded.append(int(item))
    else:
        input_data_encoded.append(label_encoders[count].transform([item])[0])
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)
prediction = classifier.predict(input_data_encoded)

# Виведення прогнозованого результату
predicted_label = preprocessing.LabelEncoder().fit(['<=50K', '>50K']).inverse_transform(prediction)
print("Predicted:", predicted_label[0])
