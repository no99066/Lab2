from sklearn.datasets import load_iris

# Завантаження класичного набору даних Iris
iris_data = load_iris()

# Перелік доступних ключів у наборі даних
print("Dataset keys:\n{}".format(iris_data.keys()))

# Короткий опис датасету
print(iris_data['DESCR'][:193] + "\n...")

# Імена можливих класів (відповідей)
print("Target names: {}".format(iris_data['target_names']))

# Назви характеристик (ознак), що використовуються для класифікації
print("Feature names:\n{}".format(iris_data['feature_names']))

# Тип структури, що містить дані ознак
print("Type of data array: {}".format(type(iris_data['data'])))

# Розміри масиву даних (кількість прикладів та ознак)
print("Shape of data array: {}".format(iris_data['data'].shape))

# Тип структури, що містить цільові значення
print("Type of target array: {}".format(type(iris_data['target'])))

# Значення цільових класів для кожного прикладу
print("Target labels:\n{}".format(iris_data['target']))
