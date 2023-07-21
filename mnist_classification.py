import tensorflow as tf
from tensorflow.keras import layers, models

# Загрузка данных MNIST
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормализация данных и преобразование меток в формат one-hot
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Создание архитектуры нейронной сети
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))  # Преобразование 28x28 в одномерный массив
model.add(layers.Dense(128, activation='relu'))  # Полносвязный слой с 128 нейронами и функцией активации ReLU
model.add(layers.Dropout(0.2))  # Dropout для регуляризации и предотвращения переобучения
model.add(layers.Dense(10, activation='softmax'))  # Выходной слой с 10 нейронами и функцией активации Softmax

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Сохранение обученной модели
model.save("mnist_model.h5")

# Оценка производительности модели
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print(f'Точность на тестовых данных: {test_accuracy}')
print(f'Потери на тестовых данных: {test_loss}')
