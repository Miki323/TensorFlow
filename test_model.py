import tensorflow as tf
import matplotlib.pyplot as plt
import random
import PySimpleGUI as sg

# Загрузка данных MNIST
mnist = tf.keras.datasets.mnist
(_, _), (test_images, _) = mnist.load_data()

# Функция для отображения случайного тестового изображения
def show_random_image():
    # Выбор случайного тестового изображения
    index = random.randint(0, 9999)
    test_image = test_images[index]

    # Подготовка тестового изображения для модели
    test_image = test_image / 255.0  # Нормализация значений пикселей
    test_image = test_image.reshape((1, 28, 28))  # Преобразование в формат (1, 28, 28) - одно изображение

    # Загрузка сохраненной модели
    model = tf.keras.models.load_model("mnist_model.h5")

    # Предсказание на тестовом изображении
    predictions = model.predict(test_image)
    predicted_label = tf.argmax(predictions[0]).numpy()

    # Отображение тестового изображения
    plt.imshow(test_images[index], cmap='gray')
    plt.axis('off')  # Убираем оси с координатами

    # Сохраняем тестовое изображение в файл "temp.png"
    plt.savefig("temp.png")
    plt.close()

    # Создаем строку с кнопками "верно" и "не верно"
    buttons_layout = sg.HSeparator()
    layout = [
        [sg.Image(filename="temp.png")],
        [buttons_layout],
        [sg.Button("Верно", key="correct", size=(10, 2), pad=(5, 5)), sg.HSeparator(), sg.Button("Не верно", key="incorrect", size=(10, 2), pad=(5, 5))],
        [sg.Text(f"Однозначно цифра: {predicted_label}", size=(20, 1), justification='center', font='Any 12')],
    ]

    window = sg.Window("Тестирование модели", layout, finalize=True)

    # Отображение окна с изображением и кнопками
    event, _ = window.read()

    # Обработка нажатия на кнопку "Верно" или "Не верно"
    if event in (sg.WIN_CLOSED, 'Exit', 'incorrect', 'correct'):
        if event == 'correct':
            sg.popup("Обратная связь", "Спасибо за обратную связь! Я получил правильную информацию.")
        elif event == 'incorrect':
            sg.popup("Обратная связь", "Спасибо за обратную связь! Я еще учусь, ты делаешь меня лучше!")

    window.close()

# Отображаем первое случайное изображение
show_random_image()
