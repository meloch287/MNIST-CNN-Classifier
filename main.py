import tensorflow as t 
from tensorflow.keras.models import Sequential as S  
from tensorflow.keras.layers import Dense as D, Flatten as F, Conv2D as C, MaxPooling2D as M, Dropout as Dr, BatchNormalization as B
from tensorflow.keras.datasets import mnist as mn 
from tensorflow.keras.callbacks import ModelCheckpoint as MC, ReduceLROnPlateau as RL, EarlyStopping as ES
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG  
from tensorflow.keras.regularizers import l2 as l2  
import numpy as n  

t.random.set_seed(42) 
n.random.seed(42) 

def ld():
    '''
    Функция для загрузки и предобработки данных MNIST
    '''
    (x, y), (xt, yt) = mn.load_data()  
    x = x.astype('float32').reshape(-1, 28, 28, 1) / 255.0  # Преобразуем в float32, меняем форму на (кол-во, 28, 28, 1) и нормализуем до [0, 1]
    xt = xt.astype('float32').reshape(-1, 28, 28, 1) / 255.0 
    return (x, y), (xt, yt)  # Возвращаем обработанные данные как кортеж


def gd():
    '''
    Функция для создания генератора данных с аугментацией
    '''
    return IDG(
        rotation_range=10,  # Случайное вращение изображений до 10 градусов
        zoom_range=0.15,  # Случайное масштабирование до 15%
        width_shift_range=0.15,  # Случайный сдвиг по ширине до 15%
        height_shift_range=0.15,  # Случайный сдвиг по высоте до 15%
        shear_range=0.15,  # Случайный наклон до 15%
        fill_mode='nearest'  # Заполнение пустых областей ближайшими пикселями
    )  # Возвращаем настроенный генератор данных


def md():
    '''
    Функция для создания свёрточной нейронной сети
    '''
    m = S([  
        # Первый свёрточный блок
        C(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.0005)),
        # 32 свёрточных фильтра размером 3x3, padding='same' сохраняет размер входа, активация ReLU, вход (28, 28, 1), L2 регуляризация
        B(),  # Нормализация 
        C(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0005)),  # Ещё один свёрточный слой с 32 фильтрами
        B(),  # Нормализация
        M(pool_size=(2, 2)),  # Пулинг 2x2 для уменьшения размерности в 2 раза
        Dr(0.25),  # Dropout 25% для предотвращения переобучения

        # Второй свёрточный блок
        C(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0005)),  # 64 фильтра 3x3
        B(),  # Нормализация
        C(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0005)),  # Ещё один свёрточный слой с 64 фильтрами
        B(),  # Нормализация
        M(pool_size=(2, 2)),  # Пулинг 2x2
        Dr(0.25),  # Dropout 25%

        # Полносвязный блок
        F(),  # Преобразование многомерного тензора в одномерный вектор
        D(256, activation='relu', kernel_regularizer=l2(0.0005)),  # Полносвязный слой с 256 нейронами и ReLU
        B(),  # Нормализация
        Dr(0.5),  # Dropout 50% для регуляризации
        D(10, activation='softmax')  # Выходной слой с 10 классами (цифры 0-9) и активацией softmax
    ])
    return m 

def m():
    '''
    Основная функция для обучения и оценки модели
    '''
    (x, y), (xt, yt) = ld()  # Загружаем и предобрабатываем данные
    dg = gd()  # Создаём генератор данных
    dg.fit(x)  # Подгоняем генератор под обучающие данные
    m = md()  # Создаём модель
    m.compile(
        optimizer=t.keras.optimizers.Adam(learning_rate=0.001),  # Оптимизатор Adam со скоростью обучения 0.001
        loss='sparse_categorical_crossentropy',  # Функция потерь для многоклассовой классификации с целыми метками
        metrics=['accuracy']  # Отслеживаем метрику точности
    )
    cb = [  
        MC('best_digit_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        RL(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ES(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]
    h = m.fit(
        dg.flow(x, y, batch_size=128), 
        epochs=50, 
        validation_data=(xt, yt),  
        callbacks=cb, 
        verbose=1
    )
    tl, ta = m.evaluate(xt, yt, verbose=0)  # Оцениваем модель на тестовых данных (потери и точность)
    print(f"\nTest accuracy: {ta:.4f}")  
    print(f"Test loss: {tl:.4f}")
    return m, h  # Возвращаем обученную модель и историю обучения

if __name__ == "__main__":
    m, h = m()  
