from catboost import CatBoostRegressor
import numpy as np

try:
    import Image
except ImportError:
    from PIL import Image

import telebot
from telebot import types

from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense

# telegram bot token
bot = telebot.TeleBot('')  # telegram bot

# ml model to predict realty price by user data
ctb_model = CatBoostRegressor()
ctb_model.load_model('model/ctb_model')

# count starts
start_count = 0


# ml model to predict realty price by photo
def get_model():
    model = Sequential()
    model.add(Conv2D(10, 1, 1, input_shape=(256, 256, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(10, 1, 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.load_weights("model/cnn_model.h5")
    model.compile(loss='mse', optimizer='rmsprop')

    return model


@bot.message_handler(commands=["start"])
def start(message):
    global user_state, start_count
    user_state = None
    bot.send_message(message.chat.id, "Добро пожаловать в Raif 2018 Realty Bot!")
    start_count += 1
    print(start_count)
    menu(message)


@bot.message_handler(commands=["menu"])
def menu(message):
    keyboard_menu = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    keyboard_menu.row('Проверить стоимость квартиры')
    keyboard_menu.row('Анализ квартиры по фотографии')
    bot.send_message(message.chat.id, "Выберите пункт меню:", reply_markup=keyboard_menu)


@bot.message_handler(func=lambda message: message.text == "Проверить стоимость квартиры")
def search_event(m):
    global user_state
    keyboard_hider = types.ReplyKeyboardRemove()
    user_state = 'searching'
    bot.send_message(m.chat.id, "Пожалуйста, через запятую введите следующую информацию о квартире: " +
                     "'Жилая площадь', 'Этажность дома', 'Этаж', 'Количество комнат', " +
                     "'Расстояние до метро', 'Возраст дома'.",
                     reply_markup=keyboard_hider)


@bot.message_handler(func=lambda message: user_state == 'searching')
def searching(m):
    global user_state
    user_state = 'searched'
    vals = m.text.split(',')

    try:
        living_square = int(vals[0].strip())
        house_story = int(vals[1].strip())
        story = int(vals[2].strip())
        rooms = int(vals[3].strip())
        house_age = int(vals[5].strip())
    except Exception as e:
        print(e)
        bot.send_message(m.chat.id, "Что-то пошло не так! Пожалуйста, попробуйте еще раз.")
        menu(m)
        return

    pred_vals = [
        [rooms, house_age, 23, 0, living_square + 20, living_square, living_square - 20, story, house_story, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    price = ctb_model.predict(pred_vals)
    bot.send_message(m.chat.id, "Цена квартиры: %s рублей" % int(price * living_square))
    menu(m)


@bot.message_handler(func=lambda message: message.text == "Анализ квартиры по фотографии")
def request_photo(m):
    keyboard_hider = types.ReplyKeyboardRemove()
    bot.send_message(m.chat.id, "Добавьте фотографию квартиры",
                     reply_markup=keyboard_hider)


@bot.message_handler(content_types=['photo'])
def handle_photo(m):
    global user_state

    fileID = m.photo[-1].file_id
    file = bot.get_file(fileID)
    downloaded_file = bot.download_file(file.file_path)
    with open('data/bot_images/temp.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)

    image = np.asarray(Image.open('data/bot_images/temp.jpg'))
    x_shape = image.shape[0]
    y_shape = image.shape[1]
    X = []
    for x_i_start in range(0, x_shape, 256):
        x_i_end = x_i_start + 256
        if x_i_end < x_shape:
            for y_i_start in range(0, y_shape, 256):
                y_i_end = y_i_start + 256
                if y_i_end < y_shape:
                    local_image = image[x_i_start: x_i_end, y_i_start: y_i_end]
                    X.append(local_image)

    X = np.asarray(X)
    cnn_model = get_model()
    predicted_price = np.mean(cnn_model.predict(X))

    bot.send_message(m.chat.id, "Цена квартиры с таким ремонтом: %s тыс. рублей за кв. метр" % int(predicted_price))
    menu(m)


while True:
    try:
        bot.skip_pending = True
        bot.polling(none_stop=True)
    except Exception as err:
        print(err)
        pass
