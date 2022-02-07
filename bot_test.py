import logging
from os import path, environ
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils.executor import start_webhook
from utils import get_list_of_styles, get_examples
from io import BytesIO

from config import TOKEN



API_TOKEN = environ.get("TOKEN")

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# В этом классе прописаны 3 состояния бота: Ожидания выбора стиля пользователем, ожидание фото от пользователя и ожидания переноса стиля
class BotStates(StatesGroup):
    waiting_select_style = State()
    waiting_photo = State()
    waiting_processing = State()


n_styles, style_names, user_style = 0, list(), 0
content_img = BytesIO()


def set_keyboard(condition=True):
    button_1 = types.InlineKeyboardButton(text="Выбрать стиль", callback="style")
    button_2 = types.InlineKeyboardButton(text="Фото", callback_data='photo')
    
    if condition is False:
        keyboard_markup = types.InlineKeyboardButton().add(button_1).add(button_2)
    else:
        pass

    return keyboard_markup


async def stylize(user_image):
    global user_style
    model = ModelStyle(user_style)
    model.load_model()
    output = model.run(user_image)
    del model
    gc.collect()
    return output


@dp.message_handler(state='*', commands=['/start' or '/help'])
async def send_welcome(message: types.Message):
    global user_style, style_name
    user_style = 0
    style_names = get_list_of_styles()
    n_styles = len(style_names)
    await BotStates.waiting_select_style.set()
    await message.reply(f"Приветсвую, *{message.from_user.username}*!\n", parse_mode='Markdown')
    await message.answer("Это бот для стилизации фотографий при помощи генеративно-сотязательной нейросети.\n"
                        f"Название странное - я так назван в честь одного из лучших фальсификаторов исскуства Новейшего времени в истории.\n"
                        f"Пока я могу стилизировать изображение только под картины Моне, на подходе Ван Гог, Гоген, Ренуар, Матисс\n"
                        f"Управление осуществляется кнопками ниже. Если возникают вопросы - воспользуйтесь командой /help\n")
    

@dp.message_handler(state=BotStates.waiting_select_style)
async def style_select(message: types.Message):
    """
    Обработчик выбора стиля, вызывается при указании номера стиля
    """
    global user_style, style_names, n_styles, content_img
    try:
        user_style = int(message.text.strip())
    except ValueError:
        await message.reply("Некорректный ввод данных\n")
        return
    if 0 < user_style <= n_styles:
        await message.answer(f"Ты выбрал стиль _{style_names[user_style - 1]}_.\n", parse_mode='Markdown')
        await BotStates.waiting_processing.set()
        await handle_go_processing(message)
    else:
        await message.reply("Некорректный ввод данных\n")
        await message.answer("Cтиля с таким номером не существует.\n")


@dp.message_handler(state=BotStates.waiting_photo, content_types=['photo'])
async def handle_photo(message: types.Message):
    """
    Вызывается при отправке пользователем фотографии
    """
    global content_img, user_style
    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    content_img = await bot.download_file(file_info.file_path)
    # content_img = Image.open(image_data)
    # print(type(image_data))
    await BotStates.waiting_select_style.set()
    await message.answer("Фотография загружена.\n", reply_markup=set_keyboard(False))

