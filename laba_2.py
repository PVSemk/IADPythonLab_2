from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import datetime
import re
import pymorphy2
from time import time
from memory_profiler import memory_usage


# Блок объявления констант
MORPH = pymorphy2.MorphAnalyzer()
TWEETS_AMOUNT = 10000.0
FREQUENCY_LIMIT = 10


# Блок объявления глобальных переменных
positive_adj_dict = {}
negative_adj_dict = {}
tweets = []
frequency_dict = {}
estimation_dict = {}
# Создаем стоп-лист и добавляем туда некоторые позиции
stop_words = set(stopwords.words("russian"))
stop_words.update({'россияхорватия'})


# Функция, очищающая данная от мусора, основанная на регулярных выражениях
def del_rubbish(raw_data):
    result = re.sub(r'[^а-яА-ЯеЁ\s]*\d*', '', raw_data)
    return result


# Функция, выполняющая стемминг при помощи библиотеки pymorphy2
def norm(raw_word):
    stemmed_word = MORPH.parse(raw_word)[0]
    return stemmed_word.normal_form


# Функция для создания и ссхранения одиночного столбчатого графика
def create_single_bar_graph(name, x, y, title='', legend='Tweets Amount', save=False, fmt='png'):
    x_pos = np.arange(len(x))
    # Создаем график
    bar = plt.bar(x_pos, y, width=0.7, align='center')
    plt.xticks(x_pos, x)
    plt.ylabel('Amount')
    plt.title(title)
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, height, ha='center', va='bottom')
    plt.legend([legend])
    # Сохраняем график если хотим
    if save:
        plt.savefig('{}.{}'.format(name, fmt), fmt='png')
        plt.close()


def first_estimation_rule(tweets, save=True):
    # Первое правило - сумма оценок слов в твите
    # По сумме оценки определяем окраску всего твита
    positive_tweets, negative_tweets, neutral_tweets = 0, 0, 0
    for tweet in tweets:
        estimation_sum = 0
        for word in tweet.split()[2:]:
            # Слова, которые встретились в менее чем 10 твитах на оценку влияния не окажут
            if word in estimation_dict.keys():
                estimation_sum += estimation_dict[word]
        if estimation_sum <= -2:
            negative_tweets += 1
        elif estimation_sum >= 2:
            positive_tweets += 1
        else:
            neutral_tweets += 1
    if save:
        # Записываем результаты в файл
        file.write('''Estimation summary\n
Good - {} - {}%\n
Bad - {} - {}%\n
Neutral - {} - {}%\n\n'''.format(positive_tweets, positive_tweets / TWEETS_AMOUNT,
                                 negative_tweets, negative_tweets / TWEETS_AMOUNT,
                                 neutral_tweets, neutral_tweets / TWEETS_AMOUNT))
        # Создаем график распределения оценок для первого правила
        # Сохраняем его в файл
        # Подписи столбцов по Ox
        x = ('Positive tweets', 'Negative tweets', 'Neutral tweets')
        # Значения столбцов
        y = [positive_tweets, negative_tweets, neutral_tweets]
        # Название графика и имя файла для сохранения
        title = 'Summary rule'
        # Создаем график
        create_single_bar_graph('Summary', x, y, title, save=True)
    return [positive_tweets, neutral_tweets, negative_tweets]


# Второе правило - доля слов каждого типа в твите
# Итоговая окраска - тип с наибольшей долей
# Если доли получаются равными, то тип - undefined
def second_estimation_rule(tweets, save=True):
    positive_tweets, negative_tweets, neutral_tweets, undefined_tweets = 0, 0, 0, 0
    for tweet in tweets:
        valuable_len = 0
        positive_proportion, negative_proportion, neutral_proportion = 0, 0, 0
        for word in tweet.split()[2:]:
            if word in estimation_dict.keys():
                if estimation_dict[word] == 1:
                    positive_proportion += 1
                elif estimation_dict[word] == 0:
                    neutral_proportion += 1
                elif estimation_dict[word] == -1:
                    negative_proportion += 1
                # Подсчитываем количество размеченных слов в твите
                valuable_len += 1
        # Возможна ситуация, что размеченных слов в твите нет
        if valuable_len == 0:
            undefined_tweets += 1
            continue
        else:
            positive_proportion = positive_proportion / float(valuable_len)
            negative_proportion = negative_proportion / float(valuable_len)
            neutral_proportion = neutral_proportion / float(valuable_len)
        if positive_proportion > negative_proportion and positive_proportion > neutral_proportion:
            positive_tweets += 1
        elif negative_proportion > positive_proportion and negative_proportion > neutral_proportion:
            negative_tweets += 1
        elif neutral_proportion > positive_proportion and neutral_proportion > negative_proportion:
            neutral_tweets += 1
        else:
            undefined_tweets += 1
    if save:
        file.write('''Proportion\n
Good - {} - {}%\n
Bad - {} - {}%\n
Neutral - {} - {}%\n
Undefined - {} - {}%\n\n'''.format(positive_tweets, positive_tweets / TWEETS_AMOUNT,
                                   negative_tweets, negative_tweets / TWEETS_AMOUNT,
                                   neutral_tweets, neutral_tweets / TWEETS_AMOUNT,
                                   undefined_tweets, undefined_tweets / TWEETS_AMOUNT))

        # Создаем график распределения оценок для второго правила
        # Сохраняем его в файл
        # Подписи столбцов по Ox
        x = ('Positive tweets', 'Negative tweets', 'Neutral tweets', 'Undefined tweets')
        # Значения столбцов
        y = [positive_tweets, negative_tweets, neutral_tweets, undefined_tweets]
        # Заголовок и имя файла
        title = 'Proportion rule'
        # Создаем график
        create_single_bar_graph('Proportion', x, y, title, save=True)
    return [positive_tweets, neutral_tweets, negative_tweets]


# Третье правило - отношение положительных к отрицательным лежит в определенном интервале
def third_estimation_rule(tweets, save=True):
    positive_tweets, negative_tweets, neutral_tweets = 0, 0, 0
    for tweet in tweets:
        positive_proportion, negative_proportion, neutral_proportion = 0, 0, 0
        for word in tweet.split()[2:]:
            if word in estimation_dict.keys():
                if estimation_dict[word] == 1:
                    positive_proportion += 1
                elif estimation_dict[word] == -1:
                    negative_proportion += 1
        # Смотрим значение
        if negative_proportion != 0 and 1.5 < positive_proportion / negative_proportion:
            positive_tweets += 1
        elif negative_proportion != 0 and 0 < positive_proportion / negative_proportion < 0.7:
            negative_tweets += 1
        elif negative_proportion == 0 and positive_proportion != 0:
            positive_tweets += 1
        else:
            neutral_tweets += 1
    if save:
        # Записываем результаты в файл
        file.write('''Pos/neg summary\n
Good - {} - {}%\n
Bad - {} - {}%\n
Neutral - {} - {}%\n\n'''.format(positive_tweets, positive_tweets / TWEETS_AMOUNT,
                                 negative_tweets, negative_tweets / TWEETS_AMOUNT,
                                 neutral_tweets, neutral_tweets / TWEETS_AMOUNT))
        # Создаем график распределения оценок для первого правила
        # Сохраняем его в файл
        # Подписи столбцов по Ox
        x = ('Positive tweets', 'Negative tweets', 'Neutral tweets')
        # Значения столбцов
        y = [positive_tweets, negative_tweets, neutral_tweets]
        # Название графика и имя файла для сохранения
        title = 'Pos/Neg Rule'
        # Создаем график
        create_single_bar_graph('Pos_neg', x, y, title, save=True)
    return [positive_tweets, neutral_tweets, negative_tweets]


# Четвертое правило - сравниваем отношение положительных к нейтральным и отрицательных к нейтральным
def fourth_estimation_rule(tweets, save=True):
    positive_tweets, negative_tweets, neutral_tweets, undefined_tweets = 0, 0, 0, 0
    for tweet in tweets:
        positive_proportion, negative_proportion, neutral_proportion = 0, 0, 0
        for word in tweet.split()[2:]:
            if word in estimation_dict.keys():
                if estimation_dict[word] == 1:
                    positive_proportion += 1
                elif estimation_dict[word] == -1:
                    negative_proportion += 1
                else:
                    neutral_proportion += 1
        # Если количество нейтральных слов - 0, то неопределенность
        if neutral_proportion != 0:
            positive_proportion = positive_proportion / float(neutral_proportion)
            negative_proportion = negative_proportion / float(neutral_proportion)
            if positive_proportion > negative_proportion:
                positive_tweets += 1
            elif negative_proportion > positive_proportion:
                negative_tweets += 1
            else:
                neutral_tweets += 1
        else:
            undefined_tweets += 1
    if save:
        file.write('''Proportion\n
Good - {} - {}%\n
Bad - {} - {}%\n
Neutral - {} - {}%\n
Undefined - {} - {}%\n\n'''.format(positive_tweets, positive_tweets / TWEETS_AMOUNT,
                                   negative_tweets, negative_tweets / TWEETS_AMOUNT,
                                   neutral_tweets, neutral_tweets / TWEETS_AMOUNT,
                                   undefined_tweets, undefined_tweets / TWEETS_AMOUNT))
        # Создаем график распределения оценок для первого правила
        # Сохраняем его в файл
        # Подписи столбцов по Ox
        x = ('Positive tweets', 'Negative tweets', 'Neutral tweets', 'Undefined tweets')
        # Значения столбцов
        y = [positive_tweets, negative_tweets, neutral_tweets, undefined_tweets]
        # Название графика и имя файла для сохранения
        title = 'Pos/Neu and Neg/Neu Rule'
        # Создаем график
        create_single_bar_graph('Pos_neu_and_Neg_neu', x, y, title, save=True)
    return [positive_tweets, neutral_tweets, negative_tweets]


# Засекаем время
toc = time()

# Считываем raw data
with open('data.txt', 'r', encoding='utf-8') as file:
    count = 0
    for line in file:
        count += 1
        if count == 2 * TWEETS_AMOUNT:
            break
        if line == '\n':
            continue
        # Удаляем лишние символы, сохраняем дату, приводим строку к единому формату
        clear_line = line[0:17].replace('\ufeff', '') + del_rubbish(line[17:]).replace('_', ' ').replace('\n', '').lower()
        tweets.append(clear_line)


# Осуществляем стемминг, переписываем исходные твитты после этой операции
for index, tweet in enumerate(tweets):
    new_tweet = ''
    # С целью убрать повторные слова в твите, преобразуем лист слов во множество
    for word in tweet.split():
        # Проверка на предлоги, частицы, etc
        if word not in stop_words:
            normed_word = norm(word)
            new_tweet += normed_word + ' '
    tweets[index] = new_tweet


# Создаем частотный словарь
with open('frequency.txt', 'w', encoding='utf-8') as file:
    for tweet in tweets:
        # Срез берем с третьего элемента, поскольку первые две позиции занимают дата и время
        for word in set(tweet.split()[2:]):
            if word in frequency_dict.keys():
                frequency_dict[word] += 1
            else:
                frequency_dict[word] = 1
    # Сортируем по возрастанию значений
    frequency_dict = sorted(frequency_dict.items(), key=lambda kv: kv[1], reverse=True)
    # Считаем процент вхождений и записываем всё в файл
    for word, count in frequency_dict:
        if count >= FREQUENCY_LIMIT:
            percent = count / TWEETS_AMOUNT * 100
            file.write('{} - {} - {}%\n'.format(word, count, round(percent, 2)))


# Считываем файл, содержащий оценки слов
# А также заполняем словари, содержащие негативно или позитивно оцененные прилагательные
with open('estimations.txt', 'r', encoding='utf-8') as file:
    for line in file:
        word = line.split()[0].replace('\ufeff', '')
        grade = int(line.split()[1].replace('\n', ''))
        estimation_dict[word] = grade
        stemmed_word = MORPH.parse(word)[0]
        if stemmed_word.tag.POS == 'ADJF' and grade == 1:
            positive_adj_dict[stemmed_word.normal_form] = 0
        elif stemmed_word.tag.POS == 'ADJF' and grade == -1:
            negative_adj_dict[stemmed_word.normal_form] = 0


# Создаем файл, содержащий итоги оценки твитов и заполняем его
with open('classifications.txt', 'w', encoding='utf-8') as file:
    # Первое правило
    first_estimation_rule(tweets)
    # Второе правило
    second_estimation_rule(tweets)
    # Третье правило
    third_estimation_rule(tweets)
    # Четвертое правило
    fourth_estimation_rule(tweets)


# Подсчитываем количество положительных и отрицательных прилагательных
# Сортируем по убыванию и записываем в листы
for tweet in tweets:
    for word in set(tweet.split()[2:]):
        if word in positive_adj_dict.keys():
            positive_adj_dict[word] += 1
        elif word in negative_adj_dict.keys():
            negative_adj_dict[word] += 1
positive_adj_list = sorted(positive_adj_dict.items(), key=lambda kv: kv[1], reverse=True)
negative_adj_list = sorted(negative_adj_dict.items(), key=lambda kv: kv[1], reverse=True)
del negative_adj_dict
del positive_adj_dict

# Записываем полученный итог в файл
with open('adjectives.txt', 'w', encoding='utf-8') as file:
    file.write('Top-5 Positive\n')
    for word, amount in positive_adj_list[:5]:
        file.write('{}-{}-{}%\n'.format(word, amount, amount/TWEETS_AMOUNT))
    file.write('\nTop-5 Negative\n')
    for word, amount in negative_adj_list[:5]:
        file.write('{} - {} - {}%\n'.format(word, amount, amount/TWEETS_AMOUNT))

# Создаем графики, показывающие топ-5 положительных и отрицательных прилагательных
# Подписи Ox
x = ('1st', '2nd', '3d', '4th', '5th')
x_pos = np.arange(len(x))
# Толщина столбца
width = 0.35
# Значения по Oy
y_positive = [adj[1] for adj in positive_adj_list[:5]]
y_negative = [adj[1] for adj in negative_adj_list[:5]]
# Графики
bar_positive = plt.bar(x_pos, y_positive, width=width, label='Positive')
for rect in bar_positive:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, height, ha='center', va='bottom')
bar_negative = plt.bar(x_pos + width, y_negative, width=width, label='Negative', color='r')
for rect in bar_negative:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, height, ha='center', va='bottom')
plt.ylabel('Amount')
plt.xticks(x_pos + width / 2, x)
plt.title('Top-5 Positive/Negative Adjectives')
plt.legend()
plt.savefig('Top_adjectives.png')


# Оцениваем распределение по времени
# Генерируем словарь, где ключи - временные промежутки
# Window - 1h, step - 30m
time_dict = {'0:00 - {}:{}0'.format(hour, minute): [0, 0, 0, 0] for hour in range(1, 24) for minute in range(0, 4, 3)}
# Нижнее окно по времени (00:00)
appropriate_tweets = []
# Значения по y для второго графика
y_2 = []
# Значения по y для первого графика
y_pos_1 = []
y_neu_1 = []
y_neg_1 = []
tweet_time_min = datetime.datetime.strptime('0:00', '%H:%M')
# Записываем распределение по времени в файл
# Для первой итерации предыдущий ключ вручную установлен на любое поле с нулевым значением
previous_key = '0:00 - 1:30'
with open('hours.txt', 'w', encoding='utf-8') as file:
    # Пробегаем  по всем возможным окнам
    for key in time_dict.keys():
        # Высчитываем текущую верхнюю границу окна
        tweet_time_max = datetime.datetime.strptime(key.split(' - ')[1], '%H:%M')
        for tweet in tweets:
            # Ищем время публикации твита
            tweet_time = datetime.datetime.strptime(tweet.split()[1], '%H:%M')
            # Смотрим, что принадлежит нашему окну
            if tweet_time_min <= tweet_time < tweet_time_max:
                appropriate_tweets.append(tweet)
        # По первому правилу подсчитываем число положительных/нейтральных/отрицательных
        time_dict[key] = first_estimation_rule(appropriate_tweets, save=False)
        appropriate_tweets_number = len(appropriate_tweets)
        time_dict[key].append(appropriate_tweets_number)
        file.write('{} : {} {}/{}/{}\n'.format(key, appropriate_tweets_number,
                                               round(time_dict[key][0] / float(appropriate_tweets_number), 2),
                                               round(time_dict[key][1] / float(appropriate_tweets_number), 2),
                                               round(time_dict[key][2] / float(appropriate_tweets_number), 2)))
        # Рассчитываем значения (разница между текущим и предыдущим) для распределения по окнам
        positive_proportion = time_dict[key][0] - time_dict[previous_key][0]
        negative_proportion = time_dict[key][1] - time_dict[previous_key][1]
        neutral_proportion = time_dict[key][2] - time_dict[previous_key][2]
        amount_difference = time_dict[key][3] - time_dict[previous_key][3]
        y_pos_1.append(round(positive_proportion / float(amount_difference), 2))
        y_neu_1.append(round(neutral_proportion / float(amount_difference), 2))
        y_neg_1.append(round(negative_proportion / float(amount_difference), 2))
        y_2.append(appropriate_tweets_number)
        # Запоминаем предыдущий ключ, чтобы просчитать разницу для распределения
        previous_key = key
        appropriate_tweets = []

# Строим графики
# Создаем области
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(27, 6))
# Значения по Ox
x = [key.split(' - ')[1] for key in time_dict.keys()]
x_pos = np.arange(len(x))
# Первый график
ax1.plot(x_pos, y_pos_1, 'g.-', linewidth=1, markersize=7, label='N_pos')
ax1.plot(x_pos, y_neu_1, 'b.--', linewidth=1, markersize=7, label='N_neu')
ax1.plot(x_pos, y_neg_1, 'r.-.', linewidth=1, markersize=7, label='N_neg')
ax1.set_xticks([])
ax1.grid(True)
ax1.legend()
ax1.set_ylabel('Fraction')
# Второй график
# Создаем столбцы
ax2.stem(x_pos, y_2, basefmt=" ")
# Размечаем координатную ось
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x)
ax2.grid(True)
ax2.set_ylabel('Number of tweets')
fig.suptitle('Distribution of tweets classes in time', fontsize=16, y=1.0)
fig.show()
fig.savefig('Time distribution.png')

# Выходные параметры
tic = time()
print('Время выполнения программы: ', tic-toc)
print('Затраченная память: ', memory_usage())