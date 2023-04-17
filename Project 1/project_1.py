#!/usr/bin/env python
# coding: utf-8

# # Прогнозирование оттока клиентов для оператора связи
# 
# Построим модель машинного обучения для бинарной классификации для прогнозирования оттока клиентов.
# 
# Оператор предоставляет два типа услуг:
# - Стационарную телефонную связь.Возможно подключение телефонного аппарата к нескольким линиям одновременно.
# - Интернет. Подключение может быть двух типов: 
#     - через телефонную линию (DSL, от англ. digital subscriber line, «цифровая абонентская линия»)
#     - оптоволоконный кабель(Fiber optic).
#     
#     
# Также доступны такие услуги:
# - Интернет-безопасность: антивирус (DeviceProtection)
# - Блокировка небезопасных сайтов (OnlineSecurity)
# - Выделенная линия технической поддержки (TechSupport)
# - Облачное хранилище файлов для резервного копирования данных (OnlineBackup)
# - Стриминговое телевидение (StreamingTV) и каталог фильмов (StreamingMovies)
# 
# За услуги клиенты могут платить каждый месяц или заключить договор на 1–2 года. Доступны различные способы расчёта и возможность получения электронного чека.

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from catboost import CatBoostClassifier, Pool

RS = 42

import warnings
warnings.filterwarnings("ignore")


# In[44]:


pd.set_option('display.max_columns', None)   # видны все столбцы
pd.set_option('display.max_rows', None)      # и строки


# ## Знакомство с данными

# In[45]:


# Загрузка данных

contract = pd.read_csv('/datasets/final_provider/contract.csv')
personal = pd.read_csv('/datasets/final_provider/personal.csv')
internet = pd.read_csv('/datasets/final_provider/internet.csv')
phone = pd.read_csv('/datasets/final_provider/phone.csv')

data = [contract, personal, internet, phone]
data_name = ['contract', 'personal', 'internet', 'phone']


# Описание файлов:
# 
# Данные состоят из файлов, полученных из разных источников. Во всех файлах столбец customerID содержит код клиента.
# 
# - contract_new.csv — информация о договоре
#     - BeginDate - дата, с которой клиент начал пользоваться услугами компании
#     - EndDate - дата, с которой клиент перестал пользоваться услугами компании (No - клиент на данный момент пользуется услугами компании)
#     - Type - периодичность оплаты
#     - PaperlessBilling - выставление бумажного счета
#     - PaymentMethod - способ оплаты
#     - MonthlyCharges - сумма ежемесячного платежа
#     - TotalCharges - итоговая сумма, которую заплатил клиент, пользуясь услугами компании  
#     
#     
# - personal_new.csv — персональные данные клиента
#     - gender - пол клиента
#     - SeniorCitizen - наличие пенсионного статуса
#     - Partner - наличие партнера
#     - Dependents - наличие иждевенцев
#     
#     
# - internet_new.csv — информация об интернет-услугах
#     - InternetService - тип подключения интернета
#     - OnlineSecurity - наличие услуги по блокировкае небезопасных сайтов
#     - OnlineBackup - наличие услуги облачного хранилища файлов для резервного копирования данных
#     - DeviceProtection - наличие услуги антивируса
#     - TechSupport - наличие выделенной линии технической поддержки
#     - StreamingTV - наличие услуги стримингового телефидения
#     - StreamingMovies - наличие услуги каталога фильмов
#     
#     
# - phone_new.csv — информация об услугах телефонии
#     - MultipleLines - наличие параллельных линий (телефонная связь)
# 
# 
# Информация о договорах актуальна на 1 февраля 2020.

# In[46]:


for i in range(len(data)):
    print('\033[1m' + data_name[i] + '\033[0m')
    display(data[i].head())
    data[i].info()
    print()


# In[47]:


# Изучение уникальных значений в каждом из столбцов

for i in range(len(data)):
    for index in data[i].columns:
        if index != 'customerID':
            print('\033[1m' + index + '\033[0m')
            print(data[i][str(index)].unique())
            print()


# In[48]:


# Изучение явных пропущенных значений

for i in range(len(data)):
    print('\033[1m' + data_name[i] + '\033[0m')
    print(data[i].isna().sum())
    print()


# In[49]:


# Изучение дубликатов

for i in range(len(data)):
    print('\033[1m' + data_name[i] + '\033[0m')
    display(data[i].groupby('customerID').count().iloc[:,0].sort_values(ascending=False).head())


# Объединим все таблицы в одну (data_total) для построения модели машинного обучения. Объединять будем через тип объединения 'outer', чтобы не упустить ни одного пользователя.
# 
# В дальнейшем будем работать с ней.

# In[50]:


data_total = contract.merge(personal, on='customerID', how='outer')
data_total = data_total.merge(internet, on='customerID', how='outer')
data_total = data_total.merge(phone, on='customerID', how='outer')
    
display(data_total.head())
data_total.info()


# Посмотрим на количество пропусков в итоговой таблице.

# In[51]:


data_total.isna().sum()


# ## Подготовка и исследовательский анализ данных

# Изменим тип данных в столбце BeginDate с object на datetime:

# In[71]:


data_total['BeginDate'] = pd.to_datetime(data_total['BeginDate'], format='%Y-%m-%d')

sns.set_theme(style="ticks")

plt.figure(figsize=(8,5))
ax = sns.histplot(data_total['BeginDate'].dt.year)
ax.set_title('Распределение количества пользователей по году начала пользования услугами компании', fontsize=14)
ax.set(xlabel='Год',
       ylabel='Количество новых пользователей')
plt.grid()


# Данные компании начинаются с 2013 года, далее каждый год количество клиентов понемногу растет. Большой скачок наблюдается в 2019 году.

# In[80]:


plt.figure(figsize=(12,5))
ax = sns.histplot(pd.to_datetime(data_total.loc[data_total['EndDate'] != 'No', 'EndDate'], format='%Y-%m-%d'))
ax.set_title('Распределение количества пользователей по дате окончания пользования услугами компании', fontsize=15)
ax.set(xlabel='Дата(год-месяц-число)',
       ylabel='Количество ушедших пользователей')
plt.grid()

print('Количество клиентов компании = ', data_total.loc[data_total['EndDate'] == 'No', 'EndDate'].count())


# C октября 2019 года каждый месяц начали уходить пользователи (400-500 человек). На дату выгрузки клиентами компании остаются 5174 человека. Будем делать предсказания для этой группы людей на дату выгрузки данных - 1 февраля 2020.

# In[81]:


plt.figure(figsize=(20,5))
ax = sns.histplot(data_total['MonthlyCharges'], bins=100)
ax.set(xlim=(15,120))
ax.set_title('Распределение количества пользователей по размеру ежемесячного платежа', fontsize=17)
ax.set(xlabel='Сумма платежа',
       ylabel='Количество пользователей')
plt.grid()

print('Минимальное значение ежемесячного платежа (за все время) =', data_total['MonthlyCharges'].min())
print('Максимальное значение ежемесячного платежа (за все время) =', data_total['MonthlyCharges'].max())


# Больше всего клиентов ежемесячно приносят компании по 20-25 денежных единиц.  

# In[82]:


data_total.query('TotalCharges == " " ')


# В столбце есть несколько значений, где нет итоговой суммы. Это связано с тем, что пользователи стали клиентами компании в день выгрузки данных. Заполним столбец 'TotalCharges' значением ежемесячного платежа.

# In[83]:


data_total.loc[data_total['TotalCharges'] == " ", 'TotalCharges'] = data_total['MonthlyCharges']


# Изменим тип данных в столбце 'TotalCharges' с object на float и построим гистрограмму.

# In[88]:


data_total['TotalCharges'] = data_total['TotalCharges'].astype('float')

plt.figure(figsize=(20,5))
ax = sns.histplot(data_total['TotalCharges'], bins=100)
ax.set(xlim=(0,8500))
ax.set_title('Распределение количества пользователей по размеру суммарного (за все время) платежа', fontsize=17)
ax.set(xlabel='Сумма платежа',
       ylabel='Количество пользователей')
plt.grid()
print('Минимальное значение трат (за все время) =', data_total['TotalCharges'].min())
print('Максимальное значение трат (за все время) =', data_total['TotalCharges'].max())


# Здесь также большое количество пользователей с небольшими годовыми платежами. Что логично - им соответствуют небольшие ежемесячные платежи.

# In[143]:


fig, axes = plt.subplots(2,3, figsize=(20,10))
fig.suptitle('Распределения количества пользователей', fontsize=20)

ax1 = sns.histplot(data_total['Type'], ax=axes[0,0])
ax1.set(xlabel='Тип платежа (ежемесячный, ежегодный, раз в два года)',
       ylabel='Количество пользователей')

ax2 = sns.histplot(data_total['PaperlessBilling'], ax=axes[0,1])
ax2.set(xlabel='Выставление бумажного счета',
        ylabel='Количество пользователей')

ax3 = sns.histplot(data_total['gender'], ax=axes[0,2])
ax3.set(xlabel='Пол',
        ylabel='Количество пользователей')

ax4 = sns.histplot(data_total['SeniorCitizen'], ax=axes[1,0])
ax4.set(xlabel='Наличие пенсионного статуса',
        ylabel='Количество пользователей')

ax5 = sns.histplot(data_total['Partner'], ax=axes[1,1])
ax5.set(xlabel='Наличие партнера',
        ylabel='Количество пользователей')

ax6 = sns.histplot(data_total['Dependents'], ax=axes[1,2])
ax6.set(xlabel='Наличие иждевенцев',
        ylabel='Количество пользователей');

plt.figure(figsize=(12,4))
ax7 = sns.histplot(data_total['PaymentMethod'])
ax7.set(xlabel='Способ оплаты',
        ylabel='Количество пользователей');


# Иходя из графиков, можно заключить, что около половины людей платят за услуги каждый месяц, четверть ежегодно и еще четверть каждые два года. 
# 
# Мужчин и женщин примерно одинаковое кол-во. Также примерно поровну тех, кто имеет и не имеею партнера. 
# 
# Большинство клиентов (>80%) не имеют пенсионного статуса, кроме того у ~70% клиентов нет иждевенцев.

# 
# 
# Далее рассмотрим столбец MultipleLines.
# Здесь есть пропуски. Выведем все строки, где этот столбец не заполнен.

# In[17]:


data_total.loc[data_total['MultipleLines'].isnull() == True].head()


# Тип интернет соединения в каждой строчке, где не заполнено наличие возможности ведения параллельных линий, - телефонная линия (DSL). Рассмотрим, чуть подробнее типы подключения.

# In[18]:


print('Тип соединения DSL и возможность проведения параллельных линий (кол-во пользователей) -',
      data_total.query('InternetService == "DSL" & MultipleLines == "Yes"')['MultipleLines'].count())
print('Тип соединения DSL и отсутствие возможности проведения параллельных линий (кол-во пользователей) -',
      data_total.query('InternetService == "DSL" & MultipleLines == "No"')['MultipleLines'].count())

print('Тип соединения Fiber optic и возможность проведения параллельных линий (кол-во пользователей) -',
      data_total.query('InternetService == "Fiber optic" & MultipleLines == "Yes"')['MultipleLines'].count())
print('Тип соединения Fiber optic и отсутствие возможности проведения параллельных линий (кол-во пользователей) -',
      data_total.query('InternetService == "Fiber optic" & MultipleLines == "No"')['MultipleLines'].count())


# Как таковой зависимсти я здесь не вижу, скорее всего для части пользователей просто забыли заполнить столбец MultipleLines. Заполним пропущенные значения типом 'unknown', так как удалить почти 10% пользователей мы не можем, а четкая зависимось обнаружена не была и рандомное заполнение не правомерно.

# In[145]:


data_total.loc[data_total['MultipleLines'].isnull() == True, 'MultipleLines'] = 'unknown'

plt.figure(figsize=(5,4))
ax = sns.histplot(data_total['MultipleLines'])
ax.set_title('Распределение количества пользователей по наличию параллельных линий', fontsize=14)
ax.set(xlabel='Наличие параллельных линий (телефонная связь)',
       ylabel='Количество пользователей')
plt.grid()


# Далее рассмотрим столбец InternetService и выведем строчки, где значения не заполнены.

# In[147]:


data_total.loc[data_total['InternetService'].isnull() == True].head()


# Если изучить все объекты из выборки выше, то видно, что если не заполнен столбец, где указано наличие интернет-соединения, то не заполнены столбцы блокировка небезопасных сайтов, облачное хранилище, антивирус, стриминговое телевидение и каталог фильмов.
# 
# Мне кажется, это значит, что интернет не подключен, соответственно не подключены и все эти сервисы. Заполним все недостающие значения в этих столбцах на 'No'.

# In[148]:


data_total = data_total.fillna('No')


# In[154]:


internet_col = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']

fig, axes = plt.subplots(2,3, figsize=(20,10))
fig.suptitle('Распределения количества пользователей', fontsize=20)

ax1 = sns.histplot(data_total['InternetService'], ax=axes[0,0])
ax1.set(xlabel='Тип подключения к интернету',
       ylabel='Количество пользователей')

ax2 = sns.histplot(data_total['OnlineSecurity'], ax=axes[0,1])
ax2.set(xlabel='Блокировка небезопасных сайтов',
       ylabel='Количество пользователей')

ax3 = sns.histplot(data_total['OnlineBackup'], ax=axes[0,2])
ax3.set(xlabel='Облачное хранилище файлов',
       ylabel='Количество пользователей')

ax4 = sns.histplot(data_total['DeviceProtection'], ax=axes[1,0])
ax4.set(xlabel='Антивирус',
       ylabel='Количество пользователей')

ax5 = sns.histplot(data_total['TechSupport'], ax=axes[1,1])
ax5.set(xlabel='Выделенная линия техн. подержки',
       ylabel='Количество пользователей')

ax6 = sns.histplot(data_total['StreamingTV'], ax=axes[1,2])
ax6.set(xlabel='Стриминговое телевидение',
       ylabel='Количество пользователей')

plt.figure(figsize=(4.5,4))
ax7 = sns.histplot(data_total['StreamingMovies'])
ax7.set(xlabel='Каталог фильмов',
       ylabel='Количество пользователей');


# In[23]:


print('Кол-во пользователей с типом интернет-соединения DSL =',
      data_total.query('InternetService == "DSL"')['InternetService'].count())
print('Кол-во пользователей с типом интернет-соединения Fiber optic =',
      data_total.query('InternetService == "Fiber optic"')['InternetService'].count())


# Около 44 % пользоватлей, указавших тип интернет-соединения выбирают DSL, остальные ~56% - Fiber optic.
# 
# Услугами, связанными с интернетом пользуется от 30 до 40% пользователей (в зависимости от услуги).

# ## Промежуточные выводы

# Данные были загружены, объединены в одну таблицу и проанализированы.
# 
# 1. В столбцах BeginDate и EndDate был изменен тип данных с object на datetime.   
# 2. В столбце TotalCharges были заполнены пропущенные значения. Они были связаны с клиентами, присоединившимися к компании в день выгрузки данных. Также был заменен тип данных с object на float.
# 3. В столбце MultipleLines не было найдено закономерностей для пропущенных значений и пропуски были обозначены типом 'unknown' (около 10%). 
# 4. В столбцах, связанных с интернет-соединением все пропуски были заменены на 'No', так как вероятнее всего у пользователей, не пользующихся интернетом просто не заполнили данную графу.
# 
# 
# Данные компании представлены с 2013 года. Кажый год количество клиентов понемногу росло. Большой скачок наблюдался в 2019 году, с октября 2019 года каждый месяц начали уходить пользователи (400-500 человек). 
# 
# Будем делать предсказание - уйдет клиент из компании или нет на датц выгрузки данных 1 февраля 2020 года.
# 
# Для этого планируется разделить данные на тренировочный (75%) и тестовый датасеты (25%). Деление будет происходить со стратификацией. Затем данные будут закодированы (для классических алгоритмов). Будут строиться модели логистической регрессии и случайного леса. Также буду использовать CatBoost, предваритеьно указав какие признаки категориальные, а какие количественные.
# Качество бинарной классификации будем смотреть на auc-roc, после чего будет выбираться лучшая модель и тестироваться на тестовой выборке. Итоговый показатель по заданию заказчика должен быть выше 0.85.

# ## Подготовка к построению модели

# Cоздадим столбец target на основе столбца EndDate: 1 - пользователь ушел (есть дата в столбце EndDate) и пользователь пользуется услугами компании на данный момент (No в столбце EndDate).

# In[24]:


data_total.loc[data_total['EndDate'] != 'No', 'target'] = 1
data_total.loc[data_total['EndDate'] == 'No', 'target'] = 0


# Создадим дополнительный признак.
# 
# Для этого сначала вместо даты ухода клиента поставим заглушку - 1 февраля 2020 (дату выгрузки данных). Затем получим кол-во дней, которое клиент пользовался услугами компании.

# In[25]:


data_total.loc[data_total['EndDate'] == 'No', 'EndDate'] = '2020-02-01 00:00:00'
data_total['EndDate'] = pd.to_datetime(data_total['EndDate'], format='%Y-%m-%d %H:%M:%S')

data_total['delta'] = data_total['EndDate'] - data_total['BeginDate']
data_total['delta'] = data_total['delta'].dt.days


# Удалим столбцы BeginDate и EndDate.

# In[26]:


data_total = data_total.drop(['BeginDate', 'EndDate'], axis=1)


# Поделим данные на тренировочный и тестовый датасеты.

# In[27]:


features = data_total.drop(['target', 'customerID'], axis=1)
target = data_total['target']

features_train, features_test, target_train, target_test = train_test_split(features, target,
                                                                            test_size=0.25,
                                                                            random_state=RS,
                                                                            stratify=target)


# Проверим, насколько корректно данные разделились на выборки

# In[28]:


print('Процент ушедших из компании пользователей в тренировочной выборке = {:.0%}'.format(target_train.sum()/target_train.count()))
print('Процент ушедших из компании пользователей в тестовой выборке = {:.0%}'.format(target_test.sum()/target_test.count()))


# Скопируем датафреймы для кодирования категориальных признаков

# In[29]:


features_train_ohe = features_train.copy()
features_test_ohe = features_test.copy()


# Закодируем категориальные признаки (OneHotEncoder)

# In[30]:


columns_ohe = ['Type', 'PaperlessBilling', 'PaymentMethod', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
               'StreamingTV', 'StreamingMovies', 'MultipleLines']

encoder_ohe = OneHotEncoder()
encoder_ohe.fit(features_train_ohe[columns_ohe])

features_ohe_train = pd.DataFrame(encoder_ohe.transform(features_train_ohe[columns_ohe]).toarray(),
                                  index=features_train_ohe.index)
features_ohe_test = pd.DataFrame(encoder_ohe.transform(features_test_ohe[columns_ohe]).toarray(),
                                 index=features_test_ohe.index)

features_ohe_train.columns = encoder_ohe.get_feature_names(columns_ohe)
features_ohe_test.columns = encoder_ohe.get_feature_names(columns_ohe)

features_train_ohe = pd.concat([features_train_ohe, features_ohe_train], axis=1)
features_test_ohe = pd.concat([features_test_ohe, features_ohe_test], axis=1)

features_train_ohe = features_train_ohe.drop(columns_ohe, axis=1)
features_test_ohe = features_test_ohe.drop(columns_ohe, axis=1)


# Посмотрим на данные после кодирования

# In[31]:


features_train_ohe.head()


# ## Построение моделей

# Будем работать с тремя моделями - LogisticRegression, RandomForestClassifier и CatBoostClassifier, подбирать гиперпараметры с помощью GridSearch.

# ### Логистическая регрессия

# In[32]:


get_ipython().run_cell_magic('time', '', "\nmodel_LR = LogisticRegression(random_state=RS, max_iter=200)\nparameters_LR = {'C':[1,10], 'solver':['liblinear']} \ngrid_LR = GridSearchCV(model_LR, parameters_LR, cv=5, scoring='roc_auc')\ngrid_LR.fit(features_train_ohe, target_train)\n\nprint('Лучшие параметры модели = ', grid_LR.best_params_)\nprint('ROC-AUC модели =', grid_LR.best_score_.round(2))")


# ### RandomForestClassifier

# In[33]:


get_ipython().run_cell_magic('time', '', "\nmodel_forest = RandomForestClassifier(random_state=RS) \nparameters_forest = {'n_estimators':[1,1000,50], 'max_depth':[1,12]}\ngrid_forest = GridSearchCV(model_forest, parameters_forest, cv=3, scoring='roc_auc')\ngrid_forest.fit(features_train_ohe, target_train)\n\nprint('Лучшие параметры модели = ', grid_forest.best_params_)\nprint('ROC-AUC модели =', grid_forest.best_score_.round(2)) ")


# ### CatBoostClassifier

# In[188]:


get_ipython().run_cell_magic('time', '', "\nmodel_CBC = CatBoostClassifier(random_state=RS, verbose=100, eval_metric='AUC:hints=skip_train~false',\n                               cat_features=['Type', 'PaperlessBilling', 'PaymentMethod', 'gender', 'SeniorCitizen',\n                                             'Partner', 'Dependents', 'InternetService', 'OnlineSecurity',\n                                             'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',\n                                             'StreamingMovies', 'MultipleLines'])\nparameters_CBC = {'depth':[1,10], 'n_estimators':[0,2000], 'learning_rate':[1]}\ngrid_CBC = GridSearchCV(model_CBC, parameters_CBC, cv=3, scoring='roc_auc')\ngrid_CBC.fit(features_train, target_train)\n\nprint('Лучшие параметры модели = ', grid_CBC.best_params_)\nprint('ROC-AUC модели =', grid_CBC.best_score_.round(2))")


# Сравним модели

# In[155]:


column=['ROC-AUC модели']

result = pd.DataFrame(index=column,
                      columns=['LogisticRegression','RandomForestClassifier',
                                'CatBoostClassifier'])
result['LogisticRegression'] = grid_LR.best_score_.round(2)
result['RandomForestClassifier'] = grid_forest.best_score_.round(2)
result['CatBoostClassifier'] = grid_CBC.best_score_.round(2) 

result


# Из таблицы можно увидеть, что лучшая модель - CatBoostClassifier. Будем тестировать ее.

# ## Тестирование модели

# In[189]:


model_CBC = CatBoostClassifier(random_state=RS, verbose=100, depth=1, n_estimators=2000, learning_rate=1, 
                               eval_metric='AUC:hints=skip_train~false', custom_loss=['AUC'],
                               cat_features=['Type', 'PaperlessBilling', 'PaymentMethod', 'gender', 'SeniorCitizen',
                                             'Partner', 'Dependents', 'InternetService', 'OnlineSecurity',
                                             'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                             'StreamingMovies', 'MultipleLines'])
model_CBC.fit(features_train, target_train, plot=True) 
predictions_test_cat = model_CBC.predict(features_test)

probabilities_test_cat = model_CBC.predict_proba(features_test)
probabilities_one_test_cat = probabilities_test_cat[:, 1]
auc_roc_cat = roc_auc_score(target_test, probabilities_one_test_cat)

print('auc-roc на тестовой выборке =', auc_roc_cat.round(2))


# Построим ROC кривую и матрицу ошибок.

# In[190]:


fpr, tpr, thresholds = roc_curve(target_test, probabilities_one_test_cat)
plt.figure(figsize=(8,6))
ax = sns.lineplot(fpr,tpr)
ax = sns.lineplot([0, 1], [0, 1], linestyle='--') # ROC-кривая случайной модели
ax.set(xlim=(-0.01, 1.0))
ax.set(ylim=(0.0, 1.01))
ax.set_title('ROC-кривая', fontsize=17)
ax.set(xlabel='Доля ложно положительных ответов',
       ylabel='Доля истинно положительных ответов')
plt.grid();


# На графике изображена ROC-кривая нашей модели и кривая модели, которая отвечает случайно (пунктирная линия). Она сильно выше случайной модели, что говорит о неплохом качестве предсказаний нашей модели CatBoost.

# In[191]:


plt.figure(figsize=(6, 5))
ax = sns.heatmap(confusion_matrix(target_test, predictions_test_cat), 
                 annot=True,
                 fmt='d')
ax.set_title('Матрица ошибок', fontsize=16)
ax.set(xlabel='Предсказания',
       ylabel='Правильные ответы');


# Исходя из матицы ошибок, видно, что 1476 ответов мы предсказали верно. 198 человек (FN) ушли из компании, хотя мы предсказали, что они останутся, а 78 человек (FP) наоборот - остались, хотя мы предсказали, что они покинут компанию. 
# 
# Получается, что в зависимости от того, что будет важнее для компании - не потерять клиента или не раздать лишних промокодов (для тех, кто собирается уйти), мы сможем подобрать наилучший порог. 

# ## Важность признаков моделей

# Проведем анализ важности признаков лучшей модели

# In[193]:


train_dataset = Pool(features_train,
                     target_train,
                     cat_features=['Type', 'PaperlessBilling', 'PaymentMethod', 'gender', 'SeniorCitizen',
                                   'Partner', 'Dependents', 'InternetService', 'OnlineSecurity',
                                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                   'StreamingMovies', 'MultipleLines'])
feature_importance = model_CBC.get_feature_importance(train_dataset, prettified = True)
display(feature_importance)


feature_importance.plot.barh(x='Feature Id', y='Importances', figsize=(20,10));
plt.xlim([-0.5, 60])
plt.xlabel('Важность признака', fontsize=16)
plt.ylabel('Признаки', fontsize=16)
plt.title('Важность признаков модели', fontsize=20)
plt.grid();


# Самым важным признаком является delta - время, которое клиент пользовался услугами компании. Затем идут InternetService и Type, TotalCharges и MonthlyCharges. Отсальные признаки сильно менее важны.

# ## Выводы

# Данные были загружены, объединены в одну таблицу и проанализированы. В нескольких стролбцах изменены типы данных, пропущенные значения заполнены.
# 
# На этапе подготовки к построению модели на основе столбца EndDate был создан целевой столбец.
# Также был создан дополнительный признак - столбец delta (на основе столбцов BeginDate и EndDate). Это кол-во дней, которое пользователь пользуется услугами компании. После были удалены столбцы BeginDate и EndDate. 
# 
# Далее данные были поделены на тренировочный и тестовый датасеты в отношении 75/25 со стратификацией 
# и закодированы, посредством OneHotEncoder. 
# 
# Были обучены модели LogisticRegression, RandomForestClassifier и CatBoostClassifier. Гиперпараметры подбирались с помощью гридсерча.
# 
# После модели сравнивались между собой, лучшей была выбрана модель CatBoostClassifier с гиперпараметрами:
# - depth=1
# - n_estimators=2000
# - learning_rate=1.
# 
# Затем модель прогонялась через тестовые данные, на которых метрика roc-auc достигла значения 0.9. Были построены roc-кривые и матрица ошибок. Также был проведен анализ важности признаков - наиболее важными признаками являются - время, которое клиент пользовался услугами компании (delta), наличие подключения к интернету (InternetService) и его тип (Type), а также итоговая и помесячная сумма платежа (TotalCharges и MonthlyCharges).
