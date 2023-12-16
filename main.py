import  csv
import pandas as pd
import duckdb
import seaborn as sns
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import plotly as plt
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


D_clients = duckdb.read_csv('datasets/D_clients.csv')
D_job = duckdb.read_csv('datasets/D_job.csv')
D_loan = duckdb.read_csv('datasets/D_loan.csv')
D_pens = duckdb.read_csv('datasets/D_pens.csv')
D_work = duckdb.read_csv('datasets/D_work.csv')
D_close_loan = duckdb.read_csv('datasets/D_close_loan.csv')
D_last_credit = duckdb.read_csv('datasets/D_last_credit.csv')
D_salary = duckdb.read_csv('datasets/D_salary.csv')
D_target = duckdb.read_csv('datasets/D_target.csv')

q = '''
SELECT DISTINCT *
FROM D_clients
JOIN D_target ON D_clients.ID = D_target.ID_CLIENT
JOIN D_salary ON D_clients.ID = D_salary.ID_CLIENT
JOIN D_loan ON D_clients.ID = D_loan.ID_CLIENT
JOIN D_close_loan ON D_close_loan.ID_LOAN = D_loan.ID_LOAN
JOIN D_job ON D_clients.ID = D_job.ID_CLIENT
JOIN 
(SELECT t1.ID, 
CAST(t1.CLOSED_FL AS INT64) AS CLOSED_FL, 
COUNT(t1.ID_LOAN) as "LOAN_NUM_TOTAL",
CAST(SUM(t1.CLOSED_FL) AS INT64) as " LOAN_NUM_CLOSED"
FROM 
(SELECT *
FROM D_clients
JOIN D_target ON D_clients.ID = D_target.ID_CLIENT
JOIN D_salary ON D_clients.ID = D_salary.ID_CLIENT
JOIN D_loan ON D_clients.ID = D_loan.ID_CLIENT
JOIN D_close_loan ON D_close_loan.ID_LOAN = D_loan.ID_LOAN
JOIN D_job ON D_clients.ID = D_job.ID_CLIENT) as t1
GROUP BY ID, CLOSED_FL) as t2
ON D_clients.ID = t2.ID
'''
db = duckdb.query(q).to_df()

st.title('Анализ клиентов банка')
st.write('Это приложение позволяет провести разведочный анализ данных, софрмировать портрет клиентов, '
         'выявить зависимости между признаками клиента и его склонности к положительному или отрицательному отклику на предложение банка.')

image = Image.open('img_1.png')
st.image(image, use_column_width=True)

tab1, tab2 = st.tabs(['Таблица', 'Разведочный анализ'])

with tab1:

    target_1 = st.checkbox('Откликнулись на предложение банка')
    target_0 = st.checkbox('Отказались от предложения банка')
    if target_1 & target_0:
        filtred_df = db
    if target_1:
        filtred_df = db[db['TARGET'] == 1]
    elif target_0:
        filtred_df = db[db['TARGET'] == 0]
    else:
        filtred_df = db
    st.dataframe(filtred_df)

with tab2:
    st.write('Разведочный анализ')

    def get_education_data(target):
        q = '''
        SELECT EDUCATION as "Образование",
        COUNT(EDUCATION) as "Кол-во" 
        FROM db
        WHERE TARGET in ({})
        GROUP BY "Образование"
        ORDER BY "Кол-во" DESC 
        '''.format(target )
        education_data = duckdb.query(q).to_df()
        return education_data


    def get_age_data(target):
        q1 = '''
        SELECT AGE, COUNT(AGE) as COUNT,
        FROM db
        WHERE TARGET in ({})
        GROUP BY  AGE
        ORDER BY AGE ASC
        '''.format(target)
        age_tg = duckdb.query(q1).to_df()
        return age_tg

    def get_gender_data(target):
        q2 = '''
        SELECT CASE
            WHEN GENDER = 1 THEN 'Мужчины'
            WHEN  GENDER = 0 THEN 'Женщины'
        END as "Пол",
        COUNT(GENDER) as "Кол-во"
        FROM db
        WHERE TARGET in ({})
        GROUP BY  GENDER
        ORDER BY GENDER ASC
        '''.format(target)
        gender_tg = duckdb.query(q2).to_df()
        return gender_tg

    def get_personal_income_target():
        q = '''
                SELECT TARGET AS "Отклик на предложение банка", PERSONAL_INCOME AS "Доход"
                FROM db
                '''
        personal_income = duckdb.query(q).to_df()
        return personal_income

    def plot_education_pie_chart(education_data):
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=education_data['Образование'], values=education_data['Кол-во'], marker_colors=['#f72585', '#7209b7', '#3a0ca3', '#4361ee', '#4cc9f0', '#560badff']))
        fig.update_layout(title='Распределение людей по уровню образования')
        return fig

    def plot_age_pie_chart(age_tg):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=age_tg['AGE'], y=age_tg['COUNT'], text=age_tg['COUNT'], textposition='auto', textangle=0,  marker_color='#4361ee', name='Количество людей'))
        fig.add_trace(go.Scatter(x=age_tg['AGE'], y=age_tg['COUNT'].groupby(age_tg['AGE']).max(), line_shape='spline', mode='lines', name='Тренд', marker_color='#f72585'))
        fig.update_layout( title='Распределение людей по возрасту', xaxis_title='Возраст', yaxis_title='Количество людей')
        return fig

    def plot_gender_pie_chart(gender_tg):
        fig = go.Figure()
        fig.add_trace(go.Pie(labels = gender_tg['Пол'], values = gender_tg['Кол-во'], marker_colors=['#f72585', '#4361ee']))
        fig.update_layout( title='Распределение людей по полу')
        return fig


    def plot_personal_income_chart(personal_income):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=db[db['TARGET'] == 1].groupby('PERSONAL_INCOME').size().reset_index(name='COUNT')['PERSONAL_INCOME'], y = db[db['TARGET'] == 1].groupby('PERSONAL_INCOME').size().reset_index(name='COUNT')['COUNT'],
                                 mode='lines', fill='tozeroy',
                                 marker_color='#4361ee',
                                 name='Откликнулись на предложение банка',
                                 opacity=0.5, textposition='bottom center' ))
        fig.add_trace(go.Scatter(x=db[db['TARGET'] == 0].groupby('PERSONAL_INCOME').size().reset_index(name='COUNT')['PERSONAL_INCOME'], y = db[db['TARGET'] == 0].groupby('PERSONAL_INCOME').size().reset_index(name='COUNT')['COUNT'],
                                 mode='lines', fill='tozeroy',
                                 marker_color='#f72585',
                                 name='Отказались от предложения банка',
                                 opacity=0.5, textposition='bottom center' ))

        fig.update_layout(xaxis=dict(range=[0, 100000]))

        fig.update_layout(title='Распределение зарплаты',
                          xaxis_title='Зарплата',
                          yaxis_title='Количество клиентов')

        return fig

    def child_chart(target):
        # fig, ax = plt.subplots(1, 2, figsize=(14, 9))
        plt.figure(figsize=(14, 9))
        sns.histplot(db[db['TARGET'] == target]['CHILD_TOTAL'], kde=True,
                     stat="density", color='#4361ee')
        mu, std = db[db['TARGET'] == target]['CHILD_TOTAL'].mean(), \
        db[db['TARGET'] == target]['CHILD_TOTAL'].std()
        xmin, xmax = plt.xlim()
        sz = db[db['TARGET'] == target]['CHILD_TOTAL'].size
        x = np.linspace(xmin, xmax, sz)
        p = sp.stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        if target == 1:
            plt.title(
            f'Распределение количества детей у клиентов, \n которые воспользовались предложением Банка')
        else:
            plt.title(
                f'Распределение количества детей у клиентов, \n которые отказались от предложения Банка')
        plt.xlabel('Количество детей')
        plt.ylabel('Плотность')
        return plt

    st.set_option('deprecation.showPyplotGlobalUse', False)

    def corr_feature(numerical_signs):
        corr = X.corr()
        plt.figure(figsize=(15, 11))
        sns.heatmap(corr,fmt='.3f', annot=True, cmap='PuRd', linewidths=.5)
        plt.title('Матрица корреляций', fontdict={'fontsize': 16, 'fontweight': 'bold'})
        return plt


    numerical_signs = db.drop(['SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'EDUCATION',
                 'MARITAL_STATUS', 'REG_ADDRESS_PROVINCE',
                 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE',
                 'FAMILY_INCOME', 'GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR',
                 'ID_CLIENT', 'ID_CLIENT_2',
                 'ID_CLIENT_3', 'ID_LOAN_2', 'ID_CLIENT_4', 'ID_2',
                 'CLOSED_FL_2', 'AGREEMENT_RK', 'ID_LOAN'], axis=1)



    target_tab_1 = st.checkbox('Откликнулись на предложение')
    target_tab_0 = st.checkbox('Отказались от предложения')

    if target_tab_1:
        target = 1
    elif target_tab_0:
        target = 0
    else:
        target = '0, 1'

    age_data = get_age_data(target)
    fig_age = plot_age_pie_chart(age_data)

    education_data = get_education_data(target)
    fig_education = plot_education_pie_chart(education_data)

    gender_data = get_gender_data(target)
    fig_gender = plot_gender_pie_chart(gender_data)

    personal_income = get_personal_income_target()
    fig_personal_income = plot_personal_income_chart(personal_income)


    st.plotly_chart(fig_education)
    st.plotly_chart(fig_age)
    st.plotly_chart(fig_gender)
    st.plotly_chart(fig_personal_income)
    st.pyplot(child_chart(1))
    st.pyplot(child_chart(0))
    st.pyplot(corr_feature(numerical_signs))
    st.markdown(
        "<p style='font-size: 16px; font-weight: bold;'>Вычисление числовых характеристик распределения числовых столбцов</p>",
        unsafe_allow_html=True
    )
    st.dataframe(db.describe())