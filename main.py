import  csv
import pandas as pd
import duckdb
import seaborn as sns
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import plotly as plt
from PIL import Image

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
JOIN D_work ON D_clients.SOCSTATUS_WORK_FL = D_work.ID
JOIN D_pens ON D_clients.SOCSTATUS_PENS_FL = D_pens.ID
--JOIN D_work ON D_clients.ID = D_work.ID
--SELECT *
--FROM D_pens
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
    if target_1:  # Если пользователь выбрал только мужчин
        filtred_df = db[db['TARGET'] == 1]
    elif target_0:  # Если пользователь выбрал только женщин
        filtred_df = db[db['TARGET'] == 0]
    else:
        filtred_df = db
    st.dataframe(filtred_df)

    # if age_filter[0] == 0 and age_filter[1] == 100:
        # if gender_1 and gender_0:  # Если пользователь выбрал оба значения пола
        #     filtered_df = db
        # elif gender_1:  # Если пользователь выбрал только мужчин
        #     filtered_df = db[db['GENDER'] == 1]
        # elif gender_0:  # Если пользователь выбрал только женщин
        #     filtered_df = db[db['GENDER'] == 0]
        # else:  # Если пользователь не выбрал значения пола
        # filtered_df = db

    # else:  # Если пользователь выбрал фильтр по возрасту
    #     filtered_df = db[(db['AGE'] >= age_filter[0]) & (db['AGE'] <= age_filter[1])]
    #     st.dataframe(filtered_df)
    #     if gender_1:  # Если пользователь выбрал только мужчин
    #         filtered_df = filtered_df[filtered_df['GENDER'] == 1]
    #         st.dataframe(filtered_df)
    #     elif gender_0:  # Если пользователь выбрал только женщин
    #         filtered_df = filtered_df[filtered_df['GENDER'] == 0]
    #         st.dataframe(filtered_df)
    #     else:
    #         filtered_df = db[(db['AGE'] >= age_filter[0]) & (db['AGE'] <= age_filter[1])]
    #         st.dataframe(filtered_df)
    # filtered_df = db[(db['AGE'] >= age_filter[0]) & (db['AGE'] <= age_filter[1])]
    # if gender_1:
    #     filtred_df = filtred_df[(filtred_df['GENDER'] == 1)]
    #     st.dataframe(filtered_gender)
    # elif gender_0:
    #     filtred_gender = filtred_df[(filtred_df['GENDER'] == 0)]
    #     st.dataframe(filtered_gender)
    # else:
    #     st.dataframe(filtered_df)
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
                SELECT TARGET AS "Отклик на предложение банка", PERSONAL_INCOME AS "Доход", count(PERSONAL_INCOME) as "Кол-во"
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
        fig.add_trace(go.Scatter(x=personal_income[
            personal_income["Отклик на предложение банка"] == 1]["Доход"],
                                 mode='lines', fill='tozeroy',
                                 marker_color='#4361ee',
                                 name='Откликнулись на предложение банка',
                                 opacity=0.1))
        fig.add_trace(go.Scatter(x=personal_income[
            personal_income["Отклик на предложение банка"] == 0]["Доход"],
                                 mode='lines', fill='tozeroy',
                                 marker_color='#f72585',
                                 name='Отказались от предложения банка',
                                 opacity=0.8))
        fig.update_layout(xaxis=dict(range=[0, 100000]))

        fig.update_layout(title='Распределение зарплаты',
                          xaxis_title='Зарплата',
                          yaxis_title='Количество клиентов')

        # fig.add_trace(
        #     go.Histogram(labels={'Доход': 'Доход', 'count': 'Количество клиентов'}, values=personal_income[personal_income["Отклик на предложение банка"] == 0]["Доход"], opacity=0.5, nbins=20
        #              # name='Отрицательный отклик'
        #                  ))

        # Добавление вертикальной линии для медианы в первом распределении
        # median_target_1 = \
        # personal_income[personal_income["Отклик на предложение банка"] == 1][
        #     "Доход"].median()
        # fig.add_vline(x=median_target_1, line_dash="dash", line_color="blue",
        #               name="Медиана для Положительного отклика")

        # Добавление вертикальной линии для медианы во втором распределении
        # median_target_0 = \
        # personal_income[personal_income["Отклик на предложение банка"] == 0][
        #     "Доход"].median()
        # fig.add_vline(x=median_target_0, line_dash="dash", line_color="red",
        #               name="Медиана для Отрицательного отклика")


        return fig




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