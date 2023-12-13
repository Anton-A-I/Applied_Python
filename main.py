import  csv
import pandas as pd
import duckdb
import seaborn as sns
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import plotly as plt

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

# gender_mapping = {0: 'Женщина', 1: 'Мужчина'}
# db['GENDER'] = db['GENDER'].map(gender_mapping)



# q1 = '''
# SELECT AGE, COUNT(AGE) as COUNT,
# FROM db
# GROUP BY  AGE
# ORDER BY AGE ASC
# '''
#
# age_tg = duckdb.query(q1).to_df()
#
# fig = go.Figure()
# fig.add_trace(go.Bar(x=age_tg['AGE'], y=age_tg['COUNT'], text=age_tg['COUNT'], textposition='auto', textangle=0,  marker_color='#4361ee', name='Количество людей'))
# fig.add_trace(go.Scatter(x=age_tg['AGE'], y=age_tg['COUNT'].groupby(age_tg['AGE']).max(), line_shape='spline', mode='lines', name='Тренд', marker_color='#f72585'))
# fig.update_layout( title='Распределение людей по возрасту', xaxis_title='Возраст', yaxis_title='Количество людей')
# fig.show()
#
# q2 = '''
# SELECT CASE
#     WHEN GENDER = 1 THEN 'Мужчины'
#     WHEN  GENDER = 0 THEN 'Женщины'
# END as "Пол",
# COUNT(GENDER) as "Кол-во"
# FROM db
# GROUP BY  GENDER
# ORDER BY GENDER ASC
# '''
#
# gender = duckdb.query(q2).to_df()
#
# fig_gender = go.Figure()
# fig_gender.add_trace(go.Pie(labels = gender['Пол'], values = gender['Кол-во'], marker_colors=['#f72585', '#4361ee']))
# fig_gender.update_layout( title='Распределение людей по полу')
# fig_gender.show()
selected_genders = st.multiselect('Выберите пол', ['Мужчины', 'Женщины', 'Все'])

if selected_genders == 'Мужчины':
    gender_filter = 1
elif selected_genders == 'Женщины':
    gender_filter = 0
else:
    gender_filter = '1,0'
# age_filter = st.slider('Фильтр по возрасту', 0, 100, (0, 100))
q3 = '''
SELECT EDUCATION as "Образование",
COUNT(EDUCATION) as "Кол-во" 
FROM db
WHERE GENDER in ({})
GROUP BY  "Образование"
ORDER BY "Кол-во" DESC 
'''.format(gender_filter if gender_filter in [0, 1] else '0, 1')

education_data = duckdb.query(q3).to_df()

fig_education = go.Figure()
fig_education.add_trace(go.Pie(labels=education_data['Образование'], values=education_data['Кол-во'], marker_colors=['#f72585', '#7209b7', '#3a0ca3', '#4361ee', '#4cc9f0', '#560badff']))
fig_education.update_layout( title='Распределение людей по уровню образования')
fig_education.show()
