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
import sklearn as sk


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
CAST(SUM(t1.CLOSED_FL) AS INT64) as "LOAN_NUM_CLOSED"
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
db = duckdb.query(q).to_df().drop(['ID_CLIENT', 'ID_CLIENT_2',
                 'ID_CLIENT_3', 'ID_LOAN_2', 'ID_CLIENT_4', 'ID_2',
                 'CLOSED_FL_2', 'ID_LOAN'], axis=1)

st.title('Анализ клиентов банка')
st.write('Это приложение позволяет провести разведочный анализ данных, софрмировать портрет клиентов, '
         'выявить зависимости между признаками клиента и его склонности к положительному или отрицательному отклику на предложение банка.')

image = Image.open('img_1.png')
st.image(image, use_column_width=True)

tab1, tab2, tab3 = st.tabs(['Таблица', 'Разведочный анализ', 'Оценка'])

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
        corr = numerical_signs.corr()
        plt.figure(figsize=(15, 11))
        sns.heatmap(corr,fmt='.3f', annot=True, cmap='PuRd', linewidths=.5)
        plt.title('Матрица корреляций', fontdict={'fontsize': 16, 'fontweight': 'bold'})
        return plt




    numerical_signs = db.drop(['EDUCATION',
                 'MARITAL_STATUS', 'REG_ADDRESS_PROVINCE',
                 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE',
                 'GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR', 'FAMILY_INCOME', 'AGREEMENT_RK', 'ID'], axis=1)



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

with (tab3):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.svm import SVC
    import streamlit as st
    import pandas as pd

    with st.form(key='my_form'):
        # Создание набора полей для ввода значений
        age = st.number_input("Возраст", min_value=18, max_value=100, value=30)
        socstatus_work = st.radio("Трудоустроен", ['Да', 'Нет'])
        socstatus_pens = st.radio("Пенсионер", ['Да', 'Нет'])
        gender = st.selectbox("Пол", db.GENDER.unique())
        child_total = st.number_input("Количество детей", min_value=0,
                                      max_value=15, value=0)
        dependants = st.number_input("Количество иждивенцев", min_value=0,
                                     max_value=15, value=0)
        personal_income = st.number_input("Личный доход", min_value=0, value=50000)
        loan_num_total = st.number_input("Общее количество кредитов", min_value=0,
                                         value=1)
        loan_num_closed = st.number_input("Количество закрытых кредитов",
                                          min_value=0, value=0)
        education = st.selectbox("Образование", db.EDUCATION.unique())
        gen_title = st.selectbox("Должность", db.GEN_TITLE.unique())
        family_income = st.number_input("Семейный доход", min_value=0, value=50000)
        submit_button = st.form_submit_button(label='Предсказать')

    input_data = {
        'AGE': age,
        'SOCSTATUS_WORK_FL': 1 if socstatus_work == 'Да' else 0,
        'SOCSTATUS_PENS_FL': 1 if socstatus_pens == 'Да' else 0,
        'GENDER': 1 if gender == 'Мужской' else 0,
        'CHILD_TOTAL': child_total,
        'DEPENDANTS': dependants,
        'PERSONAL_INCOME': personal_income,
        'LOAN_NUM_TOTAL': loan_num_total,
        'LOAN_NUM_CLOSED': loan_num_closed,
        'EDUCATION': education,
        'GEN_TITLE': gen_title,
        'FAMILY_INCOME': family_income
    }



    # Создание датафрейма из введенных значений
    input_df = pd.DataFrame([input_data])


    def forecast(input_df, db):
        # Создание экземпляра SimpleImputer с стратегией замены отсутствующих значений на среднее
        imputer = SimpleImputer(strategy='most_frequent')
        db = db[['TARGET', 'AGE', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
                 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME',
                 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED', 'EDUCATION', 'GEN_TITLE',
                 'FAMILY_INCOME']]
        db_imputed = imputer.fit_transform(db)

        db = pd.DataFrame(db_imputed, columns=db.columns)

        db_encoded = pd.get_dummies(db, columns=['EDUCATION', 'GEN_TITLE',
                                                 'FAMILY_INCOME'])
        input_df_encoded = pd.get_dummies(input_df,
                                          columns=['EDUCATION', 'GEN_TITLE',
                                                   'FAMILY_INCOME'])
        missing_columns = set(db_encoded.columns) - set(
            input_df_encoded.columns)
        for col in missing_columns:
            input_df_encoded[col] = 0
        column_order = db_encoded.columns
        input_df_encoded = input_df_encoded.reindex(columns=column_order,
                                                    fill_value=0).drop(
            'TARGET', axis=1)
        # db_encoded = pd.get_dummies(db, columns=['EDUCATION',
        #              'MARITAL_STATUS', 'REG_ADDRESS_PROVINCE',
        #              'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE',
        #              'GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR', 'FAMILY_INCOME'])

        # Подготовка данных для обучения модели
        X = db_encoded.drop('TARGET', axis=1)
        y = db_encoded['TARGET']

        # Разделение данных на обучающий и тестовый наборы
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42)
        y_train = y_train.astype('bool')
        y_test = y_test.astype('bool')

        # Нормирование значений признаков с помощью Min-Max Scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_input_scaled = scaler.transform(input_df_encoded)


        # Создание и обучение модели логистической регрессии
        # model = LogisticRegression()
        model = SVC(C=1.0, kernel='rbf', gamma='scale')
        model.fit(X_train, y_train)

        # Предсказания на тестовом наборе данных
        y_pred = model.predict(X_input_scaled)

        return y_pred
        # st.write("Предсказанное значение целевой переменной:", y_pred)


    if submit_button:
        # Вызов функции для предсказания значения целевой переменной
        y_pred = forecast(input_df, db)
        color = 'rgba(0, 128, 0, 0.3)' if y_pred == 1 else 'rgba(255, 0, 0, 0.3)'
        text_color = 'rgba(0, 128, 0, 0.8)' if y_pred == 1 else 'rgba(255, 0, 0, 0.8)'
        predict = 'Скорее всего откликнется на предложение Банка' if y_pred == 1 else 'Скорее всего отклонит предложение Банка'
        st.markdown(f'''
            <div style="border: 2px solid {color}; background-color: {color}; padding: 10px; border-radius: 10px; text-align: center">
                <span style="color: {text_color};">{predict}</span>
            </div>
        ''', unsafe_allow_html = True)
        # st.markdown(f'<p style="color: {color};"> {predict}</p>',
        #             unsafe_allow_html=True)