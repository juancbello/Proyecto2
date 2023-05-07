import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State


import plotly.graph_objs as go
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import plotly.express as px
import psycopg2

# ------------------------- Cargar base de datos -------------------------

HOST = 'heart.cejvodvrmsa6.us-east-1.rds.amazonaws.com' # Punto de enlace
PORT = 5432 # Puerto
USER = 'postgres' # Usuario
PASSWORD = '123contra' # Contraseña
DB_NAME = 'postgres' # Nombre de la base de datos

# Conectarse a la base de datos
engine = psycopg2.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, dbname=DB_NAME)
# Leer la tabla "heart_disease" en un DataFrame
df = pd.read_sql_query('SELECT * FROM heart_disease', engine)
df_graph = df.copy()
# Cerrar la conexión
engine.close()

# ------------------------- Tratamiento de datos -------------------------

df['age'], binage = pd.cut(df['age'], 10, retbins=True, labels = np.arange(10))
df['trestbps'], bintrestbps = pd.cut(df['trestbps'], 10, retbins=True, labels = np.arange(10))
df['chol'], binchol = pd.cut(df['chol'], 10, retbins=True, labels = np.arange(10))
df['thalach'], binthalach= pd.cut(df['thalach'], 10, retbins=True, labels = np.arange(10))
df['oldpeak'], binoldpeak = pd.cut(df['oldpeak'], 10, retbins=True, labels = np.arange(10))

# Eliminamos filas con valores faltantes
datos = df.dropna()

# ------------------------- Crear modelo bayesiano test rapido -------------------------

# Solo nos quedamos con las columnas que nos interesan
datos1 = datos.drop(columns=['restecg', 'thalach', 'oldpeak', 'slope', 'ca', 'thal'], axis=1)

# Convertimos la columna de diagnóstico en un valor binario
datos1['num'] = datos1['num'].apply(lambda x: 0 if x == 0 else 1)

#Función para hacer inferencia bayesiana
def bayesian_inference_tr(age, sex, cp, trestbps, chol, fbs, exang):

    # Creamos el modelo bayesiano
    model = BayesianNetwork([('age', 'chol'), ('age', 'fbs'), ('age', 'trestbps'), ('sex', 'chol'),
                            ('sex', 'fbs'), ('sex', 'trestbps'), ('chol', 'num'), ('fbs', 'num'), ('trestbps', 'num'), 
                            ('num','exang'), ('num','cp')])

    # Estimamos las distribuciones de probabilidad usando MLE y BayesianEstimator
    model.fit(datos1, estimator=MaximumLikelihoodEstimator)
    model.fit(datos1, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

    # Hacemos inferencias en el modelo bayesiano
    infer = VariableElimination(model)

    # Pasar de valores continuos a discretos
    age = pd.cut([age], binage, labels = np.arange(10))[0]
    trestbps = pd.cut([trestbps], bintrestbps, labels = np.arange(10))[0]
    chol = pd.cut([chol], binchol, labels = np.arange(10))[0]

    #inferencia para un paciente con los datos especificados
    q = infer.query(['num'], evidence={'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,'exang': exang})

    return q
# ------------------------- Crear modelo bayesiano test preciso -------------------------

# Solo nos quedamos con las columnas que nos interesan
datos2 = datos.drop(columns=['restecg', 'age', 'trestbps', 'chol', 'fbs'], axis=1)

# Convertimos la columna de diagnóstico en un valor binario
datos2['num'] = datos2['num'].apply(lambda x: 0 if x == 0 else 1)

#Función para hacer inferencia bayesiana
def bayesian_inference_tp(sex, cp, thalach, exang, oldpeak, slope, ca, thal):
    
    # Creamos el modelo bayesiano
    modelHill = BayesianNetwork([('sex', 'thal'), ('cp', 'exang'), ('oldpeak', 'slope'), ('slope', 'thalach'), ('thal', 'num'), ('num', 'ca'), ('num', 'cp'), ('num', 'oldpeak')])

    # Estimamos las distribuciones de probabilidad usando MLE
    modelHill.fit(datos2, estimator=MaximumLikelihoodEstimator)

    # Estimamos las distribuciones de probabilidad usando MLE y BayesianEstimator
    modelHill.fit(datos2, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

    # Hacemos inferencias en el modelo bayesiano
    from pgmpy.inference import VariableElimination

    infer2 = VariableElimination(modelHill)

    # Pasar de valores continuos a discretos
    thalach = pd.cut([thalach], binthalach, labels = np.arange(10))[0]
    oldpeak = pd.cut([oldpeak], binoldpeak, labels = np.arange(10))[0]

    evidence = {'sex': sex, 'cp': cp, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    q = infer2.query(['num'], evidence=evidence)

    return q

#---------------------------------------------------------------GRAFICAS-------------------------------------------------------------------------------------------------------------------------

# GRAFICA UNO SEXO Y EDAD

# Crear una columna 'disease' con valores binarios indicando si cada paciente tiene o no enfermedad cardiaca
df_graph['disease'] = df_graph['num'].apply(lambda x: 1 if x > 0 else 0)
# Agrupar los datos por edad y calcular la proporción de pacientes con enfermedad cardiaca en cada grupo
age_counts = df_graph.groupby('age')['disease'].agg(['count', 'sum'])
age_counts['proportion'] = age_counts['sum'] / age_counts['count']
# Crear gráfico de barras para las edades 
fig = px.bar(age_counts, x=age_counts.index, y='count', title='Número de pacientes con enfermedad cardiaca por edad',
             labels={'x': 'Edad', 'y': 'Número de pacientes'})
# Multiplicar por el total de pacientes la proporción de pacientes con enfermedad cardiaca para cada edad
fig.add_scatter(x=age_counts.index, y=age_counts['proportion'] * age_counts['count'], name='Pacientes con enfermedad cardiaca',
                line=dict(color='red', width=4))
# Cambiar el color de las barras a un tono azul más claro
fig.update_traces(marker_color='blue', marker_line_color='darkblue', marker_line_width=1, opacity=0.7)
# Ajustar el ancho de las barras
fig.update_layout(bargap=0.1, bargroupgap=0.05)
# Agregar leyenda
fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
# Cambiar tipo de letra y tamaño
fig.update_layout(font=dict(family='Arial', size=14))
# Cambiar las leyendas de los ejes
fig.update_layout(xaxis_title='Edad', yaxis_title='Número de pacientes')


fig1 = px.histogram(df_graph[df_graph['num'] != 0], x='age', color='sex', barmode='group', marginal='box', nbins=20,
                    labels={'age': 'Age', 'num': 'Num', 'sex': 'Sex'}, color_discrete_map={0: 'pink', 1: 'blue'},
                    category_orders={"sex": [1,0]}, 
                    title="Distribución de enfermedad cardiaca por sexo y edad")

fig1.update_traces(marker_line_width=0, opacity=0.75, hovertemplate=None)
fig1.update_layout(legend_title="Sex", legend=dict(title_font=dict(size=14), font=dict(size=12)))



# GRAFICA DOS
# GRAFICA SIGNOS VITALES
fig2 = px.scatter_matrix(df_graph, dimensions=['fbs', 'chol', 'trestbps'], color='num', opacity=0.7,
                         labels={'fbs': 'Fasting Blood Sugar', 'chol': 'Cholesterol', 'trestbps': 'Resting Blood Pressure', 'num': 'Num'})

# Arreglar el tamaño de las gráficas
fig2.update_layout(height=600, width=2000)


# GRAFICA ANGINAL

fig3 = px.histogram(df_graph, x='cp', color='exang', barmode='group', nbins=5,
                    labels={'cp': 'Chest Pain Type', 'num': 'Num', 'exang': 'Exercise Induced Angina'})

# Agregar leyenda
fig3.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
# Cambiar tipo de letra y tamaño
fig3.update_layout(font=dict(family='Arial', size=14))
# Cambiar las leyendas de los ejes
fig3.update_layout(xaxis_title='Tipo de dolor de pecho', yaxis_title='Número de pacientes')
# Cambiar la leyenda de exang por si o no
fig3.update_layout(legend_title="Angina inducida por ejercicio", legend=dict(title_font=dict(size=14), font=dict(size=12), itemsizing='constant'))
# En vez de 0 y 1, poner si y no
fig3.update_layout(xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4], ticktext=['Tipo 1', 'Tipo 2', 'Tipo 3', 'Tipo 4']))


# ------------------------- Diseño de la app -------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Inicializar la app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Contenido de las pestañas

# ------------------------- tab 1 -------------------------
# Contenido de las pestañas
tab1_content = html.Div([
    html.H1('Bienvenido al aplicativo de detección de enfermedad cardíaca'),
    html.Br(),
    html.P('Queremos mostrale datos que pueden ser de su interés antes de realizar el test de enfermedad cardíaca.'),
    html.Br(),
    html.P('En el siguiente grafico, realizamos un conteo de personas que padecen de complicaciones cardiaca. La personas se encuentran agrupadas por edad'),
    html.Br(),  
    dcc.Graph(id='graph1', figure=fig),
    html.P('En el siguiente grafico podemos comparar los signos vitales de los pacientes con la existencia de un diagnostico(color amarillo). Los datos utilizados son: colesterol, azucar en ayunas y presion arterial.'),
    html.Br(), 
    html.P('Es importante tener en cuenta que las variable FBS toma el valor de 1 si los niveles de azucar estan fuera de lo normal. A si mismo, el colesterol y la presion arterial, se encuentran divididos en 10 intervalos de la misma proporcion.'),  
    html.Br(), 
    dcc.Graph(id='graph2', figure=fig2),
    html.Br(),   
    html.P('En esta ultima visualizacion, podemos observar la cuenta de los pacientes diagnosticados con enfermedades cardiacas agrupados por tipo de angina y si presentan o no angina inducida por ejercicio.'),
    html.Br(), 
    html.P('Es importante tener en cuenta que los valores numericos de angina son:'),   
    html.Br(), 
    html.P('              1: Angina típica.'),
    html.P('              2: Angina atípica.'),
    html.P('              3: Dolor no anginal.'),
    html.P('              4: Asintomatico.'),
    dcc.Graph(id='graph3', figure=fig3)
])

# ------------------------- tab 2 -------------------------

tab2_content = html.Div([    html.H1('Test rápido de enfermedad cardíaca',style={'text-align': 'center', 'color': 'White', 'font-weight': 'bold', 'background-color': '#b3d9ff','z-index': '1'}),    
                       html.Div([        html.P('Mensaje de advertencia: Ninguna recomendacion presente en este aplicativo puede reemplazar la opinion de un medico, se recomiendo siempre consultar a un profesional de la salud.Antes de tomar el test, debe realizarse los siguientes exámenes: presión sistolica, colesterol (Prueba de sangre), glucemia en ayunas (Prueba de sangre).')    ], style={'text-align': 'center', 'color': 'red', 'font-weight': 'bold'}),

    html.Br(),  
    dcc.Store(id='n-clicks', data=0), 
    html.Div([
    
        html.Div([
            html.Label('¿Cuál es tu edad?'), #29-77
            dcc.Input(id='age', type='number', placeholder='Edad'),
        ], className='six columns', style={'margin-top': '10px'}),

        html.Div([
            html.Label('¿Eres hombre o mujer?'),
            dcc.Dropdown(id='sex', options=[{'label': 'Hombre', 'value': 1}, {'label': 'Mujer', 'value': 0}], placeholder='Sexo'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),

    html.Div([
        html.Div([
            html.Label('¿Tienes dolor torácico? Si es así, ¿qué tipo de dolor es?'),
            html.Ul([
                html.Li('Angina típica: dolor opresivo, generalmente en el centro del pecho, que se puede describir como una sensación de opresión, ardor, constricción o presión.'),
                html.Li('Angina atípica: puede presentar síntomas similares a la angina típica, pero puede tener una ubicación o características diferentes del dolor.'),
                html.Li('Dolor no anginal: puede ser descrito como un dolor punzante, agudo o quemante en el pecho, pero no está relacionado con el corazón.'),
                html.Li('Asintomático: ausencia de síntomas de dolor torácico.')
            ],style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Dropdown(id='cp', options=[{'label': 'Angina típica', 'value': 1}, {'label': 'Angina atípica', 'value': 2},
                                        {'label': 'Dolor no anginal', 'value': 3}, {'label': 'Asintomático', 'value': 4}],
                        placeholder='Dolor torácico'),
        ], className='six columns', style={'margin-top': '10px', 'margin-bottom': '10px'}),
    
        html.Div([
            html.Label('¿Cuál es tu presión arterial en reposo?'),#94-200
            html.P('La presión arterial debe medirse cuando te encuentres en reposo, sentado o acostado, durante al menos 5 minutos. Asegúrate de no haber consumido cafeína ni haber realizado actividad física intensa en la última hora antes de la medición. Además, evita hablar o moverte mientras se toma la medida. Si tienes dudas acerca de cómo medir tu presión arterial, consulta con tu médico.',
                   style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Input(id='trestbps', type='number', placeholder='Presión arterial en reposo'),
            html.Label('Teniendo en cuenta tu anterior respuesta, sientes dolor toracico cuando realizas ejercicio?'),
            dcc.Dropdown(id='exang', options=[{'label': 'Si', 'value': 1}, {'label': 'No', 'value': 0}],
                         placeholder='Angina inducida por ejercicio'),
        ], className='six columns', style={'margin-top': '10px', 'margin-bottom': '10px'}),
    ], className='row'),

    html.Div([
        html.Div([
            html.Label('¿Cuál es tu nivel de colesterol sérico?'),#126-564
            html.Div('El nivel de colesterol sérico debe ser medido después de un ayuno de al menos 12 horas. Además, se recomienda no consumir alcohol ni alimentos ricos en grasas durante las 24 horas previas al análisis de sangre. Consulta con tu médico para obtener más información sobre cómo prepararte para el análisis de colesterol.',
                     style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Input(id='chol', type='number', placeholder='Colesterol sérico'),
        ], className='six columns', style={'margin-top': '10px', 'margin-bottom': '10px'}),

        html.Div([
            html.Label('¿Glucemia en ayunas? Si es así, ¿es mayor a 125 mg/dl?'),
            html.P('La glucemia en ayunas debe ser medida después de un ayuno de al menos 8 horas. Además, se recomienda no consumir alcohol ni alimentos ricos en azúcar durante las 24 horas previas al análisis de sangre. Consulta con tu médico para obtener más información sobre cómo prepararte para el análisis de glucemia.',
                   style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Dropdown(id='fbs', options=[{'label': 'Mayor a 125 mg/dl', 'value': 1}, {'label': 'Menor a 125 mg/dl', 'value': 0}],
                         placeholder='Azúcar en sangre en ayunas'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row rounded'),

    html.Br(),
    html.Div( id='output2'),
    html.Button('Calcular', id='submit2', n_clicks=0),
    html.Br(),
    html.Br(),    

]),
    
# ------------------------- Tab 3 -------------------------
tab3_content = html.Div([
    
    html.H1('Test preciso de enfermedad cardíaca',style={'text-align': 'center', 'color': 'White', 'font-weight': 'bold', 'background-color': '#b3d9ff','z-index': '1'}),    
    html.Div([        html.P('Mensaje de advertencia: Ninguna recomendacion presente en este aplicativo puede reemplazar la opinion de un medico, se recomiendo siempre consultar a un profesional de la salud.Antes de tomar el test, debe realizarse los siguientes exámenes: Frecuencia cardíaca máxima (Prueba de esfuerzo), Depresión del segmento ST (ECG), Pendiente del segmento ST (ECG), Número de arterias coloreadas (Fluoroscopia) y Gammagrafía cardíaca con talio.')    ], style={'text-align': 'center', 'color': 'red', 'font-weight': 'bold'}),

    html.Br(),  
    dcc.Store(id='n-clicks', data=0), 
    html.Div([

        html.Div([
            html.Label('¿Eres hombre o mujer?'),
            dcc.Dropdown(id='sex', options=[{'label': 'Hombre', 'value': 1}, {'label': 'Mujer', 'value': 0}], placeholder='Sexo'),
        ], className='six columns', style={'margin-top': '10px'}),

        html.Div([
            html.Label('¿Cuál es tu frecuencia cardíaca máxima?'), #150
            dcc.Input(id='thalach', type='number', placeholder='Frecuecnia cardíaca máxima'),
        ], className='six columns', style={'margin-top': '10px'}),

    ], className='row'),

    html.Div([
        html.Div([
            html.Label('¿Tienes dolor torácico? Si es así, ¿qué tipo de dolor es?'),
            html.Ul([
                html.Li('Angina típica: dolor opresivo, generalmente en el centro del pecho, que se puede describir como una sensación de opresión, ardor, constricción o presión.'),
                html.Li('Angina atípica: puede presentar síntomas similares a la angina típica, pero puede tener una ubicación o características diferentes del dolor.'),
                html.Li('Dolor no anginal: puede ser descrito como un dolor punzante, agudo o quemante en el pecho, pero no está relacionado con el corazón.'),
                html.Li('Asintomático: ausencia de síntomas de dolor torácico.')
            ],style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Dropdown(id='cp', options=[{'label': 'Angina típica', 'value': 1}, {'label': 'Angina atípica', 'value': 2},
                                        {'label': 'Dolor no anginal', 'value': 3}, {'label': 'Asintomático', 'value': 4}],
                        placeholder='Dolor torácico'),

            html.Label('Teniendo en cuenta tu anterior respuesta, sientes dolor toracico cuando realizas ejercicio?'),
            dcc.Dropdown(id='exang', options=[{'label': 'Si', 'value': 1}, {'label': 'No', 'value': 0}],
                         placeholder='Angina inducida por ejercicio'),
            
            html.Label('¿Cuál es el número de arterias coloreadas por fluoroscopia?'),#debe ser 0

            html.Ul([
                html.Li('0: número normal'),
                html.Li('1: tener 1 arteria coloreada'),
                html.Li('2: tener 2 arterias coloreadas'),
                html.Li('3: tener 3 arterias coloreadas')
            ],style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Dropdown(id='ca', options=[{'label': '0', 'value': 0}, {'label': '1', 'value': 1},
                                        {'label': '2', 'value': 2}, {'label': '3', 'value': 3}],
                        placeholder='Número de arterias coloreadas'),
                        

        ], className='six columns', style={'margin-top': '10px', 'margin-bottom': '10px'}),

            html.Div([
            html.Label('¿Cuál es la depresión del segmento ST?'),#menor de 1
            html.P('Solo los profesionales médicos capacitados deben medir la depresión del segmento ST, ya que requiere un electrocardiograma (ECG) y una interpretación adecuada de los resultados.',
                   style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Input(id='oldpeak', type='number', placeholder='Depresión del segmento ST'),

            html.Label('¿Cuál es la pendiente del segmento ST?'),#diferente de 1
            html.Ul([
                html.Li('Ascendente: el segmento ST se eleva gradualmente, pero no alcanza la línea de base.'),
                html.Li('Plano: el segmento ST no se eleva ni se deprime.'),
                html.Li('Descendente: el segmento ST se deprime gradualmente, pero no alcanza la línea de base.'),
            ],style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Dropdown(id='slope', options=[{'label': 'Ascendente', 'value': 1}, {'label': 'Plano', 'value': 2}, {'label': 'Descendente', 'value': 3}], placeholder='Pendiente del segmento ST'),

            html.Label('¿Cuál es la gammagrafía cardíaca con talio?'),
            html.Ul([
                html.Li('Normal: no hay defectos de perfusión.'),
                html.Li('Defecto fijo: defecto de perfusión que no cambia con el tiempo.'),
                html.Li('Defecto reversible: defecto de perfusión que se vuelve normal con el tiempo.'),
            ],style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Dropdown(id='thal', options=[{'label': 'Normal', 'value': 3}, {'label': 'Defecto fijo', 'value': 6}, {'label': 'Defecto reversible', 'value': 7}], placeholder='Gammagrafía cardíaca con talio'),


        ], className='six columns', style={'margin-top': '10px', 'margin-bottom': '10px'}),
    ], className='row'),

    html.Br(),
    html.Div( id='output3'),
    html.Button('Calcular', id='submit3', n_clicks=0),
    html.Br(),
    html.Br(),    
])


# Diseño de la app
app.layout = html.Div([
    html.H1('Test Enfermedad Cardíaca'),
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='Inicio', value='tab1'),
        dcc.Tab(label='Test Rápido', value='tab2'),
        dcc.Tab(label='Test Preciso', value='tab3'),
    ]),
    html.Div(id='tab-content')
])

# ------------------------- CALLBACKS -------------------------

# Callback botón test rápido
@app.callback(Output('output2', 'children'),[Input('submit2', 'n_clicks')],[State('age', 'value'),State('sex', 'value'),State('cp', 'value'),State('trestbps', 'value'),State('chol', 'value'),State('fbs', 'value'),State('exang', 'value')])
def callback_test_rapido(n_clicks, age, sex, cp, trestbps, chol, fbs, exang):
    if not n_clicks:
        return ''
    else:
        result = bayesian_inference_tr(age, sex, cp, trestbps, chol, fbs, exang)
        probability = round(result.values[0], 2)
        #creamos recomendaciones para cada signo vital

        if probability <= 0.5:
            fondo = '#9ef0ea'
        else:
            fondo = '#ea8b8b'

        recomendacionCHOL =''
        recomendacionFBS = ''
        recomendacionTRESTBPS = ''
        recomendacionANGINA = ''
        recomendacionEXANG = ''


        #recomendaciones para el dolor de pecho
        if cp == 1 or cp == 2:
            recomendacionANGINA = 'Recuerda que la angina de pecho es una condición médica seria que requiere supervisión y tratamiento médico. Si tienes síntomas de angina de pecho, es importante buscar atención médica inmediata. Te recomendamos evitar situaciones que pueden desencadenar la angina, como el frío extremo, el esfuerzo físico intenso y el estrés emocional. Realiza un seguimiento regular con un profesional de la salud para monitorear la condición y ajustar el plan de tratamiento según sea necesario.'
        else:
            recomendacionANGINA = ''
        
        if exang == 1:
            recomendacionEXANG = 'Para prevenir o reducir los sintomas de la angina de pecho inducida, te recomendamos realizar un programa de ejercicio estructurado y supervisado por un profesional de la salud asi como tambien realizar un calentamiento adecuado antes del ejercicio y un enfriamiento adecuado después del mismo.'
        else:
            recomendacionEXANG = ''

        #definimos las reglas de color para graficar los sifnos vitales
        colors = []
        if chol >= 125 and chol <= 200:
            colors.append('#63ef54')
        else:
            colors.append('#ef3b3b')
            recomendacionCHOL ='Tus niveles de colesterol se encuentran fuera del rango normal (125-200). Se recomienda una dieta saludable, ejercicio regular, mantener un peso saludable, no fumar y limitar el alcohol. Siempre consulta a un profesional de la salud antes de hacer cambios significativos.'

        if trestbps <= 120:
            colors.append('#63ef54')
        else:
            colors.append('#ef3b3b')
            recomendacionTRESTBPS = 'Tu presion arterial esta por encima de 120/80 mmHg. Para mantenerte en niveles mas saludables, es importante seguir una dieta saludable, baja en sodio, rica en frutas, verduras y cereales integrales, hacer ejercicio regularmente, mantener un peso saludable, no fumar y limitar el consumo de alcohol.'

        #creamos un grafico de barras para representar los signos vitales
        fig = go.Figure(go.Bar(
            x=['Colesterol', 'Presión'],
            y=[chol, trestbps],
            marker_color=colors
        ))

        #recomendacion para azucar
        if fbs ==1:
            recomendacionFBS = 'Tu nivel de azucar se encuentra sobre 125 mg/Dl. Para mantenerte en niveles mas saludables, es importante seguir una dieta saludable, rica en fibras y baja en carbohidratos simples, hacer ejercicio regularmente, mantener un peso saludable y tomar medicamentos recetados por un médico si es necesario. También es importante monitorear los niveles de azúcar en sangre y consultar con un profesional de la salud.'
        else:
            recomendacionFBS = ''

        #en el return de la funcion, cargamos toda la informacion para las recomendaciones
        return html.Div([
            f'                   La probabilidad de que padezcas de alguna complicacion cardiaca es de {probability*100}%',
            html.Br(),
            html.Div(recomendacionANGINA),
            html.Br(),
            html.Div(recomendacionEXANG),
            html.Div([
    
            html.Div([
                dcc.Graph(id='example-graph',  figure=fig),
            ], className='six columns', style={'margin-top': '10px'}),

            html.Div([
                html.Label(recomendacionCHOL),
                html.Br(),
                html.Label(recomendacionFBS),
                html.Br(),
                html.Label(recomendacionTRESTBPS),
            ], className='six columns', style={'margin-top': '10px'}),
        ], className='row'),
            ],
            style={'color': 'black','background-color':fondo}
        )

# Callback botón test preciso
@app.callback(Output('output3', 'children'),[Input('submit3', 'n_clicks')],[State('sex', 'value'), State('cp', 'value'), State('thalach', 'value'), State('exang', 'value'), State('oldpeak', 'value'), State('slope', 'value'), State('ca', 'value'), State('thal', 'value')])
def callback_test_preciso(n_clicks, sex, cp, thalach, exang, oldpeak, slope, ca, thal):
    if not n_clicks:
        return ''
    else:
        result = bayesian_inference_tp(sex, cp, thalach, exang, oldpeak, slope, ca, thal)
        probability = round(result.values[0], 2)

        #creamos recomendaciones para cada signo vital
        if probability <= 0.5:
            fondo = '#9ef0ea'
        else:
            fondo = '#ea8b8b'

        recomendacionTHALACH =''
        recomendacionSLOPE = ''
        recomendacionOLDPEAK = ''
        recomendacionCA = ''
        recomendacionTHAL = ''
        recomendacionANGINA = ''
        recomendacionEXANG = ''


        #recomendaciones para el dolor de pecho
        if cp == 1 or cp == 2:
            recomendacionANGINA = 'Recuerda que la angina de pecho es una condición médica seria que requiere supervisión y tratamiento médico. Si tienes síntomas de angina de pecho, es importante buscar atención médica inmediata. Te recomendamos evitar situaciones que pueden desencadenar la angina, como el frío extremo, el esfuerzo físico intenso y el estrés emocional. Realiza un seguimiento regular con un profesional de la salud para monitorear la condición y ajustar el plan de tratamiento según sea necesario.'
        else:
            recomendacionANGINA = ''
        
        if exang == 1:
            recomendacionEXANG = 'Para prevenir o reducir los sintomas de la angina de pecho inducida, te recomendamos realizar un programa de ejercicio estructurado y supervisado por un profesional de la salud asi como tambien realizar un calentamiento adecuado antes del ejercicio y un enfriamiento adecuado después del mismo.'
        else:
            recomendacionEXANG = ''

        #definimos las reglas de color para graficar los signos vitales
        colors = []
        if thalach >= 150:
            colors.append('#63ef54')
        else:
            colors.append('#ef3b3b')
            recomendacionTHALACH = 'Tus frecuencia cardíaca máxima se encuentran fuera del rango normal (superiores a 150 bpm). Se recomienda una dieta saludable, ejercicio regular, mantener un peso saludable, no fumar y limitar el alcohol. Siempre consulta a un profesional de la salud antes de hacer cambios significativos.'

        if oldpeak > 1:
            colors.append('#63ef54')
        else:
            colors.append('#ef3b3b')
            recomendacionOLDPEAK = 'Tu depresión del segmento ST es irregular. Recomendamos que consultes con un especialista en cardiología para una evaluación más detallada. La depresión del segmento ST puede ser un indicador de enfermedad arterial coronaria y es importante identificar cualquier problema de salud lo antes posible. También te recomendamos llevar un registro regular de tu presión arterial y frecuencia cardíaca para que puedas monitorear tu salud cardiovascular.'

         #creamos un grafico de barras para representar los signos vitales
        fig = go.Figure(go.Bar(
            x=['Frecuencia cardíaca máxima', 'Depresión del segmento ST'],
            y=[thalach, oldpeak],
            marker_color=colors
        ))

        #recomendacion para pendiente
        if slope != 1:
            recomendacionOLDPEAK = 'Tu pendiente del segmento ST es irregular. Sería recomendable que consultes con un especialista para evaluar los resultados de tu electrocardiograma y obtener un diagnóstico preciso sobre la condición de tu corazón. Una pendiente irregular en el segmento ST podría ser indicativo de algún problema cardiovascular, y es importante tener un diagnóstico temprano para poder tomar las medidas necesarias y evitar complicaciones mayores.'
        else:
            recomendacionOLDPEAK = ''

        #recomendacion para Número de arterias coloreadas por fluoroscopia
        if ca != 0:
            recomendacionCA = 'Tu número de arterias coloreadas por fluoroscopia es mayor a 0.Tu resultado indica que se ha identificado la presencia de obstrucciones en al menos una de tus arterias coronarias. Se recomienda que sigas las recomendaciones de tu médico y realices un seguimiento adecuado para evaluar y tratar cualquier problema potencial en tu corazón.'
        else:
            recomendacionCA = ''
        
        #recomendacion para gammagrafía cardíaca con talio
        if thal != 3:
            recomendacionTHAL = 'Tu resultado de la gammagrafía cardíaca con talio es irregular. Se recomienda que consultes con un especialista en cardiología para una evaluación más detallada. La gammagrafía cardíaca con talio es una prueba que se utiliza para evaluar la función del corazón y detectar problemas cardíacos. Es importante identificar cualquier problema de salud lo antes posible, por lo que te recomendamos que consultes con un especialista para obtener un diagnóstico preciso.'
        else:
            recomendacionTHAL = ''

        #en el return de la funcion, cargamos toda la informacion para las recomendaciones
        return html.Div([
            f'                   La probabilidad de que padezcas de alguna complicacion cardiaca es de {probability*100}%',
            html.Br(),
            html.Div(recomendacionANGINA),
            html.Br(),
            html.Div(recomendacionEXANG),
            html.Div([
    
            html.Div([
                dcc.Graph(id='example-graph',  figure=fig),
            ], className='six columns', style={'margin-top': '10px'}),

            html.Div([
                html.Label(recomendacionTHALACH),
                html.Br(),
                html.Label(recomendacionOLDPEAK),
                html.Br(),
                html.Label(recomendacionSLOPE),
                html.Br(),
                html.Label(recomendacionCA),
                html.Br(),
                html.Label(recomendacionTHAL),

            ], className='six columns', style={'margin-top': '10px'}),
        ], className='row'),
            
            ],
            style={'color': 'black','background-color':fondo}
        )

# Callback para mostrar el contenido de la pestaña seleccionada
@app.callback(Output('tab-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab1':
        return tab1_content
    elif tab == 'tab2':
        return tab2_content
    elif tab == 'tab3':
        return tab3_content

if __name__ == '__main__':
    app.run_server(debug=True)

