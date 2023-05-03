import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import plotly.express as px

#Función para hacer inferencia bayesiana
def bayesian_inference(age, sex, cp, trestbps, chol, fbs, exang):
    
    # Cargamos los datos del archivo CSV
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    df = pd.read_csv(url, names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"], na_values='?')

    # Eliminamos filas con valores faltantes
    df = df.dropna()

    


    # Convertimos la columna de diagnóstico en un valor binario
    df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

    # Creamos el modelo bayesiano
    model = BayesianNetwork([('age', 'chol'), ('age', 'fbs'), ('age', 'trestbps'), ('sex', 'chol'),
                            ('sex', 'fbs'), ('sex', 'trestbps'), ('chol', 'num'), ('fbs', 'num'), ('trestbps', 'num'), 
                            ('num','exang'), ('num','cp')])

    # Estimamos las distribuciones de probabilidad usando MLE y BayesianEstimator
    # Estimamos las distribuciones de probabilidad usando MLE
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    model.fit(df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

    # Hacemos inferencias en el modelo bayesiano
    infer = VariableElimination(model)

    #inferencia para un paciente con los datos especificados
    q = infer.query(['num'], evidence={'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,'exang': exang})

    return q

# Función para devolver los valores normales o deseables de los parámetros
def valoresNormales(edad, sexo, cp, trestbps, chol, fbs, exang):
    #Valores deseables de los parametros
    valores ={
        'edad': (20, 80),
        'sexo': ('Hombre', 'Mujer'),
        'cp': ('Asintomático', 'Angina típica'),
        'trestbps': (90, 120),
        'chol': (150, 200),
        'fbs': (0, 120),
        'exang': ('No', 'Sí'),
    }


#Se crear la aplicación en dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server


# Se crea el diseño del tablero
app.layout = html.Div([    html.H1('Test rápido de enfermedad cardíaca',style={'text-align': 'center', 'color': 'White', 'font-weight': 'bold', 'background-color': '#b3d9ff','z-index': '1'}),    
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
            html.Label('¿Glucemia en ayunas? Si es así, ¿es mayor a 120 mg/dl?'),
            html.P('La glucemia en ayunas debe ser medida después de un ayuno de al menos 8 horas. Además, se recomienda no consumir alcohol ni alimentos ricos en azúcar durante las 24 horas previas al análisis de sangre. Consulta con tu médico para obtener más información sobre cómo prepararte para el análisis de glucemia.',
                   style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Dropdown(id='fbs', options=[{'label': 'Mayor a 120 mg/dl', 'value': 1}, {'label': 'Menor a 120 mg/dl', 'value': 0}],
                         placeholder='Azúcar en sangre en ayunas'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row rounded'),



    html.Br(),
    html.Div( id='output'),
    html.Button('Calcular', id='submit', n_clicks=0),
    html.Br(),
    html.Br(),    



], className='container', style={'font-family': 'system-ui', 'background-color': '#f2f2f2'})




@app.callback(
    Output('output', 'children'),
    [Input('submit', 'n_clicks')],
    [State('age', 'value'),State('sex', 'value'),State('cp', 'value'),State('trestbps', 'value'),State('chol', 'value'),State('fbs', 'value'),State('exang', 'value')])
    
def calculate_probability(n_clicks, age, sex, cp, trestbps, chol, fbs, exang):
    if not n_clicks:
        return ''
    else:
        result = bayesian_inference(age, sex, cp, trestbps, chol, fbs, exang)
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
@app.callback(Output('output', 'style'),
              [Input('submit', 'n_clicks')],
              [State('n-clicks', 'data')])
def toggle_div_visibility(n_clicks, data):
    if n_clicks is None:
        n_clicks = 0
    if n_clicks % 2 == 0:
        return {'display': 'none'}
    return {'display': 'block'}

@app.callback(Output('n-clicks', 'data'),
              [Input('submit', 'n_clicks')],
              [State('n-clicks', 'data')])
def update_n_clicks(n_clicks, data):
    if n_clicks is None:
        n_clicks = 0
    return n_clicks
    

#Se ejecuta la aplicación
if __name__ == '__main__':
    app.run_server(debug=True,port =8070)
