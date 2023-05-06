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

# Cargamos los datos del archivo CSV
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
df = pd.read_csv(url, names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"], na_values='?')

#ver si cada columna tiene caracteres no numericos
df.apply(lambda x: sum(x.apply(type) == str), axis=0)
#Pasar todos los valores a numericos
df = df.apply(pd.to_numeric, errors='coerce')
#Eliminar las filas con valores nulos
df = df.dropna()

df['age'], binage = pd.cut(df['age'], 10, retbins=True, labels = np.arange(10))
df['trestbps'], bintrestbps = pd.cut(df['trestbps'], 10, retbins=True, labels = np.arange(10))
df['chol'], binchol = pd.cut(df['chol'], 10, retbins=True, labels = np.arange(10))
df['thalach'], binthalach= pd.cut(df['thalach'], 10, retbins=True, labels = np.arange(10))
df['oldpeak'], binoldpeak = pd.cut(df['oldpeak'], 10, retbins=True, labels = np.arange(10))

# Eliminamos filas con valores faltantes
datos = df.dropna()

# Solo nos quedamos con las columnas que nos interesan
datos = datos.drop(columns=['restecg', 'age', 'trestbps', 'chol', 'fbs'], axis=1)

# Convertimos la columna de diagnóstico en un valor binario
datos['num'] = datos['num'].apply(lambda x: 0 if x == 0 else 1)


#Función para hacer inferencia bayesiana
def bayesian_inference(sex, cp, thalach, exang, oldpeak, slope, ca, thal):
    
    # Creamos el modelo bayesiano
    modelHill = BayesianNetwork([('sex', 'thal'), ('cp', 'exang'), ('oldpeak', 'slope'), ('slope', 'thalach'), ('thal', 'num'), ('num', 'ca'), ('num', 'cp'), ('num', 'oldpeak')])

    # Estimamos las distribuciones de probabilidad usando MLE
    modelHill.fit(datos, estimator=MaximumLikelihoodEstimator)

    # Estimamos las distribuciones de probabilidad usando MLE y BayesianEstimator
    modelHill.fit(datos, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

    # Hacemos inferencias en el modelo bayesiano
    from pgmpy.inference import VariableElimination

    infer2 = VariableElimination(modelHill)

    thalach = pd.cut([thalach], binthalach, labels = np.arange(10))[0]
    oldpeak = pd.cut([oldpeak], binoldpeak, labels = np.arange(10))[0]

    evidence = {'sex': sex, 'cp': cp, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    q = infer2.query(['num'], evidence=evidence)

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
app.layout = html.Div([    html.H1('Test preciso de enfermedad cardíaca',style={'text-align': 'center', 'color': 'White', 'font-weight': 'bold', 'background-color': '#b3d9ff','z-index': '1'}),    
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

            html.Label('¿Cuál es la pendiente del segmento ST'),#diferente de 1
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
    html.Div( id='output'),
    html.Button('Calcular', id='submit', n_clicks=0),
    html.Br(),
    html.Br(),    

], className='container', style={'font-family': 'system-ui', 'background-color': '#f2f2f2'})

@app.callback(
    Output('output', 'children'),
    [Input('submit', 'n_clicks')],
    [State('sex', 'value'), State('cp', 'value'), State('thalach', 'value'), State('exang', 'value'), State('oldpeak', 'value'), State('slope', 'value'), State('ca', 'value'), State('thal', 'value')]
)
    
def calculate_probability(n_clicks, sex, cp, thalach, exang, oldpeak, slope, ca, thal):
    if not n_clicks:
        return ''
    else:
        result = bayesian_inference(sex, cp, thalach, exang, oldpeak, slope, ca, thal)
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
