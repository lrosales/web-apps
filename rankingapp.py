import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import math
import matplotlib.pyplot as plt
from highlight_text import fig_text
from adjustText import adjust_text
from PIL import Image
from urllib.request import urlopen
from adjustText import adjust_text
from highlight_text import fig_text
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
import plotly.express as px
import altair as alt
import plotly.express as px
pd.set_option('mode.use_inf_as_na', True)




import openpyxl
from pathlib import Path

#-----------------------------RAW DATA---------------------------------------------------------------

#Read In Data

DATA_URL = (r'/Users/omar/My Drive/Datasets/delanterosamericas.xlsx')

df = pd.read_excel(DATA_URL)

#edit identical strings in name colum

num = df.groupby('Jugador').cumcount()
df.loc[num.ne(0), 'Jugador'] += ' '+num.astype(str)

df.to_excel(DATA_URL, index = False)

st.title('Delanteros')


@st.cache
def load_data():
    df = pd.read_excel(DATA_URL)
    lowercase = lambda x: str(x).lower()
    df.rename(lowercase, axis='columns', inplace=True)
    return df


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load data
df = load_data()

#convert data to str to avoid errors
df = df.astype(str)


# Notify the reader that the data was successfully loaded.
data_load_state.text("Done!")

st.subheader('Raw data')


st.dataframe(df)

#--------------------------RATING------------------------------


def new_metrics(df):  
    
    #create Goles x remate metric
    df["goles %"] = df['Goles (excepto los penaltis)'] / df['Remates'] 
    
    #goal ratio
    df['Goal Ratio'] = df['Remates'] / df['Goles (excepto los penaltis)']
    
    #Create new column 90 min played
    df['90s'] = df['Minutos jugados'] / 90
    df['90s'] = df['90s'].round()
    
    #Create column with penalty xG
    df["pxG"] = df['Penaltis a favor'] * 0.76 
    
    #Create column with  npxG
    df["npxG"] = df['xG'] - df["pxG"] 
    
    #Create column with pxG per 90
    df["pxG/90"] = df['pxG'] / df["90s"] 
    
    #Create column with  npxG per 90
    df["npxG/90"] = df['xG/90'] - df["pxG/90"] 
    
    #Create column with  xG and npG per 90
    df["Sum xG and G x90"] = df['npxG/90'] + df["Goles, excepto los penaltis/90"] 

    #Create column with  xA and Assist per 90
    df["Sum xA and Assist x90"] = df['xA/90'] + df["Asistencias/90"] 
    
    
    #goal difference from xG p90
    df["xGDifference"] = df['Goles, excepto los penaltis/90'] - df['npxG/90'] 
    
    
    
    
    
#-----------------filter players----------------------
    
data = DATA_URL
minutos = 900
edad = 45

#-----------------------------------------

#Read In Data
df = pd.read_excel(data)

#edit identical strings in name/jugador column

num = df.groupby('Jugador').cumcount()
df.loc[num.ne(0), 'Jugador'] += ' '+num.astype(str)

#convert nan to 0

df.fillna(0, inplace=True)


#New Metrics ------------------------------------

new_metrics(df)

df['goles %'].replace(np.inf, 0, inplace=True)

#save DF with new column
    
df.to_excel(data, index = False)

#Read In Data
df = pd.read_excel(data)


          
#---------------------------------------------------


#define metrics to use 

offensemetrics = ['Jugador', 'Acciones de ataque exitosas/90', 'Tiros a la portería, %', 'goles %', 'Duelos atacantes ganados, %',
                  'Carreras en progresión/90', 'Aceleraciones/90', 'Sum xA and Assist x90', 'Desmarques/90', 'Jugadas claves/90', 'Ataque en profundidad/90', 'Sum xG and G x90']

defensemetrics = ['Jugador', 'Duelos defensivos ganados, %', 'Duelos aéreos ganados, %', 'Interceptaciones/90', 'Posesión conquistada después de una interceptación']

passingmetrics = ['Jugador', 'Precisión pases hacia adelante, %', 'Precisión pases laterales, %', 
                  'Precisión pases cortos / medios, %', 'Precisión pases largos, %', 'Precisión pases en el último tercio, %', 
                  'Precisión pases en profundidad, %', 'Precisión pases progresivos, %']

# add all metrics minus two Jugador columns

totalmetrics = ['Jugador', 'Acciones de ataque exitosas/90', 'Tiros a la portería, %', 'Duelos atacantes ganados, %', 'goles %', 'Jugadas claves/90',
                'Carreras en progresión/90', 'Aceleraciones/90', 'Sum xA and Assist x90', 'Desmarques/90', 'Ataque en profundidad/90', 
                'Duelos defensivos ganados, %', 'Duelos aéreos ganados, %', 'Interceptaciones/90', 
                'Posesión conquistada después de una interceptación', 'Precisión pases hacia adelante, %', 
                'Precisión pases laterales, %', 'Precisión pases cortos / medios, %', 'Precisión pases largos, %', 'Precisión pases en el último tercio, %', 
                'Precisión pases en profundidad, %', 'Precisión pases progresivos, %', 'Sum xG and G x90']
#convert nan to 0

df.fillna(0, inplace=True)








# OFFENCE -----------------------------------------------------

def list_players():
    df = pd.read_excel(data)
    return tuple(df['Jugador'])


def calculation_avr(df, nombre: str):
    params = list(df.columns)

    # drop jugador column
    params = params[1:]
    
    # gets all values/rows of specific player
    player = df.loc[df['Jugador'] == nombre].reset_index()
    player = list(player.loc[0])

    player = player[2:]

    # print(round(sum(player)/ len(player)))
    values = []
    for x in range(len(params)):
        values.append(math.floor(stats.percentileofscore(df[params[x]], player[x])))

    Rating = np.average(values)
    average_values = Rating.round()
    return {'player': nombre, 'average_values': average_values}
    

def calculation():
    df = pd.read_excel(data)

    # Only Players with more than x minutes and Only players with less than x age
    
    df = df.loc[(df['Minutos jugados'] > minutos) & (df['Edad'] < edad) & (df['Posición específica'] != 'GK') ]
    df.Jugador.unique()

    # Drop unneeded Values
    df.drop_duplicates(subset=['Jugador'], keep=False)
    
    #define OFFENSIVE METRICS --------------------------------------------------------------------->
    
    df = df[offensemetrics]

    
    #-------------------------------------------------------------------------------------->

    ratings = []
    for player_name in list_players():

        try:
            ratings.append(calculation_avr(df, player_name))
        except KeyError:
            continue
    df['Offence'] = np.nan
    for player in ratings:
        id_row = df.index[df['Jugador'] == player['player']].tolist()
        for id_player in id_row:
            df['Offence'][id_player] = player['average_values']
    
    #save DF with new column
    
    df.to_excel(r'/Users/omar/Desktop/Python_Proyects/Data_Sets/LigaMX/offence.xlsx', index = False)


def main():
    calculation()


if __name__ == '__main__':
    main()
    

#DEFENSE --------------------------------------------------------------------------------

def list_players():
    df = pd.read_excel(data)
    return tuple(df['Jugador'])


def calculation_avr(df, nombre: str):
    params = list(df.columns)

    # drop jugador column
    params = params[1:]
    
    # gets all values/rows of specific player
    player = df.loc[df['Jugador'] == nombre].reset_index()
    player = list(player.loc[0])

    player = player[2:]

    # print(round(sum(player)/ len(player)))
    values = []
    for x in range(len(params)):
        values.append(math.floor(stats.percentileofscore(df[params[x]], player[x])))

    Rating = np.average(values)
    average_values = Rating.round()
    return {'player': nombre, 'average_values': average_values}


def calculation():
    df = pd.read_excel(data)

    # Only Players with more than x minutes and Only players with less than x age
    
    df = df.loc[(df['Minutos jugados'] > minutos) & (df['Edad'] < edad) & (df['Posición específica'] != 'GK') ]
    df.Jugador.unique()

    # Drop unneeded Values
    df.drop_duplicates(subset=['Jugador'], keep=False)
    
    #define DEFENSIVE METRICS --------------------------------------------------------------------->
    
    df = df[defensemetrics]
    
    
      #-------------------------------------------------------------------------------------->

    ratings = []
    for player_name in list_players():

        try:
            ratings.append(calculation_avr(df, player_name))
        except KeyError:
            continue
    df['Defense'] = np.nan
    for player in ratings:
        id_row = df.index[df['Jugador'] == player['player']].tolist()
        for id_player in id_row:
            df['Defense'][id_player] = player['average_values']
    
    
    df.to_excel(r'/Users/omar/Desktop/Python_Proyects/Data_Sets/LigaMX/defense.xlsx', index = False)


def main():
    calculation()


if __name__ == '__main__':
    main()
    
    
#PASSING -----------------------------------------------------------------------------------

def list_players():
    df = pd.read_excel(data)
    return tuple(df['Jugador'])


def calculation_avr(df, nombre: str):
    params = list(df.columns)

    # drop jugador column
    params = params[1:]
    
    # gets all values/rows of specific player
    player = df.loc[df['Jugador'] == nombre].reset_index()
    player = list(player.loc[0])

    player = player[2:]

    # print(round(sum(player)/ len(player)))
    values = []
    for x in range(len(params)):
        values.append(math.floor(stats.percentileofscore(df[params[x]], player[x])))

    Rating = np.average(values)
    average_values = Rating.round()
    return {'player': nombre, 'average_values': average_values}


def calculation():
    df = pd.read_excel(data)

    # Only Players with more than x minutes and Only players with less than x age
    
    df = df.loc[(df['Minutos jugados'] > minutos) & (df['Edad'] < edad) & (df['Posición específica'] != 'GK') ]
    df.Jugador.unique()

    # Drop unneeded Values
    df.drop_duplicates(subset=['Jugador'], keep=False)
    
    #define PASSING METRICS --------------------------------------------------------------------->
    
    df = df[passingmetrics]
    
    #-------------------------------------------------------------------------------------->

    ratings = []
    for player_name in list_players():

        try:
            ratings.append(calculation_avr(df, player_name))
        except KeyError:
            continue
    df['Passing'] = np.nan
    for player in ratings:
        id_row = df.index[df['Jugador'] == player['player']].tolist()
        for id_player in id_row:
            df['Passing'][id_player] = player['average_values']
    
    
    df.to_excel(r'/Users/omar/Desktop/Python_Proyects/Data_Sets/LigaMX/passing.xlsx', index = False)


def main():
    calculation()


if __name__ == '__main__':
    main()
    

#TOTAL ------------------------------------------------------

def list_players():
    df = pd.read_excel(data)
    return tuple(df['Jugador'])


def calculation_avr(df, nombre: str):
    params = list(df.columns)

    # drop jugador column
    params = params[1:]
    
    # gets all values/rows of specific player
    player = df.loc[df['Jugador'] == nombre].reset_index()
    player = list(player.loc[0])

    player = player[2:]

    # print(round(sum(player)/ len(player)))
    values = []
    for x in range(len(params)):
        values.append(math.floor(stats.percentileofscore(df[params[x]], player[x])))

    Rating = np.average(values)
    average_values = Rating.round()
    return {'player': nombre, 'average_values': average_values}


def calculation():
    df = pd.read_excel(data)

    # Only Players with more than x minutes and Only players with less than x age
    
    df = df.loc[(df['Minutos jugados'] > minutos) & (df['Edad'] < edad) & (df['Posición específica'] != 'GK') ]
    df.Jugador.unique()

    # Drop unneeded Values
    df.drop_duplicates(subset=['Jugador'], keep=False)
    
    #define TOTAL METRICS --------------------------------------------------------------------->
    
    df = df[totalmetrics]
    
    #-------------------------------------------------------------------------------------->

    ratings = []
    for player_name in list_players():

        try:
            ratings.append(calculation_avr(df, player_name))
        except KeyError:
            continue
    df['Total'] = np.nan
    for player in ratings:
        id_row = df.index[df['Jugador'] == player['player']].tolist()
        for id_player in id_row:
            df['Total'][id_player] = player['average_values']
    
    
    df.to_excel(r'/Users/omar/Desktop/Python_Proyects/Data_Sets/LigaMX/total.xlsx', index = False)


def main():
    calculation()


if __name__ == '__main__':
    main()

    
    
    
#Merge Dataframes -------------------------------------------------------------|-------------

defense = pd.read_excel(r'/Users/omar/Desktop/Python_Proyects/Data_Sets/LigaMX/defense.xlsx')
offence = pd.read_excel(r'/Users/omar/Desktop/Python_Proyects/Data_Sets/LigaMX/offence.xlsx')
passing = pd.read_excel(r'/Users/omar/Desktop/Python_Proyects/Data_Sets/LigaMX/passing.xlsx')
total = pd.read_excel(r'/Users/omar/Desktop/Python_Proyects/Data_Sets/LigaMX/total.xlsx')
info = pd.read_excel(data)
result = info.merge(defense,on='Jugador').merge(offence,on='Jugador').merge(passing,on='Jugador').merge(total,on='Jugador')

#add commas do player values
result['Valor de mercado']=result['Valor de mercado'].apply('{:,}'.format)

pd.set_option('display.max_rows', result.shape[0]+1)

#save result df to file
result.to_excel(r'/Users/omar/Desktop/Python_Proyects/Data_Sets/LigaMX/result.xlsx', index = False)

#----------------------------------------------------------------------------------------------

#Normalize Min/Max Data

scaler = MinMaxScaler()

result[['Offence', 'Defense', 'Passing', 'Total', 'Acciones de ataque exitosas/90', 'Jugadas claves/90', 'Tiros a la portería, %', 'Duelos atacantes ganados, %', 
        'Carreras en progresión/90', 'Aceleraciones/90', 'Sum xA and Assist x90', 'Desmarques/90', 'Ataque en profundidad/90',
        'Duelos defensivos ganados, %', 'Duelos aéreos ganados, %', 'Interceptaciones/90', 'Posesión conquistada después de una interceptación', 
        'Precisión pases hacia adelante, %', 'Precisión pases laterales, %', 'Precisión pases cortos / medios, %', 'Precisión pases largos, %', 
        'Precisión pases en el último tercio, %', 'Precisión pases en profundidad, %', 'goles %',
        'Precisión pases progresivos, %', 'Sum xG and G x90']] = scaler.fit_transform(result[['Offence', 'Defense', 'Passing', 'Total', 'Acciones de ataque exitosas/90',               'Tiros a la portería, %', 
        'Duelos atacantes ganados, %', 'Carreras en progresión/90', 'Aceleraciones/90', 'Sum xA and Assist x90', 'Jugadas claves/90',
        'Desmarques/90', 'Ataque en profundidad/90', 'Duelos defensivos ganados, %', 'Duelos aéreos ganados, %', 
        'Interceptaciones/90', 'Posesión conquistada después de una interceptación', 'Precisión pases hacia adelante, %', 
        'Precisión pases laterales, %', 'Precisión pases cortos / medios, %', 'Precisión pases largos, %', 'goles %',
        'Precisión pases en el último tercio, %', 'Precisión pases en profundidad, %', 'Precisión pases progresivos, %', 'Sum xG and G x90']])

#RENAME COLUMNS

result.rename(columns={
         'Total':'Total Index',
         'Offence':'Offence Index',
         'Defense':'Defense Index',
         'Passing':'Passing Index'
         }, inplace=True)

#--------------------------------------------------------------------------------------------------

#Sort By --------------------------------------------------------------------------

delanteros = result

delanteros['Total Index'] = delanteros[['Offence Index', 'Passing Index']].mean(axis=1)

delanteros = delanteros.sort_values('Total Index', ascending=False)
delanteros = delanteros.reset_index(drop=True)



#Offence and passing metrics
delanteros = (delanteros[['Jugador','Equipo', 'Equipo durante el período seleccionado', 'Edad', 'Pasaporte', 'Posición específica', 'Valor de mercado', 'Total Index', 'Minutos jugados', 'Offence Index', 
                          'Passing Index', 'Sum xG and G x90', 'Sum xA and Assist x90', 'Jugadas claves/90', 'Acciones de ataque exitosas/90', 'goles %', 
                  'Tiros a la portería, %', 'Duelos atacantes ganados, %', 'Carreras en progresión/90', 'Aceleraciones/90',  
                  'Desmarques/90', 'Ataque en profundidad/90', 'Duelos aéreos ganados, %',
                  'Precisión pases hacia adelante, %', 'Precisión pases laterales, %', 'Precisión pases cortos / medios, %', 'Precisión pases largos, %', 
                  'Precisión pases en el último tercio, %', 'Precisión pases en profundidad, %', 'Precisión pases progresivos, %']])





#result DEFENSE *Style Index Colors

def styler(v):
    if v > .84:
        return 'background-color:#3498DB' #blue
    elif v > .60:
        return 'background-color:#45B39D' #green
    if v < .15:
        return 'background-color:#E74C3C' #red
    elif v < .40:
        return 'background-color:#E67E22' #orange
    else:
        return 'background-color:#F7DC6F'  #yellow
    


st.subheader('Rating percentil vs todos los delanteros con mas de 800 minutos jugados')

st.write(delanteros.style.applymap(styler, subset=['Total Index', 'Offence Index', 'Passing Index', 'Acciones de ataque exitosas/90', 'goles %',
                  'Tiros a la portería, %', 'Duelos atacantes ganados, %', 'Carreras en progresión/90', 'Aceleraciones/90', 'Sum xA and Assist x90', 'Sum xG and G x90',
                  'Desmarques/90', 'Jugadas claves/90', 'Ataque en profundidad/90', 'Duelos aéreos ganados, %',  'Precisión pases hacia adelante, %', 
                  'Precisión pases laterales, %', 
                  'Precisión pases cortos / medios, %', 'Precisión pases largos, %', 'Precisión pases en el último tercio, %', 'Precisión pases en profundidad, %', 
                  'Precisión pases progresivos, %']).set_precision(2))

#--------------------------DELANTEROS STATS-------------------------


st.subheader('Metricas de efectividad')



#result all 3 aspects *Style Index Colors



def styler(v):
    if v > 0.08:
        return 'background-color:#E74C3C' #red
    elif v > -0.08:
         return 'background-color:#52CD34' #green
    if v < -0.08:
         return 'background-color:#E74C3C' #red
    # elif v < .40:
    #     return 'background-color:#E67E22' #orange
    # else:
    #     return 'background-color:#F7DC6F'  #yellow


#Sort By

df = df.sort_values('goles %', ascending=False)

#filter players 
minutos = 1200
edad = 45
#drop posicion
posicion = 'GK'
df = df[~(df['Minutos jugados'] <= minutos)] 
df = df[~(df['Edad'] >= edad)] 
df = df[~(df['Posición específica'] == posicion)] 



#edit identical strings in name/jugador column

num = df.groupby('Jugador').cumcount()
df.loc[num.ne(0), 'Jugador'] += ' '+num.astype(str)

#Choose columns to show

df = (df[['Jugador','Equipo', 'Posición específica', 'Pasaporte', 'Edad', 'Minutos jugados', '90s', 'npxG/90', "Goles, excepto los penaltis/90",
          'Goles (excepto los penaltis)', 'npxG', 
          'Remates', 'Remates/90', 'Goal Ratio', 'xGDifference']])


# print table

st.write(df.style.applymap(styler, subset=['xGDifference']).set_precision(2))





#-------------------- Bar chart ----------------------------

st.subheader('Los Delanteros con mejor suma de xG y Goles (Sin penales) por 90 minutos jugados')

figsize = 60,70


#Filter players

df = df[~(df['npxG/90'] < 0.30)] 


#Create column with  npxG per 90 plus Goles marcados (sin penales)

df["sumnpxGandgolesx90"] = df['npxG/90'] + df["Goles, excepto los penaltis/90"] 
df = df.sort_values('sumnpxGandgolesx90').reset_index()

#sets name to index
df.set_index("Jugador",drop=True,inplace=True)
#df = df.drop('level_0', 1)
df = df.sort_values('sumnpxGandgolesx90')

st.bar_chart(df[["npxG/90", "Goles, excepto los penaltis/90"]])


#---------------

# figsize = 60,100

# axis = df[["npxG/90", "Goles, excepto los penaltis/90"]].plot(kind="barh", stacked=True, figsize=(figsize))

# #plt.suptitle("Los jugadores con mejor se posicionan para rematar a gol en la Liga MX\n", fontsize=13, y=.92, x=.53)
# plt.title("Los jugadores con mejor suma de xG y goles por 90 minutos jugados | Liga MX Último Año", loc='left', size=15)


# plt.ylabel("Jugador")

# plt.xlabel("Suma de npxG/90 y Goles, excepto los penaltis/90")

# plt.legend(title='', loc='lower right', labels=["npxG/90", "Goles, excepto los penaltis/90"])

# fig = axis.get_figure()

# st.pyplot(fig)


