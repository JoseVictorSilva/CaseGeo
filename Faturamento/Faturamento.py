import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from plotGrafico import Grafico

def Faturamento():
   # Inicializa as variáveis guardadas
   with open('ArquivosBase/DadosProcessados.pkl','rb') as f:
      X_lista_rio, y_lista_rio, X_lista_sp, bairros_sp, colunas, y_faturamento = pickle.load(f)

   # Inicializo os arquivos, o original para usar como previsor que vem junto do faturamento, o floresta para usar como base dos dados escalados e tratados
   # e o média para colocar junto do floresta  
   base_rio = pd.read_excel('ArquivosBase/DadosDesafioCientista.xlsx')
   base_sp= pd.read_excel('ArquivosBase/FlorestaRandomicaPotencial3.xlsx')
   base_potencial_sp = pd.read_excel('MediaPotencial.xlsx')

   # Tiro a coluna potencial da floresta para adicionar a média no lugar
   base_sp.drop('potencial', axis = 1, inplace=True)
   base_sp['potencial'] = base_potencial_sp.loc[:,"Potencial"].values
   colunas = base_sp.columns.values

   # Os mesmos tratamentos feitos na main, terão de ser feitos de novo
   base_rio.fillna(base_rio['população'].mean(), inplace=True)
   base_rio.loc[base_rio['rendaMedia'] == '-', 'rendaMedia'] = 0

   base_rio['popDe20a24'] = base_rio['popDe20a24']+base_rio['popDe25a34']+base_rio['popDe35a49']
   base_rio['domiciliosA1'] =base_rio['domiciliosA1'] + base_rio['domiciliosA2']+base_rio['domiciliosB1']+base_rio['domiciliosB2']
   
   base_rio = base_rio.rename(columns={'popDe20a24': 'popDe20a49'})
   base_rio = base_rio.rename(columns={'domiciliosA1': 'domiciliosA1B2'})

   base_rio.drop('popDe25a34', axis=1, inplace=True)
   base_rio.drop('popDe35a49', axis=1, inplace=True)
   base_rio.drop('domiciliosA2', axis=1, inplace=True)
   base_rio.drop('domiciliosB1', axis=1, inplace=True)
   base_rio.drop('domiciliosB2', axis=1, inplace=True)

   base_rio['popDe20a49peso'] = base_rio['popDe20a49']

   # Troco os valores de potencial da base original para os valores que são representados pelo labelEncoder()
   base_rio['potencial'].replace({'Médio': 2, 'Alto':0, 'Baixo': 1}, inplace=True)
   
   # Crio a variável previsora e a classe para treinar com as colunas que quero
   X_lista_rio = base_rio.loc[base_rio['estado']=="RJ", ['população','popAte9','popDe10a14','popDe15a19','popDe20a49','popDe50a59','popMaisDe60',
   'domiciliosA1B2','domiciliosC1',
   'domiciliosC2','domiciliosD',
   'domiciliosE','rendaMedia',
   'popDe20a49peso', 'potencial']]
   y_lista_rio = base_rio.loc[base_rio['estado']=="RJ", "faturamento"].values

   # Crio a variável que será prevista junto de seus valores que não estou usando no momento
   X_lista_sp = base_sp.loc[base_sp['estado']=="SP", "população":"potencial"]
   bairros_sp = base_sp.loc[base_sp['estado']=="SP", "codigo":"estado"].values

   # Uso o encode na coluna potencial
   label_encoder_lista = LabelEncoder()
   X_lista_sp.iloc[:,14] = label_encoder_lista.fit_transform(X_lista_sp.iloc[:,14])

   # Escalo as classes previsoras
   scaler = StandardScaler()
   X_lista_rio = scaler.fit_transform(X_lista_rio)
   y_lista_rio = scaler.fit_transform(y_lista_rio.reshape(-1,1))

   # Uso o regressor para prever o faturamento
   regressor_multiplo_casas = LinearRegression()
   regressor_multiplo_casas.fit(X_lista_rio, y_lista_rio)
   previsoes = regressor_multiplo_casas.predict(X_lista_sp)

   # Volto as previsões para o valor comum
   X_lista_sp.iloc[:,14] = label_encoder_lista.inverse_transform(X_lista_sp.iloc[:,14].astype(int))

   previsoes = scaler.inverse_transform(previsoes)

   # O DataFrame é criado então salvo nos ArquivosBase, junto as informações que não uso + os previsores + a classe
   df = pd.DataFrame(bairros_sp,columns=colunas[0:4])
   df2 = pd.DataFrame(X_lista_sp.iloc[:,0:14],columns=colunas[4:18])
   df3 = pd.DataFrame(X_lista_sp,columns=['potencial'])
   df4 = pd.DataFrame(previsoes,columns=['faturamento'])
   dfGeral = pd.concat([df,df2,df3,df4], axis=1)
   dfGeral.to_excel(r'Faturamento.xlsx', index=False)
   Grafico()
