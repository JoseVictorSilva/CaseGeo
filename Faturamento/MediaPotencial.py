import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from Faturamento.Faturamento import Faturamento

def MediaPotencial():
   # Separo os 3 resultados encontrados pelos Modelos
   potencial_arvore= pd.read_excel('ArquivosBase/ArvorePotencial2.xlsx')
   potencial_floresta = pd.read_excel('ArquivosBase/FlorestaRandomicaPotencial3.xlsx')
   potencial_redes_neurais = pd.read_excel('ArquivosBase/RedeNeuralPotencial2.xlsx')

   # Essa guardará as informações das condições
   resultado_geral = []

   # O loop checa os calores previstos dos modelos, seguindo a regra de importância Floresta > Redes > Árvore
   #  lógica: SE ARVORE == FLORESTA E ARVORE == REDE add(ARVORE)
   # SE ARVORE == FLORESTA E FLORESTA != REDE add(ARVORE)
   # SE ARVORE != FLORESTA E FLORESTA == REDE add(FLORESTA)
   # SE ARVORE == REDE E ARVORE != FLORESTA add(ARVORE)  
   for indice, linha in potencial_arvore.iterrows():
      if linha['potencial'] == potencial_floresta.loc[indice,'potencial'] and linha['potencial'] == potencial_redes_neurais.loc[indice,'potencial']:
         resultado_geral.append(linha['potencial'])
      elif linha['potencial'] == potencial_floresta.loc[indice,'potencial'] and linha['potencial'] != potencial_redes_neurais.loc[indice,'potencial']:
         resultado_geral.append(linha['potencial'])
      elif linha['potencial'] != potencial_floresta.loc[indice,'potencial'] and potencial_floresta.loc[indice,'potencial'] == potencial_redes_neurais.loc[indice,'potencial']:
         resultado_geral.append(potencial_floresta.loc[indice,'potencial'])
      elif linha['potencial'] == potencial_redes_neurais.loc[indice,'potencial'] and potencial_floresta.loc[indice,'potencial'] != potencial_redes_neurais.loc[indice,'potencial']:
         resultado_geral.append(linha['potencial'])
      else:
         resultado_geral.append(potencial_floresta.loc[indice,'potencial'])

   # Crio o com os resultados e salvo eles para usar no arquivo de FATURAMENTO         
   df = pd.DataFrame(resultado_geral,columns=['Potencial'])
   df['Potencial'].replace({2:'Médio', 0:'Alto', 1:'Baixo'}, inplace=True)
   df.to_excel(r'MediaPotencial.xlsx', index=False)
   print('==== PASSOU PELA MÉDIA ====')
   Faturamento()
