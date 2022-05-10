import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def Arvore():
   # Inicializa as variáveis guardadas
   with open('ArquivosBase/DadosProcessados.pkl','rb') as f:
      X_lista_rio, y_lista_rio, X_lista_sp, bairros_sp, colunas, y_faturamento = pickle.load(f)

   # Importante passo para a previsão, transformar as classes em números sendo 0 = Alto, 1 = Baixo, 2 = Médio
   label_encoder_lista = LabelEncoder()
   y_lista_rio = label_encoder_lista.fit_transform(y_lista_rio)
   
   # Crio a classe previsora, com os melhores parâmetros testas, se for de curiosidade, os parâmetros estão em Testes/testes.py 
   arvore = DecisionTreeClassifier(criterion= 'entropy', min_samples_leaf= 1, min_samples_split= 5, splitter= 'best') # random = gera os mesmos resultados
   arvore.fit(X_lista_rio,y_lista_rio)
   previsoes = arvore.predict(X_lista_sp)

   # Gero um gráfico de comportamento, caso seja de interesse, está na mesma pasta do nome do arquivo
   plt.title("Quantidade de Classificações de Risco")
   plt.xlabel("Classes")
   plt.ylabel("Quantidade")

   # Separo a variável previsora por sua ocorrência ex: Baixo: 10x
   unique, counts = np.unique(previsoes, return_counts=True)

   fig = plt.bar(unique,counts, label = "Grupo 1")
   plt.savefig('Arvore/Barras.png')

   # O DataFrame é criado então salvo nos ArquivosBase, junto as informações que não uso + os previsores + a classe
   df = pd.DataFrame(bairros_sp,columns=colunas[0:4])
   df2 = pd.DataFrame(X_lista_sp[:,0:13],columns=colunas[4:17])
   df3 = pd.DataFrame(X_lista_sp[:,13],columns=['popDe20a49peso'])
   df4 = pd.DataFrame(previsoes,columns=['potencial'])
   dfGeral = pd.concat([df,df2,df3,df4], axis=1)
   dfGeral.to_excel(r'./ArquivosBase/ArvorePotencial.xlsx', index=False)
   print('==== PASSOU PELA ÁRVORE ====')