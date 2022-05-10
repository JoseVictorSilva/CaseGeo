import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

def RedeNeural():
   # Inicializa as variáveis guardadas
   with open('ArquivosBase/DadosProcessados.pkl','rb') as f:
      X_lista_rio, y_lista_rio, X_lista_sp, bairros_sp, colunas, y_faturamento  = pickle.load(f)
   
   # Importante passo para a previsão, transformar as classes em números sendo 0 = Alto, 1 = Baixo, 2 = Médio
   label_encoder_lista = LabelEncoder()
   y_lista_rio = label_encoder_lista.fit_transform(y_lista_rio)
  
   # Crio a classe previsora, com os melhores parâmetros testas, se for de curiosidade, os parâmetros estão em Testes/testes.py 
   rede_neural = MLPClassifier(activation= 'tanh', batch_size= 10, hidden_layer_sizes= (11, 11), max_iter= 1000, solver= 'sgd')
   rede_neural.fit(X_lista_rio,y_lista_rio)
   previsoes = rede_neural.predict(X_lista_sp)

   # Gero um gráfico de comportamento, caso seja de interesse, está na mesma pasta do nome do arquivo
   plt.title("Quantidade de Classificações de Risco")
   plt.xlabel("Classes")
   plt.ylabel("Quantidade")

   # Separo a variável previsora por sua ocorrência ex: Baixo: 10x
   unique, counts = np.unique(previsoes, return_counts=True)
   
   fig = plt.bar(unique,counts, label = "Grupo 1")
   plt.savefig('Barras.png')

   # O DataFrame é criado então salvo nos ArquivosBase, junto as informações que não uso + os previsores + a classe
   df = pd.DataFrame(bairros_sp,columns=colunas[0:4])
   df2 = pd.DataFrame(X_lista_sp[:,0:13],columns=colunas[4:17])
   df3 = pd.DataFrame(X_lista_sp[:,13],columns=['popDe20a49peso'])
   df4 = pd.DataFrame(previsoes,columns=['potencial'])
   dfGeral = pd.concat([df,df2,df3,df4], axis=1)
   dfGeral.to_excel(r'ArquivosBase/RedeNeuralPotencial2.xlsx', index=False)
   print('==== PASSOU PELA REDE ====')