import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Arvore.Arvore import Arvore
from Faturamento.MediaPotencial import MediaPotencial
from FlorestaRand.RandomFlorest import Floresta
from RedeNeural.Neural import RedeNeural

# Definindo a função para executar o código
def main():
    # Abro a base de dados e faço alterações para facilitar o trabalho com o público alvo, juntado-os
    base= pd.read_excel('./ArquivosBase/DadosDesafioCientista.xlsx')
    base['popDe20a24'] = base['popDe20a24']+base['popDe25a34']+base['popDe35a49']
    base['domiciliosA1'] =base['domiciliosA1'] + base['domiciliosA2']+base['domiciliosB1']+base['domiciliosB2']
    base = base.rename(columns={'popDe20a24': 'popDe20a49'})
    base = base.rename(columns={'domiciliosA1': 'domiciliosA1B2'})
    
    # Passo um filtro de erros em algumas colunas que notei Diferenças
    base.fillna(base['população'].mean(), inplace=True)
    base.loc[base['rendaMedia'] == '-', 'rendaMedia'] = 0

    # Dropo colunas que não serão usadas, facilitando o plot final.
    base.drop('popDe25a34', axis=1, inplace=True)
    base.drop('popDe35a49', axis=1, inplace=True)
    base.drop('domiciliosA2', axis=1, inplace=True)
    base.drop('domiciliosB1', axis=1, inplace=True)
    base.drop('domiciliosB2', axis=1, inplace=True)

    # A criação dessa nova coluna, permite que o peso seja alterado, para saber mais sobre o peso de cada variável, veja em Testes/melhores.py
    base['popDe20a49peso'] = base['popDe20a49']

    # Salvo uma variável chamada colunas com seus nomes, facilita também no plot final dos dataframes
    colunas = base.columns.values

    # Separo as variaves de teste (previsoras e classes) das que serão previstas, além disso uso o append para colocar a 
    #   coluna que serve para alterar o peso, lembrando, aqui eu só pego valores que são transformados em numpy array, e não suas colunas
    X_lista_rio = base.loc[base['estado']=="RJ", "população":"rendaMedia"].values
    X_lista_rio = np.append(X_lista_rio, base.loc[base['estado']=="RJ",'popDe20a49peso'].values.reshape(-1,1), axis=1)
    y_lista_rio = base.loc[base['estado']=="RJ", "potencial"].values
    y_faturamento = base.loc[base['estado']=="RJ", "faturamento"].values
    X_lista_sp = base.loc[base['estado']=="SP", "população":"rendaMedia"].values
    X_lista_sp = np.append(X_lista_sp, base.loc[base['estado']=="SP",'popDe20a49peso'].values.reshape(-1,1), axis=1)
    bairros_sp = base.loc[base['estado']=="SP", "codigo":"estado"].values

    # Parte importante, escalar os valores numéricos para que não haja nenhum valor pendendo mais ou valendo mais só por sua quantia
    scaler = StandardScaler()
    X_lista_rio = scaler.fit_transform(X_lista_rio)
    X_lista_sp = scaler.fit_transform(X_lista_sp)
    y_faturamento = scaler.fit_transform(y_faturamento.reshape(-1,1))

    # Salvo as variáveis para os próximos códigos utilizarem-nas em um arquivo pickle
    with open('./ArquivosBase/DadosProcessados.pkl', mode = 'wb') as f:
        pickle.dump([X_lista_rio, y_lista_rio, X_lista_sp, bairros_sp, colunas, y_faturamento], f)
    
    # Inicializo então os processos de previsão, usando Arvores, Random Forest e Neural Network, 
    #   para saber mais sobre as escolhas, entrar em Testes/testes.py em que testo 6 tipos de redes e crio os melhores valores
    #   Ps: Não recomendo a execução do mesmo, ele funciona 100% mas demora por ter que fazer em loop os testes 30x principalmente
    #   das Redes Neurais 
    Arvore()
    Floresta()
    RedeNeural()

    # A média do Potencial é tirado, seguindo a regra de importância Floresta > Rede > Árvore, conforme os testes informaram
    MediaPotencial()

if __name__ == '__main__':
    main()
