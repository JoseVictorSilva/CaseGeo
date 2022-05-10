import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier

def MelhoresVariaveis():
    base= pd.read_excel('ArquivosBase/DadosDesafioCientista.xlsx')
    base['popDe20a24'] = base['popDe20a24']+base['popDe25a34']+base['popDe35a49']
    base['domiciliosA1'] =base['domiciliosA1'] + base['domiciliosA2']+base['domiciliosB1']+base['domiciliosB2']
    base = base.rename(columns={'popDe20a24': 'popDe20a49'})
    base = base.rename(columns={'domiciliosA1': 'domiciliosA1B2'})
    base.fillna(base['população'].mean(), inplace=True)
    base.loc[base['rendaMedia'] == '-', 'rendaMedia'] = 0

    base.drop('popDe25a34', axis=1, inplace=True)
    base.drop('popDe35a49', axis=1, inplace=True)
    base.drop('domiciliosA2', axis=1, inplace=True)
    base.drop('domiciliosB1', axis=1, inplace=True)
    base.drop('domiciliosB2', axis=1, inplace=True)
    base['popDe20a49peso'] = base['popDe20a49']

    X_lista_rio = base.loc[base['estado']=="RJ", "população":"rendaMedia"].values
    y_lista_rio = base.loc[base['estado']=="RJ", "potencial"].values
    label_encoder_lista = LabelEncoder()
    y_lista_rio = label_encoder_lista.fit_transform(y_lista_rio)
    print(X_lista_rio)
    print(y_lista_rio)
    best_var = SelectKBest(score_func= chi2, k=4)
    fit = best_var.fit(X_lista_rio,y_lista_rio)
    features = fit.transform(X_lista_rio)

    print(f'Número original de variaveis: \n {X_lista_rio.shape[1]}\n')
    print(f'Número reduzido de variaveis: \n {features.shape[1]}\n')
    print(f'Variaveis selecionada: \n {features}\n')
    print(base.loc[:,"população":"popDe20a49peso"])

    modelo = ExtraTreesClassifier()
    modelo.fit(X_lista_rio, y_lista_rio)
    print(modelo.feature_importances_*100)
    print(base.loc[:,"população":"popDe20a49peso"])
    print(base)