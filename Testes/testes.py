import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#|===============================================================================================================|  
#|                                               !!!AVISO!!!                                                     |
#| Quanto a este arquivo, recomendo a não execução, é um arquivo de testes de melhores modelos, então ele rodará |
#| os modelos ao menos 30x para achar suas melhores configurações, e isso pode demorar MUITO, não vai travar     |
#| mas pode demorar bastante, se ainda sim quiser usar, recomendo deixar as linhas das redes neurais comentadas. |
#|===============================================================================================================|

with open('ArquivosBase/DadosProcessados.pkl','rb') as f:
    X_lista_rio, y_lista_rio, X_lista_sp, bairros_sp, colunas, y_faturamento = pickle.load(f)

# 'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'random'
# 86.875%
parametros ={'criterion':['gini', 'entropy'],
             'splitter': ['best', 'random'],
             'min_samples_split': [2,5,10],
             'min_samples_leaf': [1,5,10]}
grid_search = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid=parametros)
grid_search.fit(X_lista_rio,y_lista_rio)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f'Melhores parametros: {melhores_parametros}')
print(f'Melhores resultados: {melhor_resultado*100}%\n')

# {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 10}
# 90%

parametros ={'criterion':['gini', 'entropy'],
             'n_estimators': [10,20,40],
             'min_samples_split': [1,2,3,4],
             'min_samples_leaf': [1,2,3]}
grid_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid=parametros)
grid_search.fit(X_lista_rio,y_lista_rio)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f'Melhores parametros: {melhores_parametros}')
print(f'Melhores resultados: {melhor_resultado*100}%\n')


# {'n_neighbors': 3, 'p': 2}

parametros ={'n_neighbors':[3,5,10,20],
            'p':[1,2]}
grid_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid=parametros)
grid_search.fit(X_lista_rio,y_lista_rio)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f'Melhores parametros: {melhores_parametros}')
print(f'Melhores resultados: {melhor_resultado*100}%\n')


# {'C': 1.5, 'solver': 'lbfgs', 'tol': 0.0001}
# 80.625%

parametros ={'tol':[0.0001,0.00001,0.000001],
            'C':[1.0,1.5,2.0],
            'solver':['lbfgs','sag','saga']}
grid_search = GridSearchCV(estimator = LogisticRegression(), param_grid=parametros)
grid_search.fit(X_lista_rio,y_lista_rio)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f'Melhores parametros: {melhores_parametros}')
print(f'Melhores resultados: {melhor_resultado*100}%\n')


# {'C': 2.0, 'kernel': 'linear', 'tol': 0.001}
# 82.5%

parametros ={'tol':[0.001,0.0001,0.00001],
            'C':[1.0,1.5,2.0],
            'kernel':['rbf','linear','poly', 'sigmoid']}
grid_search = GridSearchCV(estimator = SVC(), param_grid=parametros)
grid_search.fit(X_lista_rio,y_lista_rio)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f'Melhores parametros: {melhores_parametros}')
print(f'Melhores resultados: {melhor_resultado*100}%\n')


# (18+3)/2
# {'activation': 'relu', 'batch_size': 10, 'hidden_layer_sizes': (11, 11), 'max_iter': 1000, 'solver': 'sgd'}
# 94.375%
parametros ={'activation':['relu','logistic','tanh'],
            'solver':['adam','sgd'],
            'batch_size':[10],
            'hidden_layer_sizes':[(11,11)],
            'max_iter':[1000]}
grid_search = GridSearchCV(estimator = MLPClassifier(), param_grid=parametros)
grid_search.fit(X_lista_rio,y_lista_rio)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f'Melhores parametros: {melhores_parametros}')
print(f'Melhores resultados: {melhor_resultado*100}%\n')

resultados_arvore = []
resultado_random_forest = []
resultado_knn = []
resultado_logistica = []
resultado_svm = []
resultado_rede_neural = []
for i in range(30):
    kfold = KFold(n_splits=10,shuffle=True,random_state=i)
    
    arvore = DecisionTreeClassifier(criterion= 'entropy', min_samples_leaf= 1, min_samples_split= 5, splitter= 'best')
    scores = cross_val_score(arvore, X_lista_rio,y_lista_rio, cv = kfold)
    resultados_arvore.append(scores.mean())

    print('ÁRVORE')
    print(resultados_arvore)
    
    random_forest = RandomForestClassifier(criterion= 'gini', min_samples_leaf= 3, min_samples_split= 2, n_estimators= 10)
    scores = cross_val_score(random_forest, X_lista_rio,y_lista_rio, cv = kfold)
    resultado_random_forest.append(scores.mean())
    print('FLORESTA')
    print(resultado_random_forest)

    knn = KNeighborsClassifier(n_neighbors=3, p=2)
    scores = cross_val_score(knn, X_lista_rio,y_lista_rio, cv = kfold)
    resultado_knn.append(scores.mean())
    print('KNN')
    print(resultado_knn)

    logistica = LogisticRegression(tol=0.0001, C=1.5, solver='lbfgs')
    scores = cross_val_score(logistica, X_lista_rio,y_lista_rio, cv = kfold)
    resultado_logistica.append(scores.mean())
    print('LogisticRegression')
    print(resultado_logistica)  

    svm = SVC(C= 2.0, kernel= 'linear', tol= 0.001)
    scores = cross_val_score(svm, X_lista_rio,y_lista_rio, cv = kfold)
    resultado_svm.append(scores.mean())
    print('SVC')
    print(resultado_svm)    

    rede_neural = MLPClassifier(activation= 'relu', batch_size= 10, hidden_layer_sizes= (11, 11), max_iter= 1000, solver= 'sgd')
    scores = cross_val_score(rede_neural, X_lista_rio,y_lista_rio, cv = kfold)
    resultado_rede_neural.append(scores.mean())
    print('MLPClassifier')
    print(resultado_rede_neural)

resultados = pd.DataFrame({'Arvore':resultados_arvore,'Random Forest': resultado_random_forest, 'KNN':resultado_knn} )

resultados.to_excel(r'ResultadosTestes/ResultadosModelos.xlsx', index=False)