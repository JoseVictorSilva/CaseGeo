import pickle
from re import template
from turtle import color, width
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def Grafico():
    # Usa as bases originais para pegar os valores originais e a faturamento com os valores atualizados
    base= pd.read_excel('ArquivosBase/DadosDesafioCientista.xlsx')
    faturamento = pd.read_excel('Faturamento/Faturamento.xlsx')

    # Faz a modificação dos dados
    base['popDe20a24'] = base['popDe20a24']+base['popDe25a34']+base['popDe35a49']
    base['domiciliosA1'] =base['domiciliosA1'] + base['domiciliosA2']+base['domiciliosB1']+base['domiciliosB2']
    base = base.rename(columns={'popDe20a24': 'popDe20a49'})
    base = base.rename(columns={'domiciliosA1': 'domiciliosA1B2'})
    bairros = base.loc[base['estado']=="SP", ['nome']].values

    # Usa apenas as colunas necessárias, sendo que potencial, faturamento e as infos se tornação parte do gráfico e do excel, 
    #   é melhor deixa-los em locais diferentes
    X_base = base.loc[base['estado']=="SP", ['popAte9','popDe10a14','popDe15a19',
    'popDe20a49','popDe50a59',
    'popMaisDe60','rendaMedia']].values
    potencial = faturamento.loc[:,['potencial']]
    faturamento_valores = faturamento.loc[:,['faturamento']].values
    
    # Troca o nome das colunas para uma visualização mais simples e cria os DataFrames com os dados infos|originais|potencial|faturamento
    bairros = pd.DataFrame(bairros, columns=['Bairros'])
    df = pd.DataFrame(X_base,columns=['9-','10 a 14','15 a 19',
    '20 a 49','50 a 59',
    '60+','Renda'])
    df2 = pd.DataFrame(potencial, columns=potencial.columns)
    df3 = pd.DataFrame(faturamento_valores, columns=['Faturamento'])
    
    # Junta os DataFrame e os ordena pelo potencial para facilitar a visualização no excel
    dfGeral = pd.concat([bairros,df,df2,df3], axis=1)
    dfGeral['potencial'] = pd.Categorical(dfGeral['potencial'], ["Alto", "Médio", "Baixo"])
    dfGeral.loc[dfGeral['Renda']=="-", 'Faturamento'] = 0
    dfGeral.sort_values(['potencial','Faturamento'], inplace=True, ascending=[True, False])
    dfGeral.to_excel(r'ResultadosPrevistos.xlsx', index=False)

    # Reorganizo para facilitar a visualização dessa vez no GRÁFICO
    dfGeral['potencial'] = pd.Categorical(dfGeral['potencial'], ["Baixo", "Médio", "Alto"])
    dfGeral.sort_values('potencial', inplace=True)
    
    # Criação do gráfico
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Adiciona o primeiro gráfico cruzando o potencial (Alto Médio e Baixo) entre Bairros
    fig.add_trace(
        go.Bar(x=dfGeral['potencial'], y=dfGeral['Bairros'], name="Bairros", orientation = 'h'),
        secondary_y=False,
    )

    # Adiciona o segundo gráfico cruzando o potencial (Alto Médio e Baixo) entre Faturamento
    fig.add_trace(
        go.Scatter(x=dfGeral['potencial'], y=dfGeral['Faturamento'], name="Faturamento", mode='markers'),
        secondary_y=True,
    )
    # Mudanças no gráfico, título, nome das barras e por fins
    fig.update_layout(
        title_text="Gráfico Geral dos Dados"
    )

    fig.update_xaxes(title_text="Potencial")
    fig.update_yaxes(title_text="<b>Bairros</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Faturamento</b>", secondary_y=True)
    fig.update_layout(plot_bgcolor='darkgray',width=1920, height=1080)
    fig.write_image("GraficoGeralDosDados.pdf")
    # Plota
    fig.show()

if __name__ == '__main__':
    Grafico()
