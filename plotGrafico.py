import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def Grafico():
    base= pd.read_excel('ArquivosBase/DadosDesafioCientista.xlsx')
    faturamento = pd.read_excel('Faturamento/Faturamento.xlsx')

    base['popDe20a24'] = base['popDe20a24']+base['popDe25a34']+base['popDe35a49']
    base['domiciliosA1'] =base['domiciliosA1'] + base['domiciliosA2']+base['domiciliosB1']+base['domiciliosB2']
    base = base.rename(columns={'popDe20a24': 'popDe20a49'})
    base = base.rename(columns={'domiciliosA1': 'domiciliosA1B2'})
    bairros = base.loc[base['estado']=="SP", ['nome']].values
    X_base = base.loc[base['estado']=="SP", ['popAte9','popDe10a14','popDe15a19',
    'popDe20a49','popDe50a59',
    'popMaisDe60','rendaMedia']].values
    potencial = faturamento.loc[:,['potencial']]
    faturamento_valores = faturamento.loc[:,['faturamento']].values

    bairros = pd.DataFrame(bairros, columns=['Bairros'])
    df = pd.DataFrame(X_base,columns=['9-','10 a 14','15 a 19',
    '20 a 49','50 a 59',
    '60+','Renda'])
    df2 = pd.DataFrame(potencial, columns=potencial.columns)
    df3 = pd.DataFrame(faturamento_valores, columns=['Faturamento'])
    dfGeral = pd.concat([bairros,df,df2,df3], axis=1)
    dfGeral.to_excel(r'ResultadosPrevistos.xlsx', index=False)


    # fig = make_subplots(rows=2, cols=1)

    # fig.append_trace(go.Scatter(
    #     x=dfGeral["Bairros"],
    #     y=dfGeral["potencial"],
    # ), row=1, col=1)

    # fig.append_trace(go.Line(
    #     x=dfGeral["Bairros"],
    #     y=dfGeral["Faturamento"],
    # ), row=2, col=1)

    # fig.update_layout(height=1800, width=1920, title_text="Gráficos")
    # fig.show()

    grafico1 = px.bar_polar(dfGeral, x="potencial", y="Bairros", color='potencial')
    grafico1.update_xaxes(categoryarray=['Baixo','Médio','Alto'])
    #grafico2 = px.line(dfGeral, x="Bairros", y="Faturamento")
    #grafico2.data[0].line.color = 'red'
    #grafico3 = go.Figure(data= (grafico1.data + grafico2.data))
    grafico1.show()

if __name__ == '__main__':
    Grafico()