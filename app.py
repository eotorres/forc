import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64

st.title('📈 Previsão automatizada de séries temporais')

"""
Este aplicativo de dados usa a biblioteca de código aberto do Facebook, Prophet, para gerar automaticamente valores de previsão futura a partir de um conjunto de dados importado.
Você poderá importar seus dados de um arquivo CSV e o separador precisa ser por ';', visualizar tendências e recursos, analisar o desempenho da previsão e, finalmente, baixar a previsão criada 😵
"""

"""
### Etapa 1: importar dados
"""
df = st.file_uploader('Importe o arquivo csv da série temporal aqui. As colunas devem ser rotuladas como ds e y. A entrada para o Prophet é sempre um dataframe com duas colunas: ds e y. A coluna ds (datestamp) deve ter um formato esperado pelo Pandas, idealmente AAAA-MM-DD para uma data ou AAAA-MM-DD HH:MM:SS para um carimbo de data/hora. A coluna y deve ser numérica e representa a medição que desejamos prever.', type='csv')

st.info(
            f"""
                👆 Carregue primeiro um arquivo .csv. Exemplo para experimentar:  [peyton_manning_wiki_ts.csv](https://raw.githubusercontent.com/zachrenwick/streamlit_forecasting_app/master/example_data/example_wp_log_peyton_manning.csv)
                """
        )

if df is not None:
    data = pd.read_csv(df,sep=';')
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 
    
    st.write(data)
    
    max_date = data['ds'].max()
    #st.write(max_date)

"""
### Etapa 2: selecione o horizonte de previsão

Lembre-se de que as previsões se tornam menos precisas com horizontes de previsão maiores.
"""

periods_input = st.number_input('Quantos períodos você gostaria de prever no futuro?',
min_value = 1, max_value = 365)

if df is not None:
    m = Prophet()
    m.fit(data)

"""
### Etapa 3: visualizar dados de previsão

O visual abaixo mostra valores futuros previstos. "yhat" é o valor previsto e os limites superior e inferior são (por padrão) intervalos de confiança de 80%.
"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
    """
    O próximo visual mostra os valores reais (pontos pretos) e previstos (linha azul) ao longo do tempo.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)

    """
    Os próximos visuais mostram uma tendência de alto nível de valores previstos, tendências de dia da semana e tendências anuais (se o conjunto de dados cobrir vários anos). A área sombreada em azul representa os intervalos de confiança superior e inferior.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)


"""
### Passo 4: Baixar os dados de previsão

O link abaixo permite que você baixe a previsão recém-criada para o seu computador para posterior análise e uso.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (clique com o botão direito do mouse e salvar link como ** <forecast_name>.csv**)'
    st.markdown(href, unsafe_allow_html=True)
