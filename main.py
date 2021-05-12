import streamlit as st
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
	    content:'Developed by Hemand,Akshay and Amalu'; 
	    visibility: visible;
	    display: block;
	    position: relative;
	    #background-color: red;
	    padding: 5px;
	    top: 2px;
            }</style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
import datetime
from datetime import date
import pandas as pd
import numpy as np
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

st.title('Stock Forecast App')

st.sidebar.title('Training Data')
st.sidebar.subheader('Select details:')
start_date = st.sidebar.date_input("Start date",datetime.date(2015,1,1))
end_date = st.sidebar.date_input("End date",datetime.date(2021,1,1))

ticker_list = pd.read_csv('https://raw.githubusercontent.com/HemandMS/tickerlist/main/ticker_list.txt')
ticker_select = st.sidebar.selectbox('Stock ticker', ticker_list)
tickerData = yf.Ticker(ticker_select)
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)


n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(ticker_select)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
forecast.info()


# Show and plot forecast
#st.subheader('Forecast data')
#st.write(forecast.tail())

st.subheader(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.sidebar.title('Actual Data')
st.sidebar.subheader('Select details:')
real_date = st.sidebar.date_input("Actual data till:",datetime.date(2017,1,1))

@st.cache
def load_real_data(x):
    real_data = yf.download(x, start_date, real_date)
    real_data.reset_index(inplace=True)
    return real_data

real_data = load_real_data(ticker_select)

# Plot real data

submit = st.sidebar.button('Compare')

if submit:
    st.subheader('Actual data')
    #st.write(real_data.tail())
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=real_data['Date'], y=real_data['Open'], name="stock_open"))
    fig3.add_trace(go.Scatter(x=real_data['Date'], y=real_data['Close'], name="stock_close"))
    fig3.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


