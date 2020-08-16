import streamlit as st
from stonk_class import Stonk


# formatting the Streamlit webpage
st.title("Stonk Predictions, by Lana Elauria")
st.write("DISCLAIMER: This is a personal project for me, please don't blindly take these predictions at face value.")
st.write("Input the stock code you'd like to analyze and how many days into the future you'd like to look, "
         "and this app will give a general prediction of the stock movements!")
st.write("This app takes data from Yahoo! Finance.")


stonk_code= st.sidebar.text_input("What stock would you like to analyze? Default is AMC.", "AMC")
stonks = Stonk(stonk_code)
if st.sidebar.checkbox('Show stock data'):
    st.write(f'{stonk_code} stock data:\n', stonks.data)
stonks.split_scale()
days_ahead = st.sidebar.text_input("How many days into the future would you like to predict? Default is 30.", 30)
try:
    days_ahead = int(days_ahead)
except:
    raise Exception('You must input an integer.')
stonks.define_target(num_days=days_ahead)
stonks.build_LSTM(num_layers=2,epochs=5)
stonks.eval_LSTM()
st.write("The graph below is made to check the predictions of the neural network made by this app. "
         "Compare general trends, and decide for yourself if the app is accurate enough for your use.")
stonks.check_predictions()
st.pyplot()
stonks.predict()
st.pyplot()
st.write("NOTE: The y-axis for these graphs is the normalized stock price that the model works with. "
 "The stock market is extremely unpredictable, and this model won't be able to predict anything more than "
 "general trends in stock price movements. On the same token, be extremely wary of sharp spikes in the "
 "stock market predicted by this app. It extrapolates past behavior, and if there has been a spike recently, "
 "the model will believe there will be a spike later on as well, which is usually not the case.")
