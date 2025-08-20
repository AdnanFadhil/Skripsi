import streamlit as st

page = 'Forecasting'
# Create a navigation sidebar
page = st.sidebar.selectbox('Choose a page:', ['Data', 'Forecasting','Revisi','Revisi2'])

# Load the selected page
if page == 'Data':
    # Import and run the Data page
    import Data
    Data.run()

elif page == 'Forecasting':
    # Import and run the Forecasting page
    import Forecasting
    Forecasting.run()

elif page == 'Revisi':
    # Import and run the Testing page
    import Forcast_revisi
    Forcast_revisi.run()

elif page == 'Revisi2':
    # Import and run the Testing page
    import Revisi2
    Revisi2.run()