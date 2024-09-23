import pandas as pd  
from keras.models import load_model  
from pywt import wavedec  
from sklearn.decomposition import PCA  

# Load the trained LSTM model  
lstm_model = load_model('I:\Skripsi\Code\Tes2\lstm_model.h5')  
  
# Load the CSV file containing Indonesian companies  
indonesian_companies = pd.read_csv('indonesian_stock_data.csv')  
  
  
# Define a function to fetch historical data for a selected stock  
def fetch_historical_data(ticker, period, interval):  
   historical_data = pd.read_csv(f"{ticker}_{period}_{interval}.csv")  
   return historical_data  
  
  
# Define a function to generate stock predictions  
def generate_stock_predictions(historical_data):  
   # Preprocess the historical data using Wavelet transform and PCA  
   pca = PCA(n_components=6)  
   wavelet_data = []  
   for i in range(historical_data.shape[1]):  
      coeffs = wavedec(historical_data.iloc[:, i], 'db4', level=3)  
      wavelet_data.append(coeffs[0])  
   wavelet_df = pd.DataFrame(wavelet_data).T  
   pca_df = pca.fit_transform(wavelet_df)  
  
   # Make predictions using the LSTM model  
   predictions = lstm_model.predict(pca_df)  
   return predictions  
  
  
# Define a function to fetch company information  
def fetch_company_info(ticker):  
   selected_stock = indonesian_companies[indonesian_companies['Security Code'] == ticker]  
   company_info = {  
      'Security Code': ticker,  
      'Issuer Name': selected_stock['Issuer Name'].iloc[0],  
      'Security Name': selected_stock['Security Name'].iloc[0],  
      'Group': selected_stock['Group'].iloc[0],  
      'Face Value': selected_stock['Face Value'].iloc[0],  
      'Industry': selected_stock['Industry'].iloc[0],  
      'Sector Name': selected_stock['Sector Name'].iloc[0],  
      'Industry New Name': selected_stock['Industry New Name'].iloc[0],  
      'Igroup Name': selected_stock['Igroup Name'].iloc[0],  
      'ISubgroup Name': selected_stock['ISubgroup Name'].iloc[0]  
   }  
   return company_info