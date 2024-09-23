import yfinance as yf
import pandas as pd

indonesian_companies = [
    {'ticker': 'ANTM.JK', 'name': 'Aneka Tambang Tbk'},
    {'ticker': 'BBNI.JK', 'name': 'Bank Negara Indonesia (Persero) Tbk'},
    {'ticker': 'INDF.JK', 'name': 'Indofood Sukses Makmur Tbk'},
    {'ticker': 'SMAR.JK', 'name': 'Sinar Mas Agro Resources and Technology Tbk'},
    {'ticker': 'TLKM.JK', 'name': 'Telekomunikasi Indonesia (Persero) Tbk'},
    {'ticker': 'BMRI.JK', 'name': 'Bank Mandiri (Persero) Tbk'},
    {'ticker': 'BBRI.JK', 'name': 'Bank Rakyat Indonesia (Persero) Tbk'},
    {'ticker': 'GGRM.JK', 'name': 'Gudang Garam Tbk'},
    {'ticker': 'HMSP.JK', 'name': 'H.M. Sampoerna Tbk'},
    {'ticker': 'INTP.JK', 'name': 'Indocement Tunggal Prakarsa Tbk'},
    {'ticker': 'KLBF.JK', 'name': 'Kalbe Farma Tbk'},
    {'ticker': 'PGAS.JK', 'name': 'Perusahaan Gas Negara (Persero) Tbk'},
    {'ticker': 'PPRO.JK', 'name': 'PP Properti Tbk'},
    {'ticker': 'PTBA.JK', 'name': 'Bukit Asam Tbk'},
    {'ticker': 'PWON.JK', 'name': 'Pakuwon Jati Tbk'},
    {'ticker': 'SCMA.JK', 'name': 'Surya Citra Media Tbk'},
    {'ticker': 'TINS.JK', 'name': 'Timah Tbk'},
    {'ticker': 'TKIM.JK', 'name': 'Pabrik Kertas Tjiwi Kimia Tbk'},
    {'ticker': 'UNTR.JK', 'name': 'United Tractors Tbk'},
    {'ticker': 'ADRO.JK', 'name': 'Adaro Energy Tbk'},
    {'ticker': 'AKRA.JK', 'name': 'AKR Corporindo Tbk'},
    {'ticker': 'ASII.JK', 'name': 'Astra International Tbk'},
    {'ticker': 'BBCA.JK', 'name': 'Bank Central Asia Tbk'},
    {'ticker': 'BDMN.JK', 'name': 'Bank Danamon Indonesia Tbk'},
    {'ticker': 'BJBR.JK', 'name': 'Bank Pembangunan Daerah Jawa Barat dan Banten Tbk'},
    {'ticker': 'BRIS.JK', 'name': 'Bank BRISyariah Tbk'},
    {'ticker': 'BSDE.JK', 'name': 'Bumi Serpong Damai Tbk'},
    {'ticker': 'CPIN.JK', 'name': 'Charoen Pokphand Indonesia Tbk'},
    {'ticker': 'ERAA.JK', 'name': 'Erajaya Swasembada Tbk'},
    {'ticker': 'EXCL.JK', 'name': 'XL Axiata Tbk'},
    {'ticker': 'ICBP.JK', 'name': 'Indofood CBP Sukses Makmur Tbk'},
    {'ticker': 'INKP.JK', 'name': 'Indah Kiat Pulp & Paper Tbk'},
    {'ticker': 'ITMG.JK', 'name': 'Indo Tambangraya Megah Tbk'},
    {'ticker': 'JSMR.JK', 'name': 'Jasa Marga Tbk'},
    {'ticker': 'LPKR.JK', 'name': 'Lippo Karawaci Tbk'},
    {'ticker': 'LPPF.JK', 'name': 'Matahari Department Store Tbk'},
    {'ticker': 'MEDC.JK', 'name': 'Medco Energi Internasional Tbk'},
    {'ticker': 'PNBN.JK', 'name': 'Bank Pan Indonesia Tbk'},
    {'ticker': 'PNLF.JK', 'name': 'Panin Life Tbk'},
    {'ticker': 'SMGR.JK', 'name': 'Semen Indonesia (Persero) Tbk'},
    {'ticker': 'TPIA.JK', 'name': 'Chandra Asri Petrochemical Tbk'},
    {'ticker': 'UNVR.JK', 'name': 'Unilever Indonesia Tbk'},
    {'ticker': 'WIKA.JK', 'name': 'Wijaya Karya (Persero) Tbk'}
]

stock_data_info = []  
  
for company in indonesian_companies:  
   ticker = company['ticker']  
   stock = yf.Ticker(ticker)  
   info = stock.info  
   stock_data_info.append({  
      'Security Code': ticker,  
      'Issuer Name': company['name'],  
      'Security Name': info['shortName'],  
      'Group': info['sector'],  
      'Face Value': info['currentPrice'],  
      'Industry': info['industry'],  
      'Sector Name': info['sector'],  
      'Industry New Name': info['industry'],  
      'Igroup Name': info['sector'],  
      'ISubgroup Name': info['industry']  
   })  
  
df = pd.DataFrame(stock_data_info)  
df.to_csv('indonesian_stock_data.csv', index=False)