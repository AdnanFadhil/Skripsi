# another_utils.py
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings("ignore") # Ignore some annoying warnings
from datetime import datetime
from PyEMD import EMD,EEMD,CEEMDAN #For module 'PyEMD', please use 'pip install EMD-signal' instead.
import joblib
# Import module for sample entropy
from sampen import sampen2
import plotly.graph_objects as go
from tuning.optuna_lstm_tuner import run_optuna_tuning,prepare_data_for_optuna,build_lstm_model,objective
import optuna


# Import modules for LSTM prediciton
# Sklearn
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn.metrics import r2_score # R2
from sklearn.metrics import mean_squared_error # MSE
from sklearn.metrics import mean_absolute_error # MAE
from sklearn.metrics import mean_absolute_percentage_error # MAPE
from tensorflow.keras.metrics import RootMeanSquaredError  # Correct import
# Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM,Bidirectional, Attention, Reshape, SpatialDropout1D, GaussianNoise,LeakyReLU, TimeDistributed, InputLayer
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, AdamW
#from tcn import TCN # pip install keras-tcn
from tensorflow.keras.utils import plot_model # To use plot_model, you need to install software graphviz
from tensorflow.python.client import device_lib

from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, Dropout, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.losses import Huber,LogCosh


# Define Paths
PATH = 'I:\\Streamlit_test\\'
FIGURE_PATH = PATH + 'figures\\'
LOG_PATH = PATH + 'subset\\'
MODEL_PATH = PATH + 'model\\'  # Added MODEL_PATH
DATASET_FOLDER = PATH + 'dataset'


# Set default values for the global variables
MODE = 'ceemdan'
FORM = ''
DATE_BACK = 30
PERIODS = 100
EPOCHS = 100
PATIENCE = 10
METHOD = 0  # or whatever value it should hold

# Declare LSTM model variables
# -------------------------------
# The units of LSTM layers and 3 LSTM layers will set to 4*CELLS, 2*CELLS, CELLS.
CELLS = 32
# Dropout rate of 3 Dropout layers
DROPOUT = 0.2 
# Adam optimizer loss such as 'mse','mae','mape','hinge' refer to https://keras.io/zh/losses/
OPTIMIZER_LOSS = 'mse'
# LSTM training batch_size for parallel computing, suggest 10-100
BATCH_SIZE = 16
# Proportion of validation set to training set, suggest 0-0.2
VALIDATION_SPLIT = 0.2
# Report of the training process, 0 not displayed, 1 detailed, 2 rough
VERBOSE = 1
# In the training process, whether to randomly disorder the training set
SHUFFLE = True

LSTM_MODEL = None 

# IMF Weight Configuration (add near other globals)
IMF_WEIGHTS = {
    'co-imf0': 0.65,  # High-frequency (reduced weight due to poor R²)
    'co-imf1': 0.30,  # Mid-frequency (your best performer)
    'co-imf2': 0.05   # Low-frequency
}


def declare_path(ticker, path=PATH, figure_path=FIGURE_PATH, log_path=LOG_PATH, model_path=MODEL_PATH, dataset_name=None):
    global PATH, FIGURE_PATH, LOG_PATH, MODEL_PATH, DATASET_FOLDER, DATASET_NAME, SERIES

    # Check inputs
    for x in ['path', 'figure_path', 'log_path', 'model_path']:
        if not isinstance(locals()[x], str):
            raise TypeError(f"{x} should be a string (e.g., 'D:\\CEEMDAN_LSTM\\...\\')")

    if not all([path, figure_path, log_path, model_path]):
        raise ValueError("PATH, FIGURE_PATH, LOG_PATH, and MODEL_PATH cannot be empty.")

    # Ensure base PATH exists
    PATH = path if path.endswith('\\') else path + '\\'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Adjust sub-paths
    FIGURE_PATH = figure_path if figure_path.endswith('\\') else figure_path + '\\'
    LOG_PATH = log_path if log_path.endswith('\\') else log_path + '\\'
    MODEL_PATH = model_path if model_path.endswith('\\') else model_path + '\\'

    # Create all necessary folders
    for p in [FIGURE_PATH, LOG_PATH, MODEL_PATH, DATASET_FOLDER]:
        os.makedirs(p, exist_ok=True)

    # Remove dot in ticker to match the saved filename
    base_ticker = (dataset_name or ticker).replace(".", "")
    DATASET_NAME = base_ticker

    # Load the dataset using dot-less ticker
    dataset_path = os.path.join(DATASET_FOLDER, f"{DATASET_NAME}.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Please provide a valid dataset_name.")

    print(f"Loading dataset: {DATASET_NAME}.csv")
    df_ETS = pd.read_csv(dataset_path, header=0)

    # Validate necessary columns
    if 'Date' not in df_ETS.columns or 'Close' not in df_ETS.columns:
        raise ValueError("Please ensure the dataset has 'Date' and 'Close' columns.")

    # Parse dates
    df_ETS['Date'] = pd.to_datetime(df_ETS['Date'])

    # Create time series
    SERIES = pd.Series(df_ETS['Close'].values, index=df_ETS['Date']).sort_index()

    # Save to demo_data
    SERIES.to_csv(os.path.join(PATH, 'demo_data.csv'))

    # Plot original series
    plt.figure(figsize=(10, 4))
    SERIES.plot(label='Original data', color='#0070C0')
    plt.title('Original Dataset')
    plt.xlabel('Time')
    plt.legend()
    plt.savefig(os.path.join(FIGURE_PATH, 'Original_Dataset.svg'), bbox_inches='tight')
    plt.show()

    return SERIES



# declare_vars function
def declare_vars(mode=MODE, form=FORM, data_back=DATE_BACK, periods=PERIODS, epochs=EPOCHS, patience=None):
    print('##################################')
    print('Global Variables')
    print('##################################')
    
    # Change and Check
    global MODE, FORM, DATE_BACK, PERIODS, EPOCHS, PATIENCE
    FORM = str(form)
    MODE = mode.lower()
    
    # Validate numeric inputs
    for var_name, var_value in zip(["DATE_BACK", "PERIODS", "EPOCHS"], [data_back, periods, epochs]):
        if not isinstance(var_value, int) or var_value <= 0:
            raise ValueError(f"{var_name} must be a positive integer.")
    
    DATE_BACK, PERIODS, EPOCHS = data_back, periods, epochs
    
    if patience is None:
        PATIENCE = max(1, int(EPOCHS / 10))  # Avoid zero patience
    elif not isinstance(patience, int) or patience < 0:
        raise ValueError("PATIENCE must be a non-negative integer.")
    else:
        PATIENCE = patience
    
    # Check Variables (Ensure check_vars exists)
    if 'check_vars' in globals():
        check_vars()
    else:
        print("Warning: check_vars() function not found.")

    # Show Variables
    print(f"MODE: {MODE.upper()}")
    print(f"FORM: {FORM}")
    print(f"DATE_BACK: {DATE_BACK}")
    print(f"PERIODS: {PERIODS}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"PATIENCE: {PATIENCE}")

    return MODE, FORM, DATE_BACK, PERIODS, EPOCHS, PATIENCE

# check_vars function
def check_vars():
    global FORM
    if MODE not in ['emd','eemd','ceemdan','emd_se','eemd_se','ceemdan_se']:
        raise TypeError('MODE should be emd, eemd, ceemdan, emd_se, eemd_se, or ceemdan_se rather than %s.' % str(MODE))
    if not isinstance(FORM, str):
        raise TypeError('FORM should be a string in digit such as "233" rather than %s.' % str(FORM))
    if not (isinstance(DATE_BACK, int) and DATE_BACK > 0):
        raise TypeError('DATE_BACK should be a positive integer rather than %s.' % str(DATE_BACK))
    if not (isinstance(PERIODS, int) and PERIODS >= 0):
        raise TypeError('PERIODS should be a positive integer rather than %s.' % str(PERIODS))
    if not (isinstance(EPOCHS, int) and EPOCHS > 0):
        raise TypeError('EPOCHS should be a positive integer rather than %s.' % str(EPOCHS))
    if not (isinstance(PATIENCE, int) and PATIENCE > 0):
        raise TypeError('PATIENCE should be a positive integer rather than %s.' % str(PATIENCE))
    if FORM == '' and (MODE in ['emd_se', 'eemd_se', 'ceemdan_se']):
        raise ValueError('FORM is not declared. Please declare it as form = "233".')

# check_dataset function
def check_dataset(dataset, input_form, no_se=False, use_series=False, uni_nor=False):
    file_name = ''
    # Change MODE
    global MODE
    if no_se:  # Change MODE to the MODE without '_se'
        check_vars()
        if MODE[-3:] == '_se':
            print('MODE is', str.upper(MODE), 'now, using %s instead.' % (str.upper(MODE[:-3])) )
            MODE = MODE[:-3]

    # Use SERIES as not dataset
    if use_series:
        if SERIES is None: 
            raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')

    # Check user input 
    if dataset is not None:  
        if input_form == 'series':
            if isinstance(dataset, pd.Series):  
                print('Get input pd.Series named:', str(dataset.name))
                input_dataset = dataset.copy(deep=True)
            else: 
                raise ValueError('The inputting series must be pd.Series rather than %s.' % type(dataset))
        
        elif input_form == 'df':
            if isinstance(dataset, pd.DataFrame): 
                print('Get input pd.DataFrame.')
                tmp_sum = None
                if 'sum' in dataset.columns:
                    tmp_sum = dataset['sum']
                    dataset = dataset.drop('sum', axis=1, inplace=False)
                if 'co-imf0' in dataset.columns: 
                    col_name = 'co-imf'
                else: 
                    col_name = 'imf'
                dataset.columns = [col_name + str(i) for i in range(len(dataset.columns))]  # change column names to imf0, imf1,...
                if tmp_sum is not None:  
                    dataset['sum'] = tmp_sum
                input_dataset = dataset.copy(deep=True)
            else: 
                raise ValueError('The inputting df must be pd.DataFrame rather than %s.' % type(dataset))
        else: 
            raise ValueError('Something wrong happened in module %s.' % __name__)

        file_name = ''
    
    else:  # Check default dataset and load
        if input_form == 'series':  # Check SERIES
            if not isinstance(SERIES, pd.Series): 
                raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
            else: 
                input_dataset = SERIES.copy(deep=True)
        
        elif input_form == 'df':
            check_vars()
            data_path = PATH + MODE + FORM + '_data.csv'
            if not os.path.exists(data_path):
                raise ImportError('Dataset %s does not exist in %s' % (data_path, PATH))
            else: 
                input_dataset = pd.read_csv(data_path, header=0, index_col=0)

    # Other warnings
    if METHOD == 0 and uni_nor: 
        print('Attention!!! METHOD = 0 means no using the unified normalization method. Declare METHOD by declare_uni_method(method=METHOD)')

    return input_dataset, file_name



# declare Method for unified normalization
def declare_uni_method(method=None):
    if method not in [0,1,2,3]: raise TypeError('METHOD should be 0,1,2,3.')
    global METHOD
    METHOD = method
    print('Unified normalization method (%d) is start using.'%method)


def wavelet_denoise(imf, wavelet='db4', level=1, thresholding='soft'):
    """
    Apply wavelet denoising to a given IMF.
    """
    coeffs = pywt.wavedec(imf, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Noise estimation
    uthresh = sigma * np.sqrt(2 * np.log(len(imf)))  # Universal threshold
    coeffs = [pywt.threshold(c, value=uthresh, mode=thresholding) for c in coeffs]
    return pywt.waverec(coeffs, wavelet)[:len(imf)]  # Ensure same length

def emd_decom_wavelet(series=None, trials=10, re_decom=False, re_imf=0, draw=True, base_ticker=None, compare_wavelet=True):
    dataset, file_name = check_dataset(series, input_form='series')
    series = dataset.values

    if base_ticker is None:
        base_ticker = file_name.replace(".", "")
    if not base_ticker:
        raise ValueError("Base ticker is not properly set.")

    print(f"Base Ticker: {base_ticker}")

    # Initialize decomposition
    print(f"{MODE.upper()} decomposition is running.")
    if MODE == 'emd':
        decom = EMD()
    elif MODE == 'eemd':
        decom = EEMD()
    elif MODE == 'ceemdan':
        decom = CEEMDAN()
    else:
        raise ValueError('MODE must be emd, eemd, or ceemdan.')

    decom.trials = trials
    imfs_emd = decom(series)
    imfs_num = imfs_emd.shape[0]

    if imfs_num < 8:
        print(f"Only {imfs_num} IMFs found, increasing trials.")
        decom.trials = 50
        imfs_emd = decom(series)
        imfs_num = imfs_emd.shape[0]

    # Save original IMFs for comparison
    raw_imfs = np.copy(imfs_emd)

    # Apply wavelet denoising
    imfs_denoised = np.copy(imfs_emd)
    for i in range(imfs_num):
        imfs_denoised[i] = wavelet_denoise(imfs_emd[i])

    # ✅ Optional comparison visualization
    if compare_wavelet:
        st.subheader("🔍 Analisis Pengaruh Wavelet pada IMF0 – IMF2")

        def entropy(arr):
            from scipy.stats import entropy as sp_entropy
            hist, _ = np.histogram(arr, bins=20, density=True)
            hist = hist[hist > 0]
            return sp_entropy(hist)

        summary_stats = []

        for imf_index in range(imfs_denoised.shape[0]):
            original = raw_imfs[imf_index]
            denoised = imfs_denoised[imf_index]

            # Plot
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(original, label=f'Original IMF{imf_index}', color='gray')
            ax.plot(denoised, label=f'Denoised IMF{imf_index}', color='orange')
            ax.set_title(f'IMF{imf_index}: Sebelum vs Sesudah Wavelet')
            ax.legend()
            st.pyplot(fig)

            # Stat
            summary_stats.append([
                f"IMF{imf_index}",
                np.mean(original),
                np.mean(denoised),
                np.std(original),
                np.std(denoised),
                entropy(original),
                entropy(denoised)
            ])

        # Tabel statistik
        stats_df = pd.DataFrame(summary_stats, columns=[
            "IMF", "Mean (Raw)", "Mean (Wavelet)", "Std (Raw)", "Std (Wavelet)", "Entropy (Raw)", "Entropy (Wavelet)"
        ])

        # ✅ FIX: Format hanya kolom numerik
        numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
        st.dataframe(stats_df.style.format({col: "{:.4f}" for col in numeric_cols}))



    # Plot IMFs if requested
    if draw:
        series_index = range(len(series))
        fig = plt.figure(figsize=(16, 2 * imfs_num))
        plt.subplot(1 + imfs_num, 1, 1)
        plt.plot(series_index, series, color='#0070C0')
        plt.ylabel('Original data')

        for i in range(imfs_num):
            plt.subplot(1 + imfs_num, 1, 2 + i)
            plt.plot(series_index, imfs_denoised[i], color='#F27F19')
            plt.ylabel(f'{MODE.upper()}-IMF{i}')

        image_name = f"CEEMDAN_{base_ticker}.svg"
        image_path = os.path.join(FIGURE_PATH, image_name)
        os.makedirs(FIGURE_PATH, exist_ok=True)

        fig.align_labels()
        plt.tight_layout()
        plt.savefig(image_path, bbox_inches='tight', format='svg')
        plt.show()

        print(f"Figure saved at: {image_path}")

    # Save to CSV
    imfs_df = pd.DataFrame(imfs_denoised.T)
    imfs_df.columns = ['imf' + str(i) for i in range(imfs_num)]
    if file_name == '' and not re_decom:
        imfs_df.to_csv(PATH + file_name + MODE + '_data.csv', index=False)
        print(f"{MODE.upper()} finished. Dataset saved at: {PATH + file_name + MODE + '_data.csv'}")

    return imfs_df


def integrate(df=None, inte_form=[[0, 1, 2], [3, 4], [5, 6, 7]], ticker="stock_data"):
    if type(inte_form) != list:
        raise ValueError('inte_form must be a list like [[0,1],[2,3,4],[5,6,7]].')

    check_list = sum(inte_form, [])
    if len(check_list) != len(set(check_list)):
        raise ValueError('inte_form has repeated IMFs. Please set it again.')

    df_emd = df
    if df_emd is None:
        raise ValueError("DataFrame (df) cannot be None.")

    if len(check_list) != len(df_emd.columns):
        raise ValueError(f'inte_form does not match the total number of IMFs ({len(df_emd.columns)})')

    # Integrate
    co_imfs = []
    num = len(inte_form)
    for i in range(num):
        imfs_to_integrate = [df_emd.columns[j] for j in inte_form[i]]
        co_imf = df_emd[imfs_to_integrate].sum(axis=1)
        co_imfs.append(co_imf)

    if not co_imfs:
        raise ValueError("Co-IMFs are empty, please check your integration process.")

    # Plotting
    fig, axes = plt.subplots(num + 1, 1, figsize=(16, 2 * (num + 1)))
    axes[0].plot(df_emd.index, df_emd.sum(axis=1), label='Original Data', color='#0070C0')
    axes[0].set_ylabel('Original Data')

    for i in range(num):
        axes[i + 1].plot(df_emd.index, co_imfs[i], label=f'co-imf{i}', color='#F27F19')
        axes[i + 1].set_ylabel(f'co-imf{i}')

    plt.tight_layout()
    fig_name = f"{ticker}_integration_result.svg"
    plt.savefig(os.path.join(FIGURE_PATH, fig_name), bbox_inches='tight')
    plt.close(fig)

    # Save Co-IMFs to CSV
    co_imfs_df = pd.DataFrame(co_imfs).T
    co_imfs_df.columns = [f'co-imf{i}' for i in range(num)]
    co_imfs_df.to_csv(os.path.join(FIGURE_PATH, f"{ticker}_co_imfs.csv"), index=False)

    # Display in Streamlit
    st.write("### Co-IMFs (co-imf0, co-imf1, co-imf2...)")
    st.dataframe(co_imfs_df)

    image_path = os.path.join(FIGURE_PATH, fig_name)
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Co-IMFs and Integration Results for {ticker}", use_column_width=True)
    else:
        st.error(f"Image not found at {image_path}")

    return co_imfs_df


    

def get_integration_form(num_imfs, verbose=False):
    """
    Groups any number of IMFs into exactly 3 frequency-based groups:
    - High frequency
    - Medium frequency
    - Low frequency

    Args:
        num_imfs (int): Total number of IMFs to group.
        verbose (bool): If True, prints the integration form.

    Returns:
        List[List[int]]: 3 groups of IMF indices.
    """
    if num_imfs < 3:
        return None  # Not enough IMFs to split into 3 groups

    # Calculate roughly equal group sizes
    base_size = num_imfs // 3
    remainder = num_imfs % 3

    # Distribute remainder across the first few groups
    sizes = [base_size + (1 if i < remainder else 0) for i in range(3)]

    # Build the index groups
    form = []
    start = 0
    for size in sizes:
        form.append(list(range(start, start + size)))
        start += size

    if verbose:
        print(f"Integration form for {num_imfs} IMFs: {form}")

    return form


class HiddenPrints: # used to hide the print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def create_dateback(df,uni=False,ahead=1):
    # Normalize for DataFrame
    if uni and METHOD != 0 and ahead == 1: # Unified normalization
        # Check input and load dataset
        if SERIES is None: raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
        if MODE not in ['emd','eemd','ceemdan']: raise ValueError('MODE must be emd, eemd, ceemdan if you want to try unified normalization method.')
        if not (os.path.exists(PATH+MODE+'_data.csv')): raise ImportError('Dataset %s does not exist in '%(PATH+MODE+'_data.csv'),PATH)
      
        # Load data
        df_emd = pd.read_csv(PATH+MODE+'_data.csv',header=0,index_col=0)
        # Method (1)
        print('##################################')
        if METHOD == 1:
            scalar,min0 = SERIES.max()-SERIES.min(),0 
            print('Unified normalization Method (1):')
        # Method (2)
        elif METHOD == 2:
            scalar,min0 = df_emd.max().max()-df_emd.min().min(),df_emd.min().min()
            print('Unified normalization Method (2):')
        # Method (3)
        elif METHOD == 3:
            scalar,min0 = SERIES.max()-df_emd.min().min(),df_emd.min().min()
            print('Unified normalization Method (3):')

        # Normalize
        df = (df-min0)/scalar
        scalarY = {'scalar':scalar,'min':min0}
        print(df)
        if isinstance(df, pd.DataFrame):
            trainX = df.drop('sum', axis=1, inplace=False)
            trainY = np.array(df['sum']).reshape(-1, 1)
        # Normalize for each IMF in Series
        else:
            trainY = np.array(df.values).reshape(-1, 1)
            trainX = trainY
    else:
        # Normalize without unifying
        if isinstance(df, pd.DataFrame):
            trainX = df.drop('sum', axis=1, inplace=False)
            scalarX = MinMaxScaler(feature_range=(0,1))#sklearn normalize
            trainX = scalarX.fit_transform(trainX)
            trainY = np.array(df['sum']).reshape(-1, 1)
            scalarY = MinMaxScaler(feature_range=(0,1))#sklearn normalize
            trainY = scalarY.fit_transform(trainY)
        # Normalize for each IMF in Series
        else:
            trainY = np.array(df.values).reshape(-1, 1)
            scalarY = MinMaxScaler(feature_range=(0,1))#sklearn normalize
            trainY = scalarY.fit_transform(trainY)
            trainX = trainY
    
    # Create dateback
    dataX, dataY = [], []
    ahead = ahead - 1
    for i in range(len(trainY)-DATE_BACK-ahead):
        dataX.append(np.array(trainX[i:(i+DATE_BACK)]))
        dataY.append(np.array(trainY[i+DATE_BACK+ahead]))
    return np.array(dataX),np.array(dataY),scalarY,np.array(trainX[-DATE_BACK:])

def LSTM_pred(data=None, draw=True, uni=False, show_model=True, train_set=None,
              next_pred=False, ahead=1, progress_callback=None,
              pretrained=False, model_path=None):

    import pickle, json
    from tensorflow.keras.models import load_model

    if train_set is None:
        trainX, trainY, scalarY, next_trainX = create_dateback(data, uni=uni, ahead=ahead)
    else:
        trainX, trainY, scalarY, next_trainX = train_set

    if uni and next_pred:
        raise ValueError("Next pred does not support unified normalization.")

    if PERIODS == 0:
        train_X = trainX
        y_train = trainY
    else:
        x_train, x_test = trainX[:-PERIODS], trainX[-PERIODS:]
        y_train, y_test = trainY[:-PERIODS], trainY[-PERIODS:]
        train_X = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        test_X = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    print(f"\nInput Shape: ({train_X.shape[1]}, {train_X.shape[2]})\n")

    if pretrained:
        # Load model from .h5
        model = load_model(model_path)

        # Load and show model config
        config_path = model_path.replace(".h5", "_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config = json.load(f)
            print("✅ Loaded model architecture config:")
            for i, layer in enumerate(model_config["config"]["layers"]):
                print(f"  Layer {i + 1}: {layer['class_name']} - {layer['config'].get('units', 'n/a')}")
        else:
            print("⚠️ Config not found.")

        # Optional: Load best_params
        param_path = model_path.replace(".h5", "_params.json")
        if os.path.exists(param_path):
            with open(param_path, "r") as f:
                best_params = json.load(f)
            print("✅ Loaded tuning params:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
        else:
            print("ℹ️ No tuning params found.")

    else:
        # Build and train model
        model = LSTM_model(train_X.shape)

        EarlyStop = EarlyStopping(monitor="val_loss", patience=5 * PATIENCE, verbose=VERBOSE, mode="auto")
        Reduce = ReduceLROnPlateau(monitor="val_loss", patience=PATIENCE, verbose=VERBOSE, mode="auto")

        class StreamlitEpochCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if progress_callback:
                    progress_callback(epoch + 1, logs)

        history = model.fit(
            train_X,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=VERBOSE,
            shuffle=SHUFFLE,
            callbacks=[EarlyStop, Reduce, StreamlitEpochCallback()],
        )

        # Save model
        model.save(model_path)

        # Save config
        config_path = model_path.replace(".h5", "_config.json")
        with open(config_path, "w") as f:
            f.write(model.to_json())

        # Optional: Save best_params (if present)
        if "best_params" in locals():
            param_path = model_path.replace(".h5", "_params.json")
            with open(param_path, "w") as f:
                json.dump(best_params, f, indent=4)

        # Save training history
        history_path = model_path.replace(".h5", "_history.pkl")
        with open(history_path, "wb") as f:
            pickle.dump(history.history, f)

    # Predict
    if PERIODS != 0:
        pred_test = model.predict(test_X)
        evl(y_test, pred_test)
    else:
        pred_test = np.array([])

    if next_pred:
        next_ans = model.predict(next_trainX.reshape((1, trainX.shape[1], trainX.shape[2])))
        pred_test = np.append(pred_test, next_ans)

    pred_test = pred_test.ravel().reshape(-1, 1)

    if isinstance(scalarY, MinMaxScaler):
        test_pred = scalarY.inverse_transform(pred_test)
        if PERIODS != 0 and 'y_test' in locals():
            test_y = scalarY.inverse_transform(y_test)
    else:
        test_pred = pred_test * scalarY["scalar"] + scalarY["min"]
        if PERIODS != 0 and 'y_test' in locals():
            test_y = y_test * scalarY["scalar"] + scalarY["min"]

    # === Visualization ===
    fig_name = data.name if isinstance(data, pd.Series) and str(data.name) != "None" else "DataFrame"

    # Loss from current training
    if draw and not pretrained and PERIODS != 0 and 'history' in locals():
        plt.figure(figsize=(5, 2))
        plt.plot(history.history["loss"], label="training loss")
        plt.plot(history.history["val_loss"], label="validation loss")
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.legend()
        plt.title(fig_name + " LSTM loss chart")
        st.pyplot(plt)

    # Loss from pretrained history
    elif draw and pretrained and PERIODS != 0:
        history_path = model_path.replace(".h5", "_history.pkl")
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                history_dict = pickle.load(f)
            plt.figure(figsize=(5, 2))
            plt.plot(history_dict["loss"], label="training loss")
            plt.plot(history_dict["val_loss"], label="validation loss")
            plt.ylabel("loss")
            plt.xlabel("epochs")
            plt.legend()
            plt.title(fig_name + " Pretrained LSTM loss chart")
            st.pyplot(plt)

    # Forecast visualization
    if draw and PERIODS != 0:
        plt.figure(figsize=(5, 2))
        plt.plot(test_y, label="True values")
        plt.plot(test_pred, label="Predicted values")
        plt.title(fig_name + " LSTM forecasting result")
        plt.legend()
        st.pyplot(plt)

    return test_pred, model, scalarY



# Declare LSTM variables
def declare_LSTM_vars(cells=CELLS,dropout=DROPOUT,optimizer_loss=OPTIMIZER_LOSS,batch_size=BATCH_SIZE,validation_split=VALIDATION_SPLIT,verbose=VERBOSE,shuffle=SHUFFLE):
    print('##################################')
    print('LSTM Model Variables')
    print('##################################')
    PATIENCE
    # Changepatience=
    global CELLS,DROPOUT,OPTIMIZER_LOSS,BATCH_SIZE,VALIDATION_SPLIT,VERBOSE,SHUFFLE
    CELLS,DROPOUT,OPTIMIZER_LOSS = cells,dropout,optimizer_loss
    BATCH_SIZE,VALIDATION_SPLIT,VERBOSE,SHUFFLE = batch_size,validation_split,verbose,shuffle

    # Check
    if not (type(CELLS) == int and CELLS>0): raise TypeError('CELLS should a positive integer.')
    if not (type(DROPOUT) == float and DROPOUT>0 and DROPOUT<1): raise TypeError('DROPOUT should a number between 0 and 1.')
    if not (type(BATCH_SIZE) == int and BATCH_SIZE>0):
        raise TypeError('BATCH_SIZE should be a positive integer.')
    if not (type(VALIDATION_SPLIT) == float and VALIDATION_SPLIT>0 and VALIDATION_SPLIT<1):
        raise TypeError('VALIDATION_SPLIT should be a number best between 0.1 and 0.4.')
    if VERBOSE not in [0,1,2]:
        raise TypeError('VERBOSE should be 0, 1, or 2. The detail level of the training message')
    if type(SHUFFLE) != bool:
        raise TypeError('SHUFFLE should be True or False.')
    
    # Show
    print('CELLS:'+str(CELLS))
    print('DROPOUT:'+str(DROPOUT))
    print('OPTIMIZER_LOSS:'+str(OPTIMIZER_LOSS))
    print('BATCH_SIZE:'+str(BATCH_SIZE))
    print('VALIDATION_SPLIT:'+str(VALIDATION_SPLIT))
    print('VERBOSE:'+str(VERBOSE))
    print('SHUFFLE:'+str(SHUFFLE))

LSTM_MODEL = None 

# Change Kreas model
def declare_LSTM_MODEL(model=LSTM_MODEL):
    print("LSTM_MODEL has changed to be %s and start your forecast."%model)
    global LSTM_MODEL
    LSTM_MODEL = model


def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
    Args:
        y_true: array-like of true values
        y_pred: array-like of predicted values
    Returns:
        sMAPE value
    """
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    denominator = (np.abs(y_true) + np.abs(y_pred))
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / denominator)

def evl(y_test, y_pred, scale='Close', verbose=True): 
    # Convert to 1D
    y_test = np.array(y_test).ravel()
    y_pred = np.array(y_pred).ravel()

    if len(y_test) != len(y_pred):
        raise ValueError("Length mismatch between y_test and y_pred")

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    
    # Handle zero-division in MAPE
    non_zero_idx = y_test != 0
    if not np.any(non_zero_idx):
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_test[non_zero_idx] - y_pred[non_zero_idx]) / y_test[non_zero_idx])) * 100
    
    # Calculate sMAPE
    smape_val = smape(y_test, y_pred)

    if verbose:
        print('##################################')
        print(f'Model Evaluation with scale of {scale}')
        print('##################################')
        print(f'R2:   {r2:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE:  {mae:.4f}')
        print(f'MAPE: {mape:.2f}%')
        print(f'sMAPE: {smape_val:.2f}%')  # New sMAPE output

    # Only show in Streamlit if scale is 'Close'
    if scale.lower() == 'close':
        st.subheader(f"Model Evaluation (scale: {scale})")
        st.write(f'**R²**: {r2:.4f}')
        st.write(f'**RMSE**: {rmse:.4f}')
        st.write(f'**MAE**: {mae:.4f}')
        st.write(f'**MAPE**: {mape:.2f}%')
        st.write(f'**sMAPE**: {smape_val:.2f}%')  # New sMAPE display

    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'sMAPE': smape_val}




def Respective_LSTM(df=None, draw=True, uni=False, show_model=True, next_pred=False, ahead=1):
    st.write('## Respective LSTM Forecasting')
    print('==============================================================================================')
    print('This is Respective LSTM Forecasting running...')
    print('==============================================================================================')

    input_df, file_name = check_dataset(df, input_form='df', use_series=True, uni_nor=uni)
    data_pred = []
    print('Part of Inputting dataset:')
    print(input_df)

    start = time.time()
    col_name = 'co-imf' if MODE[-3:] == '_se' else 'imf'
    df_len = len(input_df.columns)
    if 'sum' in input_df.columns:
        df_len -= 1

    for i in range(df_len):
        imf_name = col_name + str(i)
        print('==============================================================================================')
        print(str.upper(MODE) + '--' + imf_name)
        print('==============================================================================================')

        epoch_placeholder = st.empty()
        progress_epoch = st.progress(0.0)

        def epoch_update_callback(epoch, logs):
            epoch_placeholder.text(f"{imf_name} - Epoch {epoch}/{EPOCHS} - Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")
            progress_epoch.progress(epoch / EPOCHS)

        test_pred, _, scalarY = LSTM_pred_Testing(
            data=input_df[imf_name],
            draw=draw,
            uni=uni,
            show_model=show_model,
            next_pred=next_pred,
            ahead=ahead,
            model_fn=lambda shape: model,
            pretrained=True  # ✅ <-- ADD THIS
        )
        # ==== Save model ====
        model_save_path = os.path.join(MODEL_PATH, f"{base_ticker}_{imf_name}_model.h5")
        model.save(model_save_path)
        print(f"✅ Model saved at {model_save_path}")


        progress_epoch.empty()
        epoch_placeholder.text(f"✅ {imf_name} training finished.")
        data_pred.append(test_pred.ravel())

    end = time.time()

    df_pred = pd.DataFrame(data_pred).T
    df_pred.columns = [col_name + str(i) for i in range(len(df_pred.columns))]
    pd.DataFrame.to_csv(df_pred, LOG_PATH + file_name + 'respective_' + MODE + FORM + '_pred.csv')

    if PERIODS != 0:
        res_pred = df_pred.T.sum()
        if draw and file_name == '':
            plot_all('Respective', res_pred[:PERIODS])
        if file_name == '':
            input_df['sum'] = SERIES.values
        elif 'sum' not in input_df.columns:
            input_df['sum'] = input_df.T.sum().values
        df_evl = evl(input_df['sum'][-PERIODS:].values, res_pred[:PERIODS], scale='input df')
        print('Running time: %.3fs' % (end - start))
        df_evl.append(end - start)
        df_evl = pd.DataFrame(df_evl).T
        if next_pred:
            print('##################################')
            print('Today is', input_df['sum'][-1:].values, 'but predict as', res_pred[-2:-1].values)
            print('Next day is', res_pred[-1:].values)
        pd.DataFrame.to_csv(df_evl, LOG_PATH + file_name + 'respective_' + MODE + FORM + '_log.csv', index=False, header=0, mode='a')
        print('Respective LSTM Forecasting finished, check the logs', LOG_PATH + file_name + 'respective_' + MODE + FORM + '_log.csv')

    return df_pred, model, scalarY

def load_and_preprocess_data(csv_path, scalarY=None, save_global_scaler=True, scaler_save_path=None):
    df = pd.read_csv(csv_path)

    if 'Close' not in df.columns:
        raise ValueError("CSV must contain 'Close' column.")

    close_values = df[['Close']].values

    if scalarY is None:
        scalarY = MinMaxScaler()
        close_scaled = scalarY.fit_transform(close_values)

        # ✅ Save global Close scaler
        if save_global_scaler and scaler_save_path:
            scaler_path = os.path.join(scaler_save_path, 'scalarY_global.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scalarY, f)
    else:
        # ✅ If scalarY is a dict of scalers (e.g., {'co-imf0': ..., 'co-imf2': ...}), use co-imf2
        if isinstance(scalarY, dict):
            scalarY = scalarY.get('co-imf2', list(scalarY.values())[0])  # fallback to any if co-imf2 not found

        if not isinstance(scalarY, MinMaxScaler):
            raise ValueError("Invalid scalarY: must be MinMaxScaler or a dict with 'min' and 'scalar'.")

        close_scaled = scalarY.transform(close_values)

    df['Close_scaled'] = close_scaled
    return df, close_scaled, scalarY

def predict_future(model, last_data, days, scalerY, window_size=60):
    import numpy as np

    IMF_WEIGHTS = {
        'co-imf0': 0.65,  # High-frequency
        'co-imf1': 0.30,  # Mid-frequency
        'co-imf2': 0.05   # Low-frequency
    }

    predictions = []
    current_input = last_data[-window_size:].copy()

    for _ in range(days):
        X_input = np.array(current_input).reshape(1, window_size, -1)

        if isinstance(model, dict):
            # Weighted prediction from each IMF model
            next_pred = 0.0
            for imf_name, m in model.items():
                weight = IMF_WEIGHTS.get(imf_name, 0.0)
                next_pred += weight * m.predict(X_input)[0][0]
        else:
            next_pred = model.predict(X_input)[0][0]

        predictions.append(next_pred)

        # Update input with the new prediction (repeated across features)
        current_input = np.vstack([current_input[1:], [[next_pred]*current_input.shape[1]]])

    # ✅ Inverse transform if scaler is available
    if scalerY is not None:
        if isinstance(scalerY, dict):
            scalerY = scalerY.get('co-imf2', list(scalerY.values())[0])

        if hasattr(scalerY, 'inverse_transform'):
            predictions = scalerY.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()

    return predictions



# Testingggg!!!!!!!!!!!!!
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dropout, Bidirectional, LSTM,
    Attention, Dense, LayerNormalization
)
from tensorflow.keras.optimizers import Adam

def LSTM_model_Testing(shape):
    print("🔹 Building CNN + BiLSTM + Attention Model")

    inputs = Input(shape=(shape[1], shape[2]))

    # 1D Convolution to extract high-frequency/local features
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(0.2)(x)

    # Bidirectional LSTM for sequence modeling
    x = Bidirectional(LSTM(CELLS * 2, activation='relu', return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Attention Layer
    attn_out = Attention()([x, x])  # query = value = x
    attn_out = LayerNormalization()(attn_out)  # normalize after attention

    # More LSTM layers after attention (optional, for deeper learning)
    x = LSTM(CELLS, activation='relu', return_sequences=False)(attn_out)
    x = Dropout(0.3)(x)

    # Final output layer
    output = Dense(1, activation='tanh')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=OPTIMIZER_LOSS, optimizer=Adam(learning_rate=0.001))

    return model

def LSTM_model_HighFreq(shape):
    model = Sequential([
        GaussianNoise(0.002, input_shape=(shape[1], shape[2])),  # Light noise
        LSTM(64, activation='tanh',  # Increased capacity
             kernel_regularizer=l1_l2(0.001, 0.001),  # Minimal regularization
             return_sequences=False),
        Dropout(0.1),  # Minimal dropout
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(
        loss='mae',
        optimizer=Adam(learning_rate=0.01)  # Higher LR for volatility
    )
    return model

def LSTM_model_MidFreq(shape):
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, 
             input_shape=(shape[1], shape[2])),
        Dropout(0.3),
        LSTM(32, activation='tanh'),
        Dense(32, activation='swish'),  # Swish > ReLU for cycles
        Dense(1, activation='linear')
    ])
    model.compile(
        loss=Huber(delta=0.5),  # Tolerant to outliers
        optimizer=Adam(learning_rate=0.0005),
        metrics=['mse']
    )
    return model


def LSTM_model_LowFreq(shape):
    model = Sequential([
        # GRU often better for long trends
        GRU(64, return_sequences=True, 
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer='orthogonal'),
        
        GRU(32),
        
        # Trend-specific processing
        Dense(32, activation='elu'),
        Dense(1, activation='linear')
    ])
    
    optimizer = AdamW(
        learning_rate=0.0001,  # Lower for trends
        weight_decay=0.001
    )
    
    model.compile(
        loss='huber',  # Revert to reliable Huber
        optimizer=optimizer,
        metrics=[RootMeanSquaredError()]
    )
    return model

def LSTM_pred_Testing(data=None, draw=True, uni=False, show_model=True, train_set=None,
                      next_pred=False, ahead=1, progress_callback=None, model_fn=None,
                      frequency_type='medium', pretrained=False, scalarY=None):  
    if train_set is None:
        trainX, trainY, scalarY, next_trainX = create_dateback(data, uni=uni, ahead=ahead)
    else:
        trainX, trainY, scalarY, next_trainX = train_set

    if uni and next_pred:
        raise ValueError("Next pred does not support unified normalization.")

    if PERIODS == 0:
        train_X = trainX
        y_train = trainY
    else:
        x_train, x_test = trainX[:-PERIODS], trainX[-PERIODS:]
        y_train, y_test = trainY[:-PERIODS], trainY[-PERIODS:]
        train_X = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        test_X = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    print(f"\nInput Shape: ({train_X.shape[1]}, {train_X.shape[2]})\n")
    model = model_fn(train_X.shape)

    # Frequency type inference
    if model_fn is not None and hasattr(model_fn, '__name__'):
        name = model_fn.__name__.lower()
        if name == 'lstm_model_highfreq':
            frequency_type = 'high'
        elif name == 'lstm_model_midfreq':
            frequency_type = 'medium'
        elif name == 'lstm_model_lowfreq':
            frequency_type = 'low'

    class StreamlitEpochCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, logs)

    def get_training_config(frequency_type):
        config = {
            'high': {
                'early_stop_patience': 20,
                'reduce_patience': 5,
                'min_delta': 0.001,
                'min_lr': 1e-6,
                'batch_size': 64,
                'shuffle': True
            },
            'medium': {
                'early_stop_patience': 30,
                'reduce_patience': 15,
                'min_delta': 0.0005,
                'min_lr': 5e-7,
                'batch_size': 32,
                'shuffle': True
            },
            'low': {
                'early_stop_patience': 25,
                'reduce_patience': 10,
                'min_delta': 0.0001,
                'min_lr': 1e-8,
                'batch_size': 8,
                'shuffle': False
            }
        }
        return config[frequency_type]

    def train_model(model, train_X, y_train, frequency_type='medium'):
        config = get_training_config(frequency_type)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=config['early_stop_patience'],
                min_delta=config['min_delta'],
                restore_best_weights=True,
                verbose=VERBOSE
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                patience=config['reduce_patience'],
                factor=0.1 if frequency_type == 'low' else 0.5,
                min_lr=config['min_lr'],
                verbose=VERBOSE
            ),
            StreamlitEpochCallback()
        ]

        history = model.fit(
            train_X,
            y_train,
            epochs=EPOCHS,
            batch_size=config['batch_size'],
            validation_split=VALIDATION_SPLIT,
            verbose=VERBOSE,
            shuffle=config['shuffle'],
            callbacks=callbacks,
        )
        return history

    # ===================== 🚫 TRAINING SKIPPED IF PRETRAINED =====================
    if not pretrained:
        history = train_model(model, train_X, y_train, frequency_type=frequency_type)
    else:
        history = None
        print("✅ Skipping training. Using pre-trained model.")

    if PERIODS != 0:
        pred_test = model.predict(test_X)
        test_y = y_test  # ✅ Selalu definisikan test_y untuk scaling dan plotting
        if not pretrained:
            evl(test_y, pred_test)
    else:
        pred_test = np.array([])


    if next_pred:
        next_ans = model.predict(next_trainX.reshape((1, trainX.shape[1], trainX.shape[2])))
        pred_test = np.append(pred_test, next_ans)

    pred_test = pred_test.ravel().reshape(-1, 1)

    # Debug scaling
    print("\nScaling Debug:")
    print(f"Min/Max before scaling: {pred_test.min()}, {pred_test.max()}")

    if isinstance(scalarY, MinMaxScaler):
        test_pred = scalarY.inverse_transform(pred_test)
        if PERIODS != 0:
            test_y = scalarY.inverse_transform(test_y)
    else:
        test_pred = pred_test * scalarY["scalar"] + scalarY["min"]
        if PERIODS != 0:
            test_y = test_y * scalarY["scalar"] + scalarY["min"]

    print(f"Min/Max after inverse: {test_pred.min()}, {test_pred.max()}\n")

    if draw and PERIODS != 0:
        fig_name = data.name if isinstance(data, pd.Series) and str(data.name) != "None" else "DataFrame"

        if history:
            plt.figure(figsize=(5, 2))
            plt.plot(history.history["loss"], label="training loss")
            plt.plot(history.history["val_loss"], label="validation loss")
            plt.ylabel("loss")
            plt.xlabel("epochs")
            plt.legend()
            plt.title(fig_name + " LSTM loss chart")
            st.pyplot(plt)

        plt.figure(figsize=(5, 2))
        plt.plot(test_y, label="True values")
        plt.plot(test_pred, label="Predicted values")
        plt.title(fig_name + " LSTM forecasting result")
        plt.legend()
        st.pyplot(plt)

    return test_pred, model, scalarY

def Respective_LSTM_Testing(df=None, draw=True, uni=False, show_model=True, 
                            next_pred=False, ahead=1, base_ticker=None,
                            use_saved=False):
    """
    Enhanced LSTM testing with:
    - Auto-load trained models if available
    - Weighted ensemble prediction
    - Complete visualization & evaluation
    - Auto-tuning if model not yet saved
    """


    st.write('## Respective LSTM Forecasting')
    print('=' * 80)
    print(f"Starting forecasting with weights: {IMF_WEIGHTS}")

    input_df, file_name = check_dataset(df, input_form='df', use_series=True, uni_nor=uni)

    available_imfs = [col for col in input_df.columns if col.startswith('co-imf')]
    missing_weights = [imf for imf in available_imfs if imf not in IMF_WEIGHTS]
    if missing_weights:
        raise ValueError(f"IMF WEIGHT CONFIGURATION ERROR: Missing weights for: {missing_weights}")

    df_prediction = pd.DataFrame()
    model_dict = {}
    scalarY_dict = {}
    config_display = {}  # <- To store all configs for summary

    for imf_name in available_imfs:
        model_path = os.path.join(MODEL_PATH, f"{base_ticker}_{imf_name}_model.h5")
        scaler_path = os.path.join(MODEL_PATH, f"{base_ticker}_{imf_name}_scaler.pkl")
        param_path = os.path.join(MODEL_PATH, base_ticker, imf_name, 'best_lstm_params.txt')

        model_exists = os.path.exists(model_path) and os.path.exists(scaler_path)

        if model_exists:
            st.info(f"📥 Loading saved model for {imf_name}")

            # Load full model
            model = tf.keras.models.load_model(model_path, compile=False)

            # Load scaler
            with open(scaler_path, "rb") as f:
                scalarY = pickle.load(f)

            # Load parameters if available
            best_params = {}
            if os.path.exists(param_path):
                best_params = load_best_params(param_path)
                config_display[imf_name] = best_params  # for later display

                # Show params inline (not inside nested expander)
                st.markdown(f"### 🔧 Model Configuration for `{imf_name}`")
                for k, v in best_params.items():
                    st.markdown(f"- **{k}**: `{v}`")

            # Compile with params or default
            model.compile(
                optimizer=Adam(learning_rate=best_params.get("lr", 0.001)),
                loss=best_params.get("loss_fn", "mse")
            )

            test_pred, _, _ = LSTM_pred_Testing(
                data=input_df[imf_name],
                draw=draw,
                uni=uni,
                show_model=show_model,
                next_pred=next_pred,
                ahead=ahead,
                model_fn=lambda shape: model,
                pretrained=True,
                scalarY=scalarY
            )
        else:
            st.warning(f"🔧 Tuning and training model for {imf_name}")
            X_train, y_train, X_val, y_val, input_shape = prepare_data_for_optuna(input_df[imf_name])

            best_params, _ = run_optuna_tuning(
                X_train, y_train, X_val, y_val,
                input_shape=input_shape[1:],
                output_dir=os.path.join(MODEL_PATH, base_ticker, imf_name),
                n_trials=20
            )
            config_display[imf_name] = best_params

            def tuned_model_fn(shape):
                model, _ = build_lstm_model(shape[1:], trial=None, fixed_params=best_params)
                return model

            test_pred, model, scalarY = LSTM_pred_Testing(
                data=input_df[imf_name],
                draw=draw,
                uni=uni,
                show_model=show_model,
                next_pred=next_pred,
                ahead=ahead,
                model_fn=tuned_model_fn,
                pretrained=False
            )

            model.save(model_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(scalarY, f)

            st.success(f"✅ Saved trained model and scaler to: {model_path}, {scaler_path}")

        model_dict[imf_name] = model
        scalarY_dict[imf_name] = scalarY
        df_prediction[imf_name] = test_pred.ravel()

    # ===== ENSEMBLE & WEIGHTED COMBINATION =====
    res_pred = pd.Series(np.zeros(len(df_prediction)), index=df_prediction.index)
    for imf_name, weight in IMF_WEIGHTS.items():
        imf_values = df_prediction[imf_name].values.reshape(-1, 1)
        res_pred += imf_values.ravel() * weight

    # ===== INVERSE SCALING =====
    scalarY = list(scalarY_dict.values())[0]  # pick first
    if isinstance(scalarY, MinMaxScaler):
        res_pred = scalarY.inverse_transform(np.array(res_pred).reshape(-1, 1)).ravel()
    else:
        res_pred = res_pred * scalarY["scalar"] + scalarY["min"]

    # ===== EVALUATION =====
    df_evl = None
    if PERIODS > 0:
        try:
            true_df = pd.read_csv(os.path.join(DATASET_FOLDER, f"{base_ticker}.csv"))
            input_df['sum'] = true_df['Close'].values[-len(input_df):]
        except Exception as e:
            print(f"[WARN] Failed to load Close prices for {base_ticker}, falling back to SERIES: {e}")
            input_df['sum'] = SERIES.values[-len(input_df):]

        actual_values = input_df['sum'][-PERIODS:].values.reshape(-1, 1)
        if isinstance(scalarY, MinMaxScaler):
            actual_values = scalarY.inverse_transform(actual_values).ravel()
        else:
            actual_values = actual_values * scalarY["scalar"] + scalarY["min"]

        df_evl = evl(actual_values, res_pred[:PERIODS], scale='original')

        pred_path = f"{LOG_PATH}respective_{MODE}_{base_ticker}_pred.csv"
        df_prediction.to_csv(pred_path)

        log_path = f"{LOG_PATH}respective_{MODE}_{base_ticker}_log.csv"
        pd.DataFrame([df_evl]).to_csv(log_path, index=False, mode='a')

    # ===== VISUALIZATION =====
    if draw and PERIODS > 0 and df_evl is not None:
        fig, ax = plt.subplots(figsize=(12, 4))
        for imf in df_prediction.columns:
            ax.plot(df_prediction[imf][:PERIODS], label=f"{imf} ({IMF_WEIGHTS[imf]*100:.1f}%)", alpha=0.7)
        ax.set_title("Individual IMF Predictions")
        ax.legend()
        st.pyplot(fig)

    # ===== GLOBAL CONFIG DISPLAY =====
    st.subheader("📋 All Model Configurations Summary")
    for imf, params in config_display.items():
        st.markdown(f"#### ⚙️ `{imf}`")
        for k, v in params.items():
            st.markdown(f"- **{k}**: `{v}`")

    return df_prediction, model_dict, scalarY_dict




def plot_direction_accuracy(actual_prices, predicted_prices):
    """
    Plot histogram showing directional accuracy of predictions
    Args:
        actual_prices: array of actual price values
        predicted_prices: array of predicted price values
    """
    # Calculate daily returns (direction)
    actual_returns = np.diff(actual_prices)
    pred_returns = np.diff(predicted_prices)
    
    # Create directional labels
    actual_directions = ['Up' if x >= 0 else 'Down' for x in actual_returns]
    pred_directions = ['Up' if x >= 0 else 'Down' for x in pred_returns]
    
    # Create confusion matrix
    categories = ['Actual Up', 'Actual Down']
    results = {
        'Predicted Up': [
            sum((a == 'Up') & (p == 'Up') for a, p in zip(actual_directions, pred_directions)),
            sum((a == 'Down') & (p == 'Up') for a, p in zip(actual_directions, pred_directions))
        ],
        'Predicted Down': [
            sum((a == 'Up') & (p == 'Down') for a, p in zip(actual_directions, pred_directions)),
            sum((a == 'Down') & (p == 'Down') for a, p in zip(actual_directions, pred_directions))
        ]
    }
    
    # Create plot
    fig = go.Figure()
    
    colors = {'Predicted Up': '#4CAF50', 'Predicted Down': '#F44336'}
    
    for pred in ['Predicted Up', 'Predicted Down']:
        fig.add_trace(go.Bar(
            x=categories,
            y=results[pred],
            name=pred,
            marker_color=colors[pred],
            text=results[pred],
            textposition='auto'
        ))
    
    fig.update_layout(
        barmode='group',
        title='Directional Prediction Accuracy',
        xaxis_title='Actual Market Direction',
        yaxis_title='Count',
        hovermode='x unified',
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=str(yi),
                xanchor='center',
                yanchor='bottom',
                showarrow=False,
                font=dict(size=12)
            ) for xi, x in enumerate(categories) 
              for yi, y in zip(results['Predicted Up'] + results['Predicted Down'], 
                              [results['Predicted Up'][xi], results['Predicted Down'][xi]])
        ]
    )
    
    return fig

def analyze_comparison_df(comparison_df):
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # ===============================
    # Step 1: Tambah Kolom Analisanya
    # ===============================
    comparison_df['Error'] = comparison_df['Predicted Close'] - comparison_df['Actual Close']
    comparison_df['Abs Error'] = comparison_df['Error'].abs()
    comparison_df['% Error'] = comparison_df['Error'] / comparison_df['Actual Close'] * 100

    comparison_df['Pred Direction'] = comparison_df['Predicted Close'].diff()
    comparison_df['Actual Direction'] = comparison_df['Actual Close'].diff()
    comparison_df['Direction Match'] = (comparison_df['Pred Direction'] * comparison_df['Actual Direction']) > 0

    # ===============================
    # Step 2: Hitung Evaluasi Metrik
    # ===============================
    y_true = comparison_df['Actual Close']
    y_pred = comparison_df['Predicted Close']

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    directional_accuracy = 100 * comparison_df['Direction Match'].sum() / (len(comparison_df) - 1)

    # ===============================
    # Step 3: Tampilkan Metrik
    # ===============================
    st.subheader("📊 Evaluasi Akurasi Prediksi (30 Hari Forecast)")
    st.metric("R² Score", f"{r2:.3f}")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("Directional Accuracy", f"{directional_accuracy:.1f}%")

    # ===============================
    # Step 4: Visualisasi Prediksi vs Aktual
    # ===============================
    st.subheader("📈 Forecast vs Actual")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=comparison_df['Date'], y=comparison_df['Actual Close'], name='Actual', mode='lines+markers'))
    fig1.add_trace(go.Scatter(x=comparison_df['Date'], y=comparison_df['Predicted Close'], name='Forecast', mode='lines+markers'))
    fig1.update_layout(title="Forecast vs Actual", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig1, use_container_width=True)

    # ===============================
    # Step 5: Error Distribution
    # ===============================
    st.subheader("📉 Distribusi Error Prediksi")
    fig2 = px.histogram(comparison_df, x='Error', nbins=20, marginal='box', title="Error Distribution")
    fig2.update_layout(xaxis_title='Prediction Error', yaxis_title='Count')
    st.plotly_chart(fig2, use_container_width=True)

    # ===============================
    # Step 6: Akurasi Arah Naik/Turun
    # ===============================
    st.subheader("🔀 Prediksi Arah (Up/Down)")
    fig3 = px.line(comparison_df, x='Date', y=['Actual Direction', 'Pred Direction'], title="Actual vs Predicted Movement")
    st.plotly_chart(fig3, use_container_width=True)

    # ===============================
    # Step 7: Tampilkan DataFrame Detail
    # ===============================
    st.subheader("📋 Detail Perbandingan Prediksi vs Aktual")
    st.dataframe(comparison_df.style.format({
        "Predicted Close": "{:.2f}",
        "Actual Close": "{:.2f}",
        "Error": "{:.2f}",
        "Abs Error": "{:.2f}",
        "% Error": "{:.2f}%"
    }))


def load_best_params(param_path):
    """
    Load Optuna best parameters from a text file.
    Returns a dictionary of parameters.
    """
    best_params = {}
    with open(param_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':', 1)
                val = val.strip()
                try:
                    # Try to interpret as int or float
                    if '.' in val:
                        best_params[key.strip()] = float(val)
                    else:
                        best_params[key.strip()] = int(val)
                except:
                    # Otherwise keep as string
                    best_params[key.strip()] = val
    return best_params