# optuna_lstm_tuner.py

import os
import optuna
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout, Bidirectional, Conv1D,MaxPooling1D
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.optimizers import Adam, Adamax,RMSprop,Nadam
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping, Callback

def build_lstm_model(input_shape, trial=None, fixed_params=None):
    from keras.layers import LayerNormalization, Attention, Input, Conv1D, MaxPooling1D, Dense, Dropout, LSTM, GRU, Bidirectional
    from keras.models import Sequential, Model
    import tensorflow as tf

    if fixed_params:
        params = fixed_params
    else:
        params = {
                    "num_layers": trial.suggest_int("num_layers", 1, 5),
                    "units": trial.suggest_int("units", 32, 256),
                    "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                    "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
                    "model_type": trial.suggest_categorical("model_type", ["LSTM", "GRU", "CNN-LSTM", "Attn-LSTM"]),
                    "loss_fn": trial.suggest_categorical("loss_fn", ["mse", "huber"]),
                    "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamax", "nadam", "rmsprop"]),
                    "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                }

    model_type = params.get("model_type", "LSTM")
    layer_unit_log = []  # log per-layer units and dropout

    # === Attn-LSTM uses functional API ===
    if model_type == "Attn-LSTM":
        inp = Input(shape=input_shape)
        x = inp

        for i in range(params["num_layers"]):
            return_sequences = True  # Attention requires sequences
            layer_units = max(4, int(params["units"] / (2 ** i)))
            rnn = LSTM(layer_units, return_sequences=return_sequences)
            rnn_type = "LSTM"

            if params["bidirectional"]:
                x = Bidirectional(rnn)(x)
                layer_type = f"Bi-{rnn_type}"
            else:
                x = rnn(x)
                layer_type = rnn_type

            x = Dropout(params["dropout"])(x)
            layer_unit_log.append(f"layer{i+1}({layer_type},{layer_units}), drop{i+1}({params['dropout']:.2f})")

        x = LayerNormalization()(x)
        x = Attention()([x, x])
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        out = Dense(1)(x)
        model = Model(inputs=inp, outputs=out)

    else:
        model = Sequential()
        model.add(Input(shape=input_shape))

        if model_type == "CNN-LSTM":
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LayerNormalization())

        for i in range(params["num_layers"]):
            return_sequences = i < params["num_layers"] - 1
            layer_units = max(4, int(params["units"] / (2 ** i)))

            if model_type == "GRU":
                rnn = GRU(layer_units, return_sequences=return_sequences)
                rnn_type = "GRU"
            else:
                rnn = LSTM(layer_units, return_sequences=return_sequences)
                rnn_type = "LSTM"

            if params["bidirectional"]:
                model.add(Bidirectional(rnn))
                layer_type = f"Bi-{rnn_type}"
            else:
                model.add(rnn)
                layer_type = rnn_type

            model.add(Dropout(params["dropout"]))
            layer_unit_log.append(f"layer{i+1}({layer_type},{layer_units}), drop{i+1}({params['dropout']:.2f})")

        model.add(Dense(1))

    # === Loss & Optimizer ===
    loss = tf.keras.losses.Huber() if params["loss_fn"] == "huber" else tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"]) if params["optimizer"] == "adam" else tf.keras.optimizers.Adamax(learning_rate=params["lr"])
    model.compile(optimizer=optimizer, loss=loss)

    params["layer_config"] = layer_unit_log
    return model, params

# Custom pruning callback to avoid pruning too early
class SafePruneCallback(TFKerasPruningCallback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch < 5:  # Wait until at least epoch 5
            return
        super().on_epoch_end(epoch, logs)

def objective(trial, X_train, y_train, X_val, y_val, input_shape):
    tf.keras.backend.clear_session()

    model, _ = build_lstm_model(input_shape, trial)

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 20, 80)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        SafePruneCallback(trial, monitor="val_loss")
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    preds = model.predict(X_val, verbose=0)
    return mean_squared_error(y_val, preds)


def run_optuna_tuning(X_train, y_train, X_val, y_val, input_shape, output_dir, n_trials=30):
    def wrapped_objective(trial):
        return objective(trial, X_train, y_train, X_val, y_val, input_shape)

    study = optuna.create_study(direction='minimize')
    study.optimize(wrapped_objective, n_trials=n_trials)

    best_params = study.best_trial.params

    # 🧠 Rebuild model with full config (including layer_config)
    model, full_params = build_lstm_model(input_shape, trial=None, fixed_params=best_params)

    # 📝 Save to txt
    os.makedirs(output_dir, exist_ok=True)
    param_path = os.path.join(output_dir, "best_lstm_params.txt")
    with open(param_path, "w") as f:
        for k, v in full_params.items():
            if isinstance(v, list):  # for layer_config
                f.write(f"{k}: {v}\n")
            else:
                f.write(f"{k}: {v}\n")

    print("✅ Best parameters saved to", param_path)
    return full_params, study.best_value



def save_model_summary(model, save_dir, name="model_summary.txt"):
    path = os.path.join(save_dir, name)
    with open(path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print("📄 Model summary saved:", path)

from sklearn.model_selection import train_test_split

def prepare_data_for_optuna(series, test_size=0.2, ahead=1):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    def create_dateback(data, n_lags=60, ahead=1):
        X, y = [], []
        for i in range(len(data) - n_lags - ahead + 1):
            X.append(data[i:i + n_lags])
            y.append(data[i + n_lags + ahead - 1])
        return np.array(X), np.array(y)

    # Scale the series
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    # Create sequences
    X, y = create_dateback(series_scaled, n_lags=60, ahead=ahead)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Add feature dimension

    # Split train-validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)
    return X_train, y_train, X_val, y_val, X.shape

def build_lstm_model_high(input_shape, trial):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=trial.suggest_float("lr", 1e-4, 5e-3, log=True)),
                  loss="mse")
    return model


def build_lstm_model_mid(input_shape, trial):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer=Nadam(learning_rate=trial.suggest_float("lr", 1e-4, 5e-3, log=True)),
                  loss="huber")
    return model


def build_lstm_model_low(input_shape, trial):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(GRU(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(16))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(learning_rate=trial.suggest_float("lr", 1e-4, 5e-3, log=True)),
                  loss="mse")
    return model

def frequency_objective(trial, X_train, y_train, X_val, y_val, input_shape, model_builder):
    tf.keras.backend.clear_session()
    model = model_builder(input_shape, trial)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        SafePruneCallback(trial, monitor="val_loss")
    ]

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=trial.suggest_int("epochs", 30, 100),
              batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
              callbacks=callbacks,
              verbose=0)

    preds = model.predict(X_val, verbose=0)
    return mean_squared_error(y_val, preds)

def run_optuna_tuning_high(X_train, y_train, X_val, y_val, input_shape, output_dir):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: frequency_objective(
        trial, X_train, y_train, X_val, y_val, input_shape, build_lstm_model_high), n_trials=30)
    model = build_lstm_model_high(input_shape, trial=study.best_trial)
    return model, study.best_params


def run_optuna_tuning_mid(X_train, y_train, X_val, y_val, input_shape, output_dir):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: frequency_objective(
        trial, X_train, y_train, X_val, y_val, input_shape, build_lstm_model_mid), n_trials=30)
    model = build_lstm_model_mid(input_shape, trial=study.best_trial)
    return model, study.best_params


def run_optuna_tuning_low(X_train, y_train, X_val, y_val, input_shape, output_dir):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: frequency_objective(
        trial, X_train, y_train, X_val, y_val, input_shape, build_lstm_model_low), n_trials=30)
    model = build_lstm_model_low(input_shape, trial=study.best_trial)
    return model, study.best_params
