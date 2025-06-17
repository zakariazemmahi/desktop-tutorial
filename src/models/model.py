import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

def create_sequences(eth_prices, btc_prices, seq_length):
    X, y = [], []
    for i in range(len(eth_prices) - seq_length):
        X.append(np.column_stack((eth_prices[i:i+seq_length], btc_prices[i:i+seq_length])))
        y.append(btc_prices[i+seq_length])
    return np.array(X), np.array(y)

def build_gru_model(seq_length, units=128, dropout=0.1, learning_rate=0.01):
    model = Sequential()
    model.add(GRU(units=units, input_shape=(seq_length, 2)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def train_gru_model(data, seq_length=32, units=128, dropout=0.1, learning_rate=0.01):
    eth_scaler = MinMaxScaler()
    btc_scaler = MinMaxScaler()
    eth_scaled = eth_scaler.fit_transform(data[['close_eth']])
    btc_scaled = btc_scaler.fit_transform(data[['close_btc']])

    X, y = create_sequences(eth_scaled, btc_scaled, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 2))

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = build_gru_model(seq_length, units, dropout, learning_rate)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test)
    y_test_inv = btc_scaler.inverse_transform(y_test)
    y_pred_inv = btc_scaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)

    print(f"MSE : {mse:.2f} | RMSE : {rmse:.2f} | R² : {r2:.4f}")

    model.save("best_best_model.h5")
    print("✅ Modèle sauvegardé sous le nom 'best_best_model.h5'")
    return model
