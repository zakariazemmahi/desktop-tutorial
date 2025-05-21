import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(btc_df, eth_df):
    """
    Prétraite les données BTC et ETH et crée des features d'ingénierie.

    Args:
        btc_df (pandas.DataFrame): DataFrame contenant les données BTC
        eth_df (pandas.DataFrame): DataFrame contenant les données ETH

    Returns:
        pandas.DataFrame: DataFrame combiné avec features d'ingénierie
    """
    df = pd.merge(btc_df[['time', 'close', 'high', 'low', 'volumefrom', 'volumeto']],
                  eth_df[['time', 'close', 'high', 'low', 'volumefrom', 'volumeto']],
                  on='time',
                  suffixes=('_btc', '_eth'))
    df = df.sort_values('time')

    # Création des features techniques pour ETH
    df['eth_ma7'] = df['close_eth'].rolling(window=7).mean()
    df['eth_ma14'] = df['close_eth'].rolling(window=14).mean()
    df['eth_ma30'] = df['close_eth'].rolling(window=30).mean()
    df['eth_volatility7'] = df['close_eth'].rolling(window=7).std()
    df['eth_daily_range'] = df['high_eth'] - df['low_eth']
    df['eth_volume_price_ratio'] = df['volumeto_eth'] / df['close_eth']
    df['eth_roc5'] = df['close_eth'].pct_change(periods=5)
    df['eth_roc10'] = df['close_eth'].pct_change(periods=10)
    df['btc_return'] = df['close_btc'].pct_change()
    df['eth_return'] = df['close_eth'].pct_change()
    df['eth_momentum5'] = df['close_eth'] / df['close_eth'].shift(5)
    df['eth_momentum10'] = df['close_eth'] / df['close_eth'].shift(10)

    # Suppression des lignes avec valeurs manquantes
    df.dropna(inplace=True)
    return df


def create_sequences(df, seq_length=30):
    """
    Crée des séquences temporelles pour l'apprentissage du modèle.

    Args:
        df (pandas.DataFrame): DataFrame avec les données prétraitées
        seq_length (int): Longueur des séquences temporelles

    Returns:
        tuple: X (features) et y (target) sous forme de np.array
    """
    features_list = [
        'close_eth', 'eth_ma7', 'eth_ma14', 'eth_ma30',
        'eth_volatility7', 'eth_daily_range', 'eth_volume_price_ratio',
        'eth_roc5', 'eth_roc10', 'eth_momentum5', 'eth_momentum10', 'eth_return'
    ]
    target = 'close_btc'
    X, y = [], []
    for i in range(len(df) - seq_length):
        feature_sequence = df[features_list].iloc[i:i+seq_length].values
        X.append(feature_sequence)
        y.append(df[target].iloc[i+seq_length])
    return np.array(X), np.array(y).reshape(-1, 1)


def scale_data(data):
    """
    Normalise les données avec MinMaxScaler.

    Args:
        data (pandas.DataFrame): DataFrame avec les données à normaliser

    Returns:
        tuple: DataFrame normalisé, feature_scaler, btc_scaler
    """
    feature_scaler = MinMaxScaler()
    btc_scaler = MinMaxScaler()

    features_list = [
        'close_eth', 'eth_ma7', 'eth_ma14', 'eth_ma30',
        'eth_volatility7', 'eth_daily_range', 'eth_volume_price_ratio',
        'eth_roc5', 'eth_roc10', 'eth_momentum5', 'eth_momentum10', 'eth_return'
    ]

    data[features_list] = feature_scaler.fit_transform(data[features_list])
    data['close_btc'] = btc_scaler.fit_transform(data[['close_btc']])

    return data, feature_scaler, btc_scaler


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Divise les données en ensembles d'entraînement, validation et test.

    Args:
        X (np.array): Features
        y (np.array): Target
        train_ratio (float): Proportion des données pour l'entraînement
        val_ratio (float): Proportion des données pour la validation

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)

    X_train = X[:train_size]
    X_val = X[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]

    y_train = y[:train_size]
    y_val = y[train_size:train_size+val_size]
    y_test = y[train_size+val_size:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_features_list():
    """
    Retourne la liste des features utilisées dans le modèle.

    Returns:
        list: Liste des noms de features
    """
    return [
        'close_eth', 'eth_ma7', 'eth_ma14', 'eth_ma30',
        'eth_volatility7', 'eth_daily_range', 'eth_volume_price_ratio',
        'eth_roc5', 'eth_roc10', 'eth_momentum5', 'eth_momentum10', 'eth_return'
    ]
