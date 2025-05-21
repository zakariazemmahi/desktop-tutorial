import requests
import pandas as pd
import time
from datetime import datetime

def collect_data_crypto_compare(symbol, start_timestamp, end_timestamp):
    """
    Récupère les données historiques d'une cryptomonnaie depuis l'API CryptoCompare.
    
    Args:
        symbol (str): Symbole de la cryptomonnaie (ex: 'BTC', 'ETH')
        start_timestamp (int): Timestamp Unix de début
        end_timestamp (int): Timestamp Unix de fin
        
    Returns:
        pandas.DataFrame: DataFrame contenant les données historiques
    """
    url = 'https://min-api.cryptocompare.com/data/v2/histoday'
    days = (end_timestamp - start_timestamp) // (24 * 3600) + 1
    limit = min(2000, days)

    params = {
        'fsym': symbol,
        'tsym': 'USD',
        'limit': limit,
        'toTs': end_timestamp
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['Response'] == 'Success':
            df = pd.DataFrame(data['Data']['Data'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        else:
            print(f"Erreur API: {data['Message']}")
            return None
    else:
        print(f"Erreur HTTP: {response.status_code}")
        return None


def get_crypto_data(symbols=['BTC', 'ETH'], days=730):
    """
    Récupère les données historiques pour plusieurs cryptomonnaies.
    
    Args:
        symbols (list): Liste des symboles des cryptomonnaies
        days (int): Nombre de jours d'historique à récupérer
        
    Returns:
        dict: Dictionnaire contenant les DataFrames pour chaque symbole
    """
    end_timestamp = int(time.time())
    start_timestamp = end_timestamp - (days * 24 * 3600)
    
    data_dict = {}
    
    for symbol in symbols:
        print(f"Récupération des données pour {symbol}...")
        data = collect_data_crypto_compare(symbol, start_timestamp, end_timestamp)
        if data is not None:
            data_dict[symbol] = data
            print(f"✓ {len(data)} entrées récupérées pour {symbol}")
        else:
            print(f"✗ Échec de la récupération des données pour {symbol}")
    
    return data_dict
