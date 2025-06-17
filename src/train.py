import time
from datetime import datetime
from data.collector import collect_data_crypto_compare
from features.preprocessing import preprocess_data
from models.model import train_gru_model

if __name__ == "__main__":
    today = datetime.today()
    end_timestamp = int(time.mktime(today.timetuple()))
    start_timestamp = 1614556800  # 1 mars 2021

    btc_data = collect_data_crypto_compare('BTC', end_timestamp)
    eth_data = collect_data_crypto_compare('ETH', end_timestamp)

    if btc_data is not None and eth_data is not None:
        data = preprocess_data(btc_data, eth_data)
        train_gru_model(data)
    else:
        print("❌ Échec de la récupération des données.")
