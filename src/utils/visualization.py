import matplotlib.pyplot as plt

def plot_predictions(dates, real, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, real, label='Prix BTC réel', color='blue')
    plt.plot(dates, predicted, label='Prix BTC prédit', color='red', linestyle='--')
    plt.title('Prédiction du prix du Bitcoin avec GRU')
    plt.xlabel('Date')
    plt.ylabel('Prix BTC (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
