import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def plot_predictions(test_dates, y_test, y_pred, y_pred_corrected=None, save_path=None):
    """
    Trace les prédictions par rapport aux valeurs réelles.

    Args:
        test_dates (np.array): Dates pour l'axe x
        y_test (np.array): Valeurs réelles
        y_pred (np.array): Valeurs prédites
        y_pred_corrected (np.array, optional): Valeurs prédites corrigées
        save_path (str, optional): Chemin pour sauvegarder le graphique
    """
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, label='Réel', color='blue')
    plt.plot(test_dates, y_pred, label='Prédit (non corrigé)', color='red', linestyle='--')

    if y_pred_corrected is not None:
        plt.plot(test_dates, y_pred_corrected, label='Prédit (corrigé)', color='green', linestyle='-.')

    plt.title('Prix BTC: Réel vs Prédit')
    plt.xlabel('Date')
    plt.ylabel('Prix (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_future_predictions(start_date, predictions, save_path=None):
    """
    Trace les prédictions futures.

    Args:
        start_date (datetime): Date de début des prédictions
        predictions (list): Liste des prix prédits
        save_path (str, optional): Chemin pour sauvegarder le graphique
    """
    dates = pd.date_range(start=start_date, periods=len(predictions))

    plt.figure(figsize=(12, 6))
    plt.plot(dates, predictions, label='Prédictions futures', color='green', marker='o')
    plt.title('Prédictions futures du prix BTC')
    plt.xlabel('Date')
    plt.ylabel('Prix prédit (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def calculate_metrics(y_true, y_pred):
    """
    Calcule les métriques d'évaluation du modèle.

    Args:
        y_true (np.array): Valeurs réelles
        y_pred (np.array): Valeurs prédites

    Returns:
        dict: Dictionnaire contenant les métriques
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_true - y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'Bias': bias
    }


def print_metrics(metrics, prefix=""):
    """
    Affiche les métriques d'évaluation de manière formatée.

    Args:
        metrics (dict): Dictionnaire contenant les métriques
        prefix (str, optional): Préfixe à afficher avant les métriques
    """
    print(f"{prefix}MSE: {metrics['MSE']:.2f}, RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R²']:.4f}, Bias: {metrics['Bias']:.2f}")


def plot_training_history(history, save_path=None):
    """
    Trace l'historique d'entraînement du modèle.

    Args:
        history (tensorflow.keras.callbacks.History): Historique d'entraînement
        save_path (str, optional): Chemin pour sauvegarder le graphique
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Perte (MSE)')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='train')
    plt.plot(history.history['val_mae'], label='validation')
    plt.title('Erreur absolue moyenne (MAE)')
    plt.xlabel('Époque')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
