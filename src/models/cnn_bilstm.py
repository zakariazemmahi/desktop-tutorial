from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

def build_cnn_bilstm_model(seq_length, features=12):
    """
    Construit un modèle CNN-BiLSTM pour la prédiction de prix.

    Args:
        seq_length (int): Longueur des séquences temporelles
        features (int): Nombre de features en entrée

    Returns:
        tensorflow.keras.models.Sequential: Modèle compilé
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
               input_shape=(seq_length, features)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=True,
                          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
        Dropout(0.3),
        LSTM(128, return_sequences=False,
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Entraîne le modèle avec early stopping et learning rate reduction.

    Args:
        model (tensorflow.keras.models.Sequential): Modèle à entraîner
        X_train (np.array): Données d'entraînement (features)
        y_train (np.array): Données d'entraînement (target)
        X_val (np.array): Données de validation (features)
        y_val (np.array): Données de validation (target)
        epochs (int): Nombre maximal d'époques
        batch_size (int): Taille des batchs

    Returns:
        tuple: Historique d'entraînement, modèle entraîné
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00005,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return history, model


def predict_future_price(model, feature_scaler, btc_scaler, bias, last_sequence, days_ahead=30):
    """
    Prédit les prix futurs basés sur la dernière séquence connue.

    Args:
        model (tensorflow.keras.models.Sequential): Modèle entraîné
        feature_scaler (sklearn.preprocessing.MinMaxScaler): Scaler pour les features
        btc_scaler (sklearn.preprocessing.MinMaxScaler): Scaler pour le prix BTC
        bias (float): Biais de correction
        last_sequence (np.array): Dernière séquence connue
        days_ahead (int): Nombre de jours à prédire

    Returns:
        list: Liste des prix prédits
    """
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(days_ahead):
        # Prédiction pour le jour suivant
        pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))

        # Dénormalisation de la prédiction
        pred_btc = btc_scaler.inverse_transform(pred)[0][0]

        # Application de la correction de biais
        pred_btc_corrected = pred_btc + bias
        predictions.append(pred_btc_corrected)

        # Mise à jour nécessaire pour prédire le jour suivant
        # Cette partie simplifiée suppose que nous avons juste besoin de déplacer la séquence
        # Dans un cas réel, il faudrait mettre à jour tous les features

    return predictions
