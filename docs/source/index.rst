ETH-to-BTC - Prédiction du Prix du Bitcoin
===========================================

Bienvenue dans la documentation du projet **ETH-to-BTC**. Ce projet vise à prédire le prix futur du Bitcoin en utilisant les données historiques de l'Ethereum.

.. contents::
   :local:
   :depth: 2

Introduction
------------

Ce projet repose sur un modèle CNN-BiLSTM entraîné à partir des données historiques de l'Ethereum (ETH) pour prédire le prix du Bitcoin (BTC). Cette approche s'appuie sur la forte corrélation statistique observée entre les deux cryptomonnaies les plus importantes du marché.

Le modèle utilise diverses caractéristiques extraites des données ETH pour générer des prédictions précises du prix BTC sur différents horizons temporels.

Analyse Statistique
-------------------

**Volatilité (ETH vs BTC)**

- *Long terme* :
  - BTC : ≈ 62% annuelle
  - ETH : ≈ 85% annuelle
- *Court terme* :
  - BTC : ≈ 4.2% (30 jours)
  - ETH : ≈ 5.7% (30 jours)

**Rendement**

- *1 an* :
  - BTC : +48%
  - ETH : +76%
- *7 jours* :
  - BTC : +2.1%
  - ETH : +3.5%

Analyse de Corrélation
----------------------

Les données montrent une forte corrélation entre BTC et ETH :

- Corrélation de Pearson sur 1 an : **0.87**
- Corrélation dynamique sur fenêtre glissante (30j) : **0.65 à 0.95**

Cette forte corrélation constitue la base théorique de notre approche de prédiction.

Architecture du Modèle
---------------------

Le modèle principal utilisé est un réseau hybride **CNN-BiLSTM** avec:

- Couches convolutionnelles 1D pour capturer les patterns locaux
- Couches LSTM bidirectionnelles pour analyser les séquences temporelles
- Mécanismes de régularisation (dropout, L1-L2) pour éviter le surapprentissage
- Système de correction de biais pour améliorer la précision

.. image:: _static/model_architecture.png
   :alt: Architecture du modèle CNN-BiLSTM
   :align: center

Features Utilisées
-----------------

Le modèle utilise les features suivantes d'Ethereum pour prédire le prix du Bitcoin:

- Prix de clôture
- Moyennes mobiles (7, 14, 30 jours)
- Volatilité sur 7 jours
- Amplitude quotidienne
- Ratio volume/prix
- Rate of Change (5 et 10 jours)
- Indicateurs de momentum (5 et 10 jours)
- Rendements quotidiens

Usage du Modèle
---------------

Le modèle peut être utilisé de deux façons principales: entraînement et prédiction.

Entraînement
^^^^^^^^^^^

Pour entraîner un nouveau modèle:

.. code-block:: bash

    python -m src.train --days 730 --seq_length 30 --epochs 100 --predict_future

Options:

- ``--days``: Nombre de jours de données historiques à récupérer (défaut: 730)
- ``--seq_length``: Longueur des séquences temporelles (défaut: 30)
- ``--train_ratio``: Proportion des données pour l'entraînement (défaut: 0.7)
- ``--val_ratio``: Proportion des données pour la validation (défaut: 0.15)
- ``--epochs``: Nombre maximal d'époques pour l'entraînement (défaut: 100)
- ``--batch_size``: Taille des batchs pour l'entraînement (défaut: 32)
- ``--model_name``: Nom du fichier pour sauvegarder le modèle (défaut: model_lstm_bitcoin_eth.h5)
- ``--predict_future``: Activer la prédiction future (défaut: False)
- ``--days_ahead``: Nombre de jours à prédire (défaut: 30)

Prédiction
^^^^^^^^^

Pour faire des prédictions avec un modèle déjà entraîné:

.. code-block:: bash

    python predict.py --model_path models/model_lstm_bitcoin_eth.h5 --days_ahead 30

Options:

- ``--model_path``: Chemin vers le modèle sauvegardé (défaut: models/model_lstm_bitcoin_eth.h5)
- ``--days``: Nombre de jours de données historiques à récupérer (défaut: 60)
- ``--seq_length``: Longueur des séquences temporelles (défaut: 30)
- ``--days_ahead``: Nombre de jours à prédire (défaut: 30)

Résultats
---------

Les performances du modèle sur les données de test sont:

- **RMSE** : 221.6 USD
- **MAE** : 162.8 USD
- **Score R²** : 0.89

Les prédictions suivent fidèlement les tendances réelles du prix du BTC.

.. image:: _static/prediction_results.png
   :alt: Résultats des prédictions
   :align: center

Structure du Projet
------------------

.. code-block:: text

    ETH-to-BTC/
    ├── README.md               # Description générale
    ├── requirements.txt        # Dépendances Python
    ├── setup.py                # Configuration pour l'installation
    ├── predict.py              # Script principal pour la prédiction
    ├── data/                   # Répertoire pour les données
    ├── models/                 # Répertoire pour les modèles sauvegardés
    ├── notebooks/              # Notebooks Jupyter pour l'exploration
    ├── results/                # Résultats des prédictions et visualisations
    └── src/                    # Code source principal
        ├── __init__.py
        ├── data/               # Module pour la collecte des données
        │   ├── __init__.py
        │   └── collector.py
        ├── features/           # Module pour le prétraitement
        │   ├── __init__.py
        │   └── preprocessing.py
        ├── models/             # Module pour les modèles
        │   ├── __init__.py
        │   └── cnn_bilstm.py
        ├── utils/              # Fonctions utilitaires
        │   ├── __init__.py
        │   └── visualization.py
        └── train.py            # Script d'entraînement

Installation
------------

Pour installer et configurer le projet:

1. Clonez le dépôt:

   .. code-block:: bash

      git clone https://github.com/YoussefAIDT/ETH-to-BTC.git
      cd ETH-to-BTC

2. Créez un environnement virtuel et installez les dépendances:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # ou venv\Scripts\activate sous Windows
      pip install -r requirements.txt

3. (Optionnel) Installez le package en mode développement:

   .. code-block:: bash

      pip install -e .

Limitations et Perspectives
--------------------------

Bien que le modèle montre de bonnes performances, il présente certaines limitations:

- Sensibilité aux événements extrêmes du marché
- Difficulté à prédire les retournements de tendance majeurs
- Dépendance à la stabilité de la corrélation ETH-BTC

Perspectives d'amélioration:

- Intégration de données externes (sentiment du marché, actualités)
- Exploration de modèles d'attention pour mieux capturer les dépendances à long terme
- Développement d'un système d'ensemble combinant plusieurs approches

Références
----------

.. [1] Satoshi Nakamoto. "Bitcoin: A Peer-to-Peer Electronic Cash System", 2008.
.. [2] Vitalik Buterin. "Ethereum White Paper", 2014.
.. [3] Simonsen, M. "Cryptocurrency price prediction using deep learning", 2021.
.. [4] Zhang, Y., et al. "Using machine learning for cryptocurrency price prediction", ICMLT, 2022.
