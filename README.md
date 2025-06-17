# ETH-to-BTC

<div align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.8+-orange.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.0+-red.svg" alt="TensorFlow">
</div>

<div align="center">
  <h2>🚀 Prédiction Bitcoin avec Ethereum</h2>
  <p><em>Modèle prédictif avancé utilisant les corrélations temporelles ETH-BTC</em></p>
</div>

## 📋 Vue d'ensemble

Ce projet de recherche révolutionnaire explore la relation symbiotique entre **Ethereum** et **Bitcoin** pour développer un modèle prédictif de nouvelle génération. En combinant analyse statistique rigoureuse et techniques de deep learning avancées, nous dévoilons les patterns cachés qui régissent les mouvements de ces crypto-monnaies.

## 🎯 Objectif Principal

Notre hypothèse centrale repose sur le fait que le marché Ethereum, grâce à ses caractéristiques uniques (adoption massive des smart contracts, flexibilité applicative, évolution technologique rapide), agit comme un **indicateur avancé** pour le Bitcoin.

### 🔬 Caractéristiques Clés

| Aspect | Description |
|--------|-------------|
| 📊 **Analyse Statistique** | Corrélation de Pearson de 0,82 entre ETH et BTC, indiquant une forte dépendance linéaire positive |
| 🧠 **Deep Learning** | Architecture GRU avec couche Dropout, adaptée à la modélisation des séquences temporelles et à la réduction du surapprentissage |
| ⚡ **Prédiction** | Le modèle prédit le prix du BTC au jour J+1 à partir d'une séquence de 32 jours passés, combinant les données ETH et BTC |

## 🔬 Approche Méthodologique

Notre stratégie multi-dimensionnelle combine :

1. **📈 Analyse Statistique Approfondie** - Quantification des relations temporelles ETH-BTC
2. **📉 Modélisation ARIMA** - Établissement d'une baseline de référence robuste  
3. **🤖 Deep Learning Avancé** - Architecture GRU avec Dropout, efficace pour capturer les dépendances séquentielles des prix
4. **🎯 Validation Empirique** - Tests rigoureux sur données historiques étendues
5. **🖥️ Interface Streamlit** – Déploiement d'une application interactive pour visualiser les prédictions du modèle à partir de nouvelles données

## 🏗️ Architecture du Projet

```
ETH-to-BTC/
├── 📋 README.md                       # Documentation principale du projet
├── 📦 requirements.txt                # Fichier listant les dépendances Python
├── ⚙️ setup.py                        # Fichier de configuration pour le packaging
├── 🚀 app/
│   └── app.py                         # Application Streamlit (interface utilisateur)
├── 🤖 models/                         # Modèles entraînés
│   ├── best_best_model.h5
│   ├── model_gru_bitcoin_eth.h5
│   ├── model_lstm_bitcoin_eth.h5
│   └── model_rnn_simple_bitcoin_eth.h5
├── 📊 data/                           # Dossier pour les données brutes ou nettoyées
├── 📓 notebooks/                      # Notebooks d'analyse et d'expérimentation
│   ├── Analyse_Statistique_Corrélation_Choix_Modèle.ipynb
│   ├── BTC_to_ETH_Best_Model_Search.ipynb
│   ├── ETH-to-BTC_Streamlit.ipynb
│   └── pmdarima.ipynb
├── 💻 src/                            # Code source organisé par fonctionnalité
│   ├── 🎯 predict.py                  # Script principal pour la prédiction
│   ├── 🧠 train.py                    # Script pour entraîner les modèles
│   ├── data/
│   │   ├── __init__.py
│   │   └── collector.py              # Script de collecte des données
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocessing.py          # Fonctions de nettoyage & transformation
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py                  # Définition des architectures de modèles
│   └── utils/
│       ├── __init__.py
│       └── visualization.py          # Fonctions de visualisation
├── 📚 docs/                           # Fichiers de documentation
├── 🧪 .gitattributes                  # Fichier Git (gestion du texte/format)
└── 📖 .readthedocs.yml               # Configuration ReadTheDocs
```

## 🚀 Installation et Démarrage Rapide

> ⚠️ **Important** : Ce projet est conçu exclusivement pour fonctionner sur **Google Colab**. Aucune installation locale n'est nécessaire ou supportée.

### 🌟 Avantages de Google Colab

- ✅ Environnement préconfigué avec GPU gratuit
- ✅ Toutes les bibliothèques nécessaires disponibles
- ✅ Aucune configuration requise
- ✅ Accès immédiat depuis n'importe quel navigateur

### 📦 Deux Options d'Utilisation

#### Option 1 : Utiliser les notebooks existants (Recommandé)

1. **Accédez au dossier notebooks du projet**
2. **Téléchargez les fichiers nécessaires** :
   - `Analyse_Statistique_Corrélation_Choix_Modèle.ipynb` - Analyse complète et modèles de deep learning
   - `pmdarima.ipynb` - Modélisation ARIMA automatisée
   - `BTC-to-ETH_best_model_search.ipynb` - Recherche et sélection du meilleur modèle
   - `BTC-to-ETH_Streamlit.ipynb` - Application Streamlit interactive

3. **Uploadez dans Google Colab** :
   - Ouvrez [Google Colab](https://colab.research.google.com/)
   - Cliquez sur "Importer" → "Choisir un fichier"
   - Sélectionnez le notebook désiré
   - Exécutez toutes les cellules pour accéder à l'application

#### Option 2 : Créer un nouveau notebook

1. **Créez un nouveau notebook sur Google Colab**

2. **Copiez et exécutez ce code dans une cellule :**

```python
# Installation des dépendances
!pip install streamlit pyngrok --quiet

# Clonage du projet
!git clone --recursive https://github.com/YoussefAIDT/ETH-to-BTC.git

# Navigation vers le dossier app
%cd ETH-to-BTC/app

# Configuration du token ngrok (remplacez par votre token)
!ngrok authtoken VOTRE_TOKEN_NGROK
```

3. **Lancez l'application :**

```python
from pyngrok import ngrok

# Ouvre un tunnel vers http://localhost:8501
public_url = ngrok.connect(8501)
print("🚀 L'application est disponible ici :", public_url)

# Démarre Streamlit en arrière-plan
!streamlit run app.py &
```

### 🔑 Configuration Ngrok

Pour utiliser l'Option 2, vous devez :

1. **Créer un compte** sur [Ngrok](https://ngrok.com/)
2. **Obtenir votre token** depuis le dashboard Ngrok
3. **Remplacer** `VOTRE_TOKEN_NGROK` dans le code par votre token personnel

### ✅ Démarrage en 3 étapes

1. **Ouvrez** [Google Colab](https://colab.research.google.com/)
2. **Choisissez** votre méthode (Option 1 ou 2)
3. **Exécutez** les cellules et profitez de l'application !

## 💡 Points Clés

### 🔍 Innovation
Premier modèle exploitant systématiquement ETH comme prédicteur de BTC

### ⚡ Performance
Le modèle GRU atteint un **R² de 0,98**, indiquant une très forte capacité à expliquer la variance des prix réels du Bitcoin, ce qui confirme la qualité de la prédiction.

### 📊 Validation
Le modèle a été soumis à des tests statistiques rigoureux ainsi qu'à une validation croisée étendue, garantissant sa robustesse et sa fiabilité sur des données historiques variées.

## 📊 Utilisation

### Via Google Colab (Recommandé)

**Méthode 1 : Notebook existant**
1. Téléchargez `BTC-to-ETH_Streamlit.ipynb`
2. Importez-le dans Google Colab
3. Exécutez toutes les cellules
4. Accédez à l'application via l'URL générée

**Méthode 2 : Manuel**
```python
# Dans Google Colab
!git clone https://github.com/YoussefAIDT/ETH-to-BTC.git
%cd ETH-to-BTC

# Utiliser le modèle pour la prédiction
from src.predict import predict_btc_price

# Faire une prédiction
prediction = predict_btc_price(eth_data, btc_data)
print(f"Prix BTC prédit : ${prediction:.2f}")
```

## 📚 Documentation et Notebooks

Le projet comprend plusieurs notebooks spécialisés disponibles dans le dossier `notebooks/` :

| Notebook | Description |
|----------|-------------|
| 📊 **Analyse_Statistique_Corrélation_Choix_Modèle.ipynb** | Analyse complète des données et implémentation des modèles de deep learning pour la prédiction ETH/BTC |
| 📈 **pmdarima.ipynb** | Modélisation ARIMA automatisée avec pmdarima pour l'analyse des séries temporelles |
| 🎯 **BTC-to-ETH_best_model_search.ipynb** | Recherche et sélection automatique du meilleur modèle de prédiction basé sur les métriques de performance |
| 🚀 **BTC-to-ETH_Streamlit.ipynb** | Application web interactive Streamlit pour la prédiction en temps réel avec interface utilisateur intuitive |


## 👥 Auteurs

**Développé par :**
- **Youssef ES-SAAIDI** - [🐙 YoussefAIDT GitHub](https://github.com/YoussefAIDT)
- **Zakariae ZEMMAHI** - [🐙 zakariazemmahi GitHub](https://github.com/zakariazemmahi)

## 🆘 Support et Aide

Si vous rencontrez des difficultés :

1. **📖 Vérifiez** que vous utilisez bien Google Colab
2. **🔑 Assurez-vous** d'avoir un token Ngrok valide (Option 2)
3. **🐛 Ouvrez** une [issue sur GitHub](https://github.com/YoussefAIDT/ETH-to-BTC/issues) avec les détails de l'erreur
4. **💬 Contactez** l'équipe de développement

### Liens utiles
- [🐛 Signaler un bug](https://github.com/YoussefAIDT/ETH-to-BTC/issues)
- [👨‍💻 Contact développeur](https://github.com/YoussefAIDT)

---

> **Note:** Cette documentation est en développement actif. Pour les dernières mises à jour, consultez le repository GitHub.
