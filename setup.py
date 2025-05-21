from setuptools import setup, find_packages

setup(
    name="btc-eth-prediction",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tensorflow>=2.6.0",
        "requests",
    ],
    author="ES-SAAIDI Youssef, ZEMMAHI Zakariae",
    author_email="youssefessaaidi281@gmail.com, zakariaezemmahi@gmail.com",
    description="Modèle de prédiction du prix du Bitcoin basé sur les données historiques d'Ethereum.",
    keywords="bitcoin, ethereum, prediction, deep learning, LSTM, GRU, CNN",
    url="https://github.com/YoussefAIDT/ETH-to-BTC",
)
