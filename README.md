## About The Project

Sulfur(VI) fluoride exchange (SuFEx) has emerged as a powerful click reaction for constructing diverse S(VI)-based linkages across chemical biology, materials science, and drug discovery. However, the very diversity of SuFEx hubs and native nucleophiles that underpins its versatility also gives rise to a high-dimensional reaction landscape in which yields and chemoselectivities are difficult to anticipate, limiting its reliability as a general-purpose click chemistry tool. Here, we introduce SuFExPredictor, a machine learning framework developed to predict SuFEx yield and chemoselectivity.

## Installation

We recommend using Conda to manage the virtual environment and dependencies.

```bash
# Clone this repository
git clone https://github.com/THD-202/SuFExPredictor.git

# Create and activate conda environment
conda create --name SuFExPredictor python=3.11.7
conda activate SuFExPredictor

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

## 1. Yield prediction 
Place your dataset into the `SuFExPredictor/Prediction/Yield/` directory, then
```bash
cd SuFExPredictor/Prediction/Yield
python pre_yield.py
```

## 2. Chemoselectivity prediction 
Place your dataset into the `SuFExPredictor/Prediction/Chemoselectivity/` directory, then
```bash
cd SuFExPredictor/Prediction/Chemoselectivity
python pre_Chemoselectivity.py
```


