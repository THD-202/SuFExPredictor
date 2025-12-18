## About The Project

Sulfur(VI) fluoride exchange (SuFEx) has emerged as a next-generation click reaction with broad utility across chemical biology, materials science, and drug discovery. Its diverse linkage types greatly expand accessible chemical space but also hinder the predictability of key outcomes, including chemoselectivity and yield. To improve the reliability of SuFEx and facilitate its wider application, we implemented a data–computation–experiment loop to enhance the predictive power of SuFEx chemistry.

## Installation

We recommend using Conda to manage the virtual environment and dependencies.

```bash
# Clone this repository
git clone https://github.com/THD-202/SuFEx.git

# Create and activate conda environment
conda create --name SuFEx python=3.11.7
conda activate SuFEx

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

# 1. Yield prediction 
    Place your dataset into the `Dataset/Prediction/Yield/` directory, then
```sh
    cd SuFEx/Prediction/Yield
    python pre_yield.py
```

# 2. Chemoselectivity prediction 
    Place your dataset into the `Dataset/Prediction/Chemoselectivity/` directory, then
```sh
    cd SuFEx/Prediction/Chemoselectivity
    python pre_Chemoselectivity.py
```


