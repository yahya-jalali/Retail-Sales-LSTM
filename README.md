# Retail Sales Forecasting using LSTM Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Keras](https://img.shields.io/badge/Keras-Deep_Learning-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Executive Summary

This project implements an End-to-End Time Series Forecasting pipeline to predict future daily sales for a UK-based online retail store. Built strictly upon the IBM Data Science Methodology, the project handles real-world data challenges including noise, missing values, and seasonality.

The core engine is a Long Short-Term Memory (LSTM) neural network, designed to capture complex temporal dependencies in transactional data. The model processes over 500,000 transactions, aggregates them into daily time-series, and forecasts future sales with high robustness against outliers.

## Project Methodology

The workflow follows the standard IBM Data Science Lifecycle phases:

Phase 1: Data Acquisition
Automated and stable download mechanism implemented using requests. The script automatically checks for the dataset and downloads it from the UCI Machine Learning Repository if missing, ensuring reproducibility across different environments.

Phase 2: Data Cleaning and Wrangling
Raw data contained significant noise. Key cleaning steps included:
1. Filtering Cancellations: Removed transactions where InvoiceNo started with C.
2. Sanity Checks: Eliminated records with negative quantities or missing CustomerIDs.
3. Feature Engineering: Created a TotalSales feature (Quantity x UnitPrice).
4. Outcome: Reduced dataset from 540,000 to 398,000 high-quality records.

Phase 3: Exploratory Data Analysis (EDA)
Visual analysis revealed strong seasonality in the data, specifically identifying significant sales spikes during November and December (pre-holiday season), which validated the choice of LSTM for capturing non-linear trends.

Phase 4: Preprocessing for Deep Learning
Data was transformed to fit the 3D tensor format required by LSTM:
- Resampling: Enforced daily continuity by filling missing dates with 0 to preserve the time-axis integrity.
- Scaling: Applied MinMaxScaler to normalize data into the [0, 1] range for efficient gradient descent.
- Sequence Generation: Constructed a sliding window of 60 Days (Look-back) to predict the next day sales.

## Model Architecture

The Neural Network is built using the TensorFlow/Keras Sequential API:

| Layer | Parameters | Function |
| :--- | :--- | :--- |
| LSTM | Units: 50 | Extracts long-term temporal patterns from the 60-day window. |
| Dropout | Rate: 0.2 | Regularization layer to prevent overfitting by randomly disabling 20% of neurons. |
| Dense | Units: 1 | Final regression layer outputting the predicted sales value. |
| Optimizer | Adam | Adaptive learning rate (LR=0.001) for stable convergence. |
| Loss | MSE | Mean Squared Error used as the objective function. |

## Performance and Results

The model was trained over 50 epochs with a batch size of 32.
- Convergence: The training logs indicate a quick convergence within the first 10 epochs.
- Generalization: The gap between Training Loss (approx 0.0068) and Validation Loss (approx 0.0123) remained stable, indicating the model successfully learned general trends without memorizing noise.
- Robustness: The architecture successfully handled the volatility caused by seasonal spikes.

## Repository Structure
```text
├── data/
│   ├── raw/            # Raw Excel dataset (Auto-downloaded)
│   └── processed/      # Cleaned and processed CSV files
├── notebooks/
│   └── Retail_Forecast_LSTM.ipynb   # Main analysis and modeling notebook
├── reports/
│   └── figures/        # Generated plots and visualizations
├── src/
│   └── config.py       # Configuration settings
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

## How to Run

1. Clone the Repository
   git clone https://github.com/YOUR_USERNAME/Retail-Sales-LSTM.git
   cd Retail-Sales-LSTM

2. Install Dependencies
   pip install -r requirements.txt

3. Execute the Notebook
   Launch Jupyter Notebook. The script includes an automated data downloader, so no manual file setup is required.
   jupyter notebook notebooks/Retail_Forecast_LSTM.ipynb

## Requirements

The project relies on the following core libraries:
- python>=3.8
- tensorflow
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- openpyxl
- requests
- tqdm

## License

MIT License

Copyright (c) 2024 Yahya Jalali

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Author: Yahya Jalali
