<div align="center">

<!-- Main Title & Banner -->
<h1>🛍️ Retail Sales Forecasting using LSTM</h1>
<h3>End-to-End Time Series Prediction Pipeline | IBM Data Science Methodology</h3>

<br>

<!-- Tech Stack - Big Badges with Context -->
<table>
  <tr>
    <td align="center"><b>Core Logic & Science</b></td>
    <td align="center"><b>Deep Learning Engine</b></td>
    <td align="center"><b>Data Engineering</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/TensorFlow-Backend-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
      <br>
      <img src="https://img.shields.io/badge/Keras-High%20Level%20API-D00000?style=for-the-badge&logo=keras&logoColor=white" />
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white" />
      <br>
      <img src="https://img.shields.io/badge/NumPy-Math%20Calc-013243?style=for-the-badge&logo=numpy&logoColor=white" />
    </td>
  </tr>
  <tr>
    <td align="center"><b>Visualization</b></td>
    <td align="center"><b>Preprocessing</b></td>
    <td align="center"><b>License</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://img.shields.io/badge/Matplotlib-Plotting-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black" />
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/Scikit_Learn-Normalization-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
    </td>
  </tr>
</table>

</div>


# <img src="https://img.icons8.com/3d-fluency/94/target.png" width="50" height="50" style="vertical-align:bottom"/> Deep Learning for Retail Sales Forecasting
**End-to-End Time Series Analysis Based on IBM Data Science Methodology**

This project implements an End-to-End Time Series Forecasting pipeline to predict future daily sales for a UK-based online retail store. Built strictly upon the IBM Data Science Methodology, the project handles real-world data challenges including noise, missing values, and seasonality.

The core engine is a Long Short-Term Memory (LSTM) neural network, designed to capture complex temporal dependencies in transactional data. The model processes over 500,000 transactions, aggregates them into daily time-series, and forecasts future sales with high robustness against outliers.

<br>

## <img src="https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/external-process-productivity-flaticons-lineal-color-flat-icons-3.png" width="40" height="40" style="vertical-align:middle"/> Project Methodology (IBM Lifecycle)

This project is strictly structured around the **IBM Data Science Professional Certificate** methodology, ensuring a standardized and industrial-grade workflow.

### 🔹 Phase 1: Data Acquisition
*   **Source:** UCI Machine Learning Repository (Online Retail Dataset).
*   **Mechanism:** An automated Python script (`requests` + `os`) checks for the dataset locally. If missing, it fetches the `.xlsx` file directly from the source, ensuring reproducibility across different environments.

### 🔹 Phase 2: Data Cleaning & Wrangling ("Cleaning Data from Dirt to Clean!")
The raw data contained significant noise. Key cleaning steps included:
*   **Handling Cancellations:** Removed transactions where `InvoiceNo` started with 'C' to prevent negative sales skew.
*   **Sanity Checks:** Eliminated records with negative quantities or missing `CustomerID`.
*   **Feature Engineering:** Created a `TotalSales` feature (`Quantity` × `UnitPrice`).
*   **Outcome:** Reduced dataset from **540,000** to **398,000** high-quality records.

### 🔹 Phase 3: Exploratory Data Analysis (EDA)
*   **Visual Analysis:** Plotted daily sales trends to identify patterns.
*   **Discovery:** Strong seasonality was observed, specifically significant sales spikes during **November and December** (Pre-holiday season), validating the need for a model capable of learning non-linear dependencies (LSTM).

### 🔹 Phase 4: Preprocessing for Deep Learning
To feed the data into the LSTM network, rigorous transformation was required:
1.  **Resampling:** Enforced daily continuity by filling missing dates with 0 to preserve the time-axis integrity.
2.  **Scaling:** Applied `MinMaxScaler` to normalize data into the `[0, 1]` range for efficient gradient descent.
3.  **Sequence Generation:** Constructed a sliding window of **60 Days (Look-back)** to predict the next day's sales (Shape: `Samples, 60, 1`).

## <img src="https://img.icons8.com/color/48/000000/artificial-intelligence.png" width="30" height="30" style="vertical-align:middle"/> Model Architecture

The Neural Network is built using the TensorFlow/Keras Sequential API:

| Layer | Parameters | Function |
| :--- | :--- | :--- |
| LSTM | Units: 50 | Extracts long-term temporal patterns from the 60-day window. |
| Dropout | Rate: 0.2 | Regularization layer to prevent overfitting by randomly disabling 20% of neurons. |
| Dense | Units: 1 | Final regression layer outputting the predicted sales value. |
| Optimizer | Adam | Adaptive learning rate (LR=0.001) for stable convergence. |
| Loss | MSE | Mean Squared Error used as the objective function. |

<br>

<br>

## <img src="https://img.icons8.com/3d-fluency/94/chart.png" width="35" height="35" style="vertical-align:bottom"/> Experimental Results & Performance Analysis

The core achievement of this project is the LSTM model's ability to **handle extreme volatility** and isolate the signal from a noisy, real-world transactional dataset. Despite the irregular spikes and the presence of cancelled transactions, the model successfully converged.

### 🏆 Model Robustness & Data Handling
The LSTM architecture (50 units, Look-back 60) proved exceptionally capable of handling the specific challenges of retail data:

*   **Handling Seasonality:** The model successfully identified the massive demand surge in **November/December**, learning that sales follow a yearly cyclical pattern rather than a random walk.
*   **Noise Immunity:** By utilizing a `Dropout(0.2)` layer and aggressive preprocessing (Phase 2), the model ignored outliers (such as bulk cancellations), preventing them from distorting the forecast.
*   **Temporal Continuity:** The resampling strategy in Phase 4 allowed the model to handle "zero-sales days" (holidays) without breaking the sequence, ensuring a smooth gradient descent.

### 📊 Quantitative Metrics (Final Epoch)
Performance measured on the validation set shows a stable and generalized model:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Training Loss (MSE)** | `0.0068` | High precision in learning historical patterns. |
| **Validation Loss (MSE)** | `0.0123` | Minimal gap vs training loss indicates **no overfitting**. |
| **Stability** | `High` | Loss stabilized after Epoch 10, showing rapid feature extraction. |

### 📉 Visual Inference
The forecast visualization (saved in `reports/figures`) demonstrates that the predicted trendline tightly follows the actual sales curve, accurately reacting to the weekly oscillations while maintaining the correct long-term trajectory.

## <img src="https://img.icons8.com/color/48/000000/folder-invoices--v1.png" width="30" height="30" style="vertical-align:middle"/> Repository Structure

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
