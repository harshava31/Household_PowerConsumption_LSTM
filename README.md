# Household_PowerConsumption_LSTM

This project implements a time-series forecasting model to predict **Global Active Power** using Long Short-Term Memory (LSTM) networks. The model is built using PyTorch and trained on the [UCI Household Electric Power Consumption dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption).

## ## Project Overview
The goal is to predict the next minute's electrical power consumption based on a historical sequence of several variables, including voltage, global intensity, and sub-metering data.

### Key Features
* **Data Preprocessing:** Handles missing values, performs outlier removal using the IQR method, and scales features using `MinMaxScaler`.
* **Sequential Modeling:** Utilizes a 3-layer stacked LSTM architecture to capture long-term temporal dependencies.
* **Evaluation:** Achieves high predictive accuracy, measured by R-squared ($R^2$) and Root Mean Squared Error (RMSE).

---

## ## Dataset Summary
The [dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) contains 2,075,259 samples gathered between December 2006 and November 2010. 

**Key Variables:**
* **Global_active_power:** Household global minute-averaged active power (kW).
* **Global_reactive_power:** Household global minute-averaged reactive power (kW).
* **Voltage:** Minute-averaged voltage (V).
* **Global_intensity:** Household global minute-averaged current intensity (A).
* **Sub_metering_1, 2, & 3:** Energy consumption for specific appliance groups (e.g., kitchen, laundry, climate control).

---

## ## Model Architecture
The model is implemented as a PyTorch `nn.Module` with the following structure:
* **Input Layer:** Accepts a sequence of 50 time steps across 7 features.
* **LSTM Layers:** 3 stacked LSTM layers with 64 hidden units each.
* **Output Layer:** A fully connected (Linear) layer that outputs the predicted `Global_active_power`.

---

## ## Performance Results
The model was trained for 20 epochs with early stopping (patience = 3). 

| Metric | Value |
| :--- | :--- |
| **Test RMSE** | 0.1473 |
| **Test $R^2$ Score** | 0.9325 |

---

## ## Requirements
To run this notebook, you will need the following libraries:
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `torch` (PyTorch)

## ## Usage
1.  **Load Data:** Ensure `household_power_consumption.txt` is available in your environment.
2.  **Preprocess:** Run the data cleaning and scaling cells to prepare the tensors.
3.  **Train:** Execute the training loop; the best weights will be saved as `best_model_weights.pth`.
4.  **Evaluate:** Generate the "Actual vs Predicted" plots to visualize model performance.

---

## ## References
* [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.
* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
