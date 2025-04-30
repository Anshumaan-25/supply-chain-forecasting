# Supply Chain Forecasting

This project implements a **demand forecasting model** for a fashion and beauty startup's supply chain. The goal is to predict the number of products sold using historical sales data and operational features.

## Project Overview

The project consists of the following components:

- **`code/`**: Python script to build, train, and evaluate the demand forecasting model.
  - `supply_chain_forecasting.py`: The main Python script that contains the entire model pipeline, from data preprocessing to training and evaluation.

- **`data/`**: Contains the raw dataset used to train and test the model.
  - `supply_chain_data.csv`: The dataset with features like product type, price, availability, stock levels, shipping details, and more.

- **`models/`**: The directory where the trained model is saved.
  - `demand_forecasting_model.h5`: The trained Keras model that predicts the number of products sold.

- **`report/`**: The final report detailing the methodology, evaluation, and results.
  - `final_report_supply_chain_forecasting.docx`: Word document containing the project description, data preprocessing steps, model architecture, evaluation, and results.
  - `train_val_loss.png`: Plot showing the training vs. validation loss.
  - `true_vs_predicted.png`: Plot showing true vs predicted sales.

## How to Run the Code

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/supply-chain-forecasting.git
