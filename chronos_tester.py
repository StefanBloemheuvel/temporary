import pandas as pd 
import torch
from chronos import ChronosPipeline
import matplotlib.pyplot as plt  
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error  
from statsmodels.tsa.holtwinters import ExponentialSmoothing  

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.1, shuffle=False)

# Initialize the ChronosPipeline
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="mps",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

# Convert the training data into a tensor
context = torch.tensor(train_df["#Passengers"].values)

# Forecast using the pipeline
forecast = pipeline.predict(
    context=context,
    prediction_length=len(test_df),
    num_samples=1,  # Use 1 sample for point forecasts to calculate MSE
)

# Calculate median forecasts from Chronos
predicted_values_chronos = forecast[0].numpy().squeeze()

# Exponential Smoothing
# es_model = ExponentialSmoothing(train_df["#Passengers"], trend="add", seasonal="add", seasonal_periods=12) # and by giving the hint of 12 observation seasonality...
# es_model = ExponentialSmoothing(train_df["#Passengers"], trend='add') # This will improve the results of the exponential model
es_model = ExponentialSmoothing(train_df["#Passengers"])

es_model_fit = es_model.fit()
es_predictions = es_model_fit.forecast(len(test_df))

# Calculate MSE for each method
mse_chronos = mean_squared_error(test_df["#Passengers"], predicted_values_chronos)
mse_es = mean_squared_error(test_df["#Passengers"], es_predictions)

# Print MSE results
print("MSE - Chronos: ", mse_chronos)
print("MSE - Exponential Smoothing: ", mse_es)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(df["#Passengers"], color="royalblue", label="Historical Data")
plt.plot(range(len(train_df), len(train_df) + len(test_df)), predicted_values_chronos, color="tomato", label="Chronos Forecast")
plt.plot(range(len(train_df), len(train_df) + len(test_df)), es_predictions, color="orange", label="Exponential Smoothing Forecast")
plt.legend()
plt.grid()
plt.show()
