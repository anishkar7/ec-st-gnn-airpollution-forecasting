import os
import pandas as pd

print("📊 Extracting Real Thesis Metrics from Dataset...\n")

# 1. Load your actual dataset
data_path = os.path.join(os.getcwd(), "data", "processed", "processed_delhi_data.csv")
df = pd.read_csv(data_path)

# 2. Extract Dataset Summary (Section A)
print("--- A. Dataset Summary ---")
print(f"Rows: {len(df)}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Number of stations: {df['station'].nunique()}\n")

# 3. Extract Pollutant Statistics (Section B)
print("--- B. Pollutant Statistics ---")
pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']

# Calculate Mean, Median, Std, Min, Max
stats = df[pollutants].agg(['mean', 'median', 'std', 'min', 'max']).T

# Format output to perfectly match your thesis table
stats.index.name = 'Pollutant'
stats.columns = ['Mean', 'Median', 'Std', 'Min', 'Max']
stats = stats.round(3)

# Print as a clean Markdown table ready to be pasted into your thesis
print(stats.to_markdown())
print("\n✅ Copy the table above directly into your thesis document!")