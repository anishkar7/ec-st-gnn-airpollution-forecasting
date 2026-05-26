import os
import pandas as pd

print("📊 Extracting thesis metrics from the dataset...\n")

# 1. Load the processed dataset from disk
data_path = os.path.join(os.getcwd(), "data", "processed", "processed_delhi_data.csv")
df = pd.read_csv(data_path)

# 2. Print a brief dataset summary (for Section A)
print("--- A. Dataset Summary ---")
print(f"Rows: {len(df)}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Number of stations: {df['station'].nunique()}\n")

# 3. Compute pollutant summary statistics (for Section B)
print("--- B. Pollutant Statistics ---")
pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']

# Calculate mean, median, std, min and max for each pollutant
stats = df[pollutants].agg(['mean', 'median', 'std', 'min', 'max']).T

# Format the table for direct inclusion in the thesis
stats.index.name = 'Pollutant'
stats.columns = ['Mean', 'Median', 'Std', 'Min', 'Max']
stats = stats.round(3)

# Output the stats as a Markdown table that can be pasted into the thesis
print(stats.to_markdown())
print("\n✅ Copy the table above directly into your thesis document!")