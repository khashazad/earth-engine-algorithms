import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

target_final_intercept = 1

data = pd.read_csv(os.path.relpath("./graphs/estimate vs observed/data/v1.csv"))
data_original = pd.read_csv(os.path.relpath("./graphs/estimate vs observed/data/original.csv"))

data['estimate_original'] = data_original['estimate']
data['date'] = pd.to_datetime(data['date'], unit='ms')

filtered_data = data[data['z'] != 0]

grouped_data = filtered_data.groupby(filtered_data["point"])

plt.figure(figsize=(10, 6))

for group_name, group_data in grouped_data:
    dates = group_data['date']
    estimate = group_data['estimate']
    estimate_original = group_data['estimate_original']
    z = group_data['z']
    intp = group_data['INTP']

    plt.plot(dates, estimate, label='Estimate - Optimized', linestyle='-')
    # plt.plot(dates, estimate_original, label='Estimate - Original', linestyle='solid')
    plt.scatter(dates, z, label='Observed', s=10)

    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.title(f'Kalman Estimate - Point {group_name}')
    plt.legend()

    plt.savefig(f"./graphs/estimate vs observed/graphs/graph_group_{group_name}")
    plt.show()
