import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_original = pd.read_csv('rewards_2.csv')
results_custom = pd.read_csv('rewards_custom_fix.csv')

results_original['Moving_average'] = results_original['Reward'].rolling(window=2500).mean()
results_custom = results_custom[results_custom['Level'] == '1-1']
window_size = max(1, len(results_custom) // (len(results_original) // 2500))
results_custom['Moving_average'] = results_custom['Reward'].rolling(window=window_size).mean()

results_merged = pd.merge(results_original[['Episode', 'Moving_average']], results_custom[['Episode', 'Moving_average']], on='Episode', suffixes=('_First', '_Second'))

plt.figure(figsize=(10, 6))
plt.plot(results_merged['Episode'], results_merged['Moving_average_First'], label='Moving Average Agent1 (Original Level)', color='red')
plt.plot(results_merged['Episode'], results_merged['Moving_average_Second'], label='Moving Average Agent2 (Original + Custom Levels)', color='blue')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards achieved on Level 1-1 during training (Moving Average)')
plt.legend()
plt.show()