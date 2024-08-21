import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_original = pd.read_csv('rewards_2.csv')
results_custom = pd.read_csv('rewards_custom_fix.csv')

results_custom = results_custom[results_custom['Level'] == '1-1']

results_merged = pd.merge(results_original[['Episode', 'Reward']], results_custom[['Episode', 'Reward']], on='Episode', suffixes=('_First', '_Second'))

# line of best fit for original model
best_fit_original = np.polyfit(results_merged['Episode'], results_merged['Reward_First'], 1)
best_fit_org_function = np.poly1d(best_fit_original)

# line of best fit for custom model
best_fit_custom = np.polyfit(results_merged['Episode'], results_merged['Reward_Second'], 1)
best_fit_ctm_function = np.poly1d(best_fit_custom)

plt.figure(figsize=(10, 6))
plt.plot(results_merged['Episode'], best_fit_org_function(results_merged['Episode']), label='Trend Line Agent 1(Original Level)', color='red', linewidth=2)
plt.plot(results_merged['Episode'], best_fit_ctm_function(results_merged['Episode']), label='Trend Line Agent 2 (Original + Custom Levels)', color='blue', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Trend Line for Reward vs Episode on Level 1-1')
plt.legend()
plt.show()