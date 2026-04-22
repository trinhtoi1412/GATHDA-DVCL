import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.DataFrame({
    'herb_id': [3535]*10 + [4615]*10 + [4842]*10,
    'positive_degree': [9465]*10 + [8946]*10 + [8629]*10,
    'rank': list(range(1, 11)) * 3,
    'prob': [0.9992, 0.9990, 0.9981, 0.9972, 0.9954, 0.9954, 0.9954, 0.9952, 0.9951, 0.9950,
             0.9999, 0.9999, 0.9998, 0.9997, 0.9997, 0.9997, 0.9997, 0.9997, 0.9997, 0.9997,
             0.9984, 0.9979, 0.9969, 0.9958, 0.9955, 0.9946, 0.9936, 0.9935, 0.9923, 0.9922]
})

plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='rank', y='prob', hue='herb_id', marker='o', palette='viridis')
plt.title('Prediction Confidence across Top-K Ranks')
plt.xlabel('Rank')
plt.ylabel('Predicted Probability')
plt.xticks(range(1, 11))
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('topk_confidence.png', dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data.groupby('herb_id').first(), x='positive_degree', y='prob', s=100, color='red')
plt.title('Correlation between Node Degree and Max Prediction Probability')
plt.xlabel('Positive Degree (Ground Truth)')
plt.ylabel('Top-1 Probability')
plt.savefig('degree_correlation.png', dpi=300)
plt.show()