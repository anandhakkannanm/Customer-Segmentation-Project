import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Cluster Summary
summary = df.groupby('Cluster')[['Age',
                                 'Annual Income (k$)',
                                 'Spending Score (1-100)']].mean()

print("\nCUSTOMER SEGMENT SUMMARY\n")
print(summary)

# Save Result
summary.to_csv("outputs/customer_segment_summary.csv")

print("\nSummary Saved Successfully!")