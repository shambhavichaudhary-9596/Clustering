# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:23:23 2026

@author: shambhavic
"""

import pandas as pd
import pickle
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from preprocessing import preprocess_data

# load data
conn = sqlite3.connect("clustering.db")
df = pd.read_sql("SELECT * FROM ecommerce_behavior", conn)
conn.close()


# preprocess
user_df = preprocess_data(df)

# drop user_id if present
X = user_df.copy()
X = X.drop(columns=['user_id'], errors='ignore')

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# model
km = KMeans(n_clusters=3, random_state=42, n_init='auto')
km.fit(X_scaled)

# save
import os
os.makedirs('models', exist_ok=True)
pickle.dump(km, open('models/kmeans_model.pkl','wb'))
pickle.dump(scaler, open('models/scaler.pkl','wb'))

print("Training complete!")