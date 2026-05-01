# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:42:11 2026

@author: shambhavic
"""

import pickle
from preprocessing import preprocess_data

def predict_clusters(df):

    # load artifacts
    model = pickle.load(open('models/kmeans_model.pkl','rb'))
    scaler = pickle.load(open('models/scaler.pkl','rb'))

    # preprocess
    user_df = preprocess_data(df)

    # scale
    X_scaled = scaler.transform(user_df.drop(columns=['user_id'], errors='ignore'))

    # predict
    clusters = model.predict(X_scaled)

    user_df['cluster'] = clusters
    user_df = user_df.reset_index()
    return user_df[['user_id', 'cluster']]


if __name__ == "__main__":
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect("clustering.db")
    df = pd.read_sql("SELECT * FROM ecommerce_behavior", conn)
    conn.close()

    result = predict_clusters(df)
    print(result.head())

   