# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:44:25 2026

@author: shambhavic
"""

# app.py

import streamlit as st
import pandas as pd
import sqlite3
from predict import predict_clusters

st.title("Customer Segmentation")

# load data from DB
conn = sqlite3.connect("clustering.db")
df = pd.read_sql("SELECT * FROM ecommerce_behavior", conn)
conn.close()

if st.button("Run Clustering"):
    result = predict_clusters(df)
    st.write(result.head())