# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:03:40 2026

@author: shambhavic
"""

import pandas as pd
import numpy as np

def preprocess_data(df):
    
    # ---------- Basic cleaning ----------
    df = df.drop_duplicates()
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # ----------Handling Missing Values -------
    # missing handling
    df['category_code'] = df['category_code'].fillna('unknown')
    df['brand'] = df['brand'].fillna('unknown')

    # drop critical missing
    df = df.dropna(subset=['user_session'])
    
    # ---------- Feature Engineering ----------
    user_df = df.groupby(['user_id', 'event_type']).size().unstack(fill_value=0)

    user_df = user_df.rename(columns={
        'view': 'total_view',
        'cart': 'total_cart',
        'purchase': 'total_purchase'
    })

    # total events
    user_df['total_events'] = user_df[['total_view','total_cart','total_purchase']].sum(axis=1)

    # avg price
    df_price = df.groupby('user_id')['price'].mean().rename('avg_price')

    # sessions
    df_session = df.groupby('user_id')['user_session'].nunique().rename('total_session')

    # unique features
    df_cat = df.groupby('user_id')['category_code'].nunique().rename('unique_category')
    df_brand = df.groupby('user_id')['brand'].nunique().rename('unique_brand')

    # merge all
    user_df = user_df.join([df_price, df_session, df_cat, df_brand])

    # ---------- Ratios ----------
    user_df['purchase_rate'] = user_df['total_purchase'] / user_df['total_events']
    user_df['cart_rate'] = user_df['total_cart'] / user_df['total_events']

    user_df['cart_to_purchase_ratio'] = (
        user_df['total_purchase'] / user_df['total_cart']
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # ---------- Log Transform ----------
    cols = ['total_cart','total_purchase','total_view','total_events',
            'avg_price','unique_category','unique_brand']

    for col in cols:
        user_df[col] = np.log1p(user_df[col])

    return user_df