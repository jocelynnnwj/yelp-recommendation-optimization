"""
Method Description: 
I designed a hybrid recommender system that combines a latent factor collaborative filtering (CF) model with a boosted tree regressor to predict user ratings on Yelp. The first component is a matrix factorization model, which learns user and business embeddings and bias terms using stochastic gradient descent. This model captures implicit user-business interactions and general rating patterns. 

To improve predictive accuracy, I also extract 27 informative features for each (user, business) pair, including review statistics, business attributes, and user profile signals. These features are used to train an XGBoost regressor, which models complex nonlinear relationships and cold-start cases. 

My final prediction is a weighted combination of the MF and XGBoost predictions, where the blend parameter is chosen for best validation RMSE. Missing or unknown features are imputed with robust defaults or random samples to ensure model coverage.

Key improvements include: 
- Blending MF and XGBoost outputs for stronger generalization
- More robust feature extraction with improved handling of missing data
- Optimized MF and XGBoost hyperparameters for both performance and speed

Error Distribution:
>=0 and <1: 102256    
>=1 and <2: 32831
>=2 and <3: 6149
>=3 and <4: 808
>=4: 0

RMSE: 0.977853    

Execution Time: 800.58  
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from pyspark import SparkContext

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext('local[*]', 'yelp-hybrid')
sc.setLogLevel("ERROR")

random.seed(1234)

# Matrix Factorization & uniquely implemented
class MFRecommender:
    def __init__(self, latent_dim=20, steps=16, learn_rate=0.02, reg_weight=0.05):
        self.k = latent_dim
        self.steps = steps
        self.lr = learn_rate
        self.lmbd = reg_weight

    def fit(self, df, uid_map, iid_map):
        mu = df['stars'].astype(float).mean()
        self.mu = mu
        n_u, n_i = len(uid_map), len(iid_map)
        self.u_bias = np.zeros(n_u)
        self.i_bias = np.zeros(n_i)
        self.Umat = np.random.normal(0, 0.15, (n_u, self.k))
        self.Imat = np.random.normal(0, 0.15, (n_i, self.k))
        u_idx, i_idx = df['user_idx'].values, df['biz_idx'].values
        ratings = df['stars'].astype(float).values
        for _ in range(self.steps):
            for j in range(len(ratings)):
                u, i, r = u_idx[j], i_idx[j], ratings[j]
                pred = mu + self.u_bias[u] + self.i_bias[i] + np.dot(self.Umat[u], self.Imat[i])
                err = r - pred
                self.u_bias[u] += self.lr * (err - self.lmbd * self.u_bias[u])
                self.i_bias[i] += self.lr * (err - self.lmbd * self.i_bias[i])
                self.Umat[u] += self.lr * (err * self.Imat[i] - self.lmbd * self.Umat[u])
                self.Imat[i] += self.lr * (err * self.Umat[u] - self.lmbd * self.Imat[i])

    def predict(self, u_indices, i_indices):
        preds = []
        for u, i in zip(u_indices, i_indices):
            val = self.mu
            if u >= 0: val += self.u_bias[u]
            if i >= 0: val += self.i_bias[i]
            if u >= 0 and i >= 0: val += np.dot(self.Umat[u], self.Imat[i])
            preds.append(np.clip(val, 1.0, 5.0))
        return np.array(preds)

# Business/User Preprocessing
def parse_attrs(at, key, fallback=0):
    if not at: return fallback
    val = at.get(key)
    if val is None: return fallback
    if isinstance(val, str) and val.lower() == 'false': return 0
    try: return int(val)
    except: return 1 if val else 0

def feat_biz(row):
    attr = row.get('attributes', {})
    vals = [
        row.get('stars', 0.0), row.get('review_count', 0),
        row.get('latitude', 0.0), row.get('longitude', 0.0),
        parse_attrs(attr, 'RestaurantsPriceRange2', random.randint(1, 4)),
        parse_attrs(attr, 'BusinessAcceptsCreditCards', random.randint(0, 1)),
        parse_attrs(attr, 'ByAppointmentOnly', 0),
        parse_attrs(attr, 'RestaurantsReservations', 0),
        parse_attrs(attr, 'RestaurantsTableService', 0),
        parse_attrs(attr, 'WheelchairAccessible', 0)
    ]
    return (row['business_id'], vals)

def feat_user(row):
    return (
        row['user_id'],
        [
            row.get('review_count', 0),
            len(str(row.get('friends', 'None')).split(',')) if row.get('friends', 'None') != 'None' else 0,
            row.get('useful', 0), row.get('funny', 0), row.get('cool', 0), row.get('fans', 0),
            len(str(row.get('elite', 'None')).split(',')) if row.get('elite', 'None') != 'None' else 0,
            row.get('average_stars', 0.0),
            row.get('compliment_hot', 0), row.get('compliment_profile', 0), row.get('compliment_list', 0),
            row.get('compliment_note', 0), row.get('compliment_plain', 0), row.get('compliment_cool', 0),
            row.get('compliment_funny', 0), row.get('compliment_writer', 0), row.get('compliment_photos', 0)
        ]
    )

def construct_feature_matrix(df, biz_dict, user_dict, feat_names):
    feats = {key: [] for key in feat_names}
    for _, row in df.iterrows():
        b = biz_dict.get(row['business_id'], [0] * 10)
        u = user_dict.get(row['user_id'], [0] * 17)
        for i in range(10):
            feats[feat_names[i]].append(b[i])
        for i in range(17):
            feats[feat_names[10 + i]].append(u[i])
    for col in feat_names:
        df[col] = feats[col]
    return df

def numericize(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# Main
if __name__ == "__main__":
    t0 = time.time()
    in_dir, test_csv, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    train_path = os.path.join(in_dir, "yelp_train.csv")
    business_path = os.path.join(in_dir, "business.json")
    user_path = os.path.join(in_dir, "user.json")

    train_rdd = sc.textFile(train_path)
    train_header = train_rdd.first()
    train_rows = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(',')).collect()
    train_df = pd.DataFrame(train_rows, columns=train_header.split(','))

    test_rdd = sc.textFile(test_csv)
    test_header = test_rdd.first()
    test_rows = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(',')).collect()
    test_df = pd.DataFrame(test_rows, columns=test_header.split(','))

    user_id_seq = test_df['user_id'].tolist()
    biz_id_seq = test_df['business_id'].tolist()

    business_data = dict(sc.textFile(business_path).map(json.loads).map(feat_biz).collect())
    user_data = dict(sc.textFile(user_path).map(json.loads).map(feat_user).collect())

    feature_names = [
        'biz_stars', 'biz_review_count', 'latitude', 'longitude',
        'price_range', 'credit_card', 'appt_only',
        'reservations', 'table_service', 'wheelchair',
        'user_review_count', 'user_friends', 'user_useful', 'user_funny', 'user_cool', 'user_fans',
        'user_elite', 'user_avg_stars', 'comp_hot', 'comp_profile', 'comp_list', 'comp_note',
        'comp_plain', 'comp_cool', 'comp_funny', 'comp_writer', 'comp_photos'
    ]

    # Features for XGB
    train_df = construct_feature_matrix(train_df, business_data, user_data, feature_names)
    test_df = construct_feature_matrix(test_df, business_data, user_data, feature_names)
    train_df = numericize(train_df, feature_names + ['stars'])
    test_df = numericize(test_df, feature_names)

    X_train = train_df[feature_names].values
    y_train = train_df['stars'].astype(float).values
    X_test = test_df[feature_names].values

    # Index map for MF
    u2idx = {u: i for i, u in enumerate(train_df['user_id'].unique())}
    b2idx = {b: i for i, b in enumerate(train_df['business_id'].unique())}
    train_df['user_idx'] = train_df['user_id'].map(lambda x: u2idx.get(x, -1))
    train_df['biz_idx'] = train_df['business_id'].map(lambda x: b2idx.get(x, -1))

    # Matrix factorization model
    mf = MFRecommender(latent_dim=18, steps=18, learn_rate=0.015, reg_weight=0.04)
    mf.fit(train_df, u2idx, b2idx)

    # MF predictions for test
    test_u_idx = [u2idx.get(u, -1) for u in user_id_seq]
    test_b_idx = [b2idx.get(b, -1) for b in biz_id_seq]
    mf_preds = mf.predict(test_u_idx, test_b_idx)

    # XGBoost model
    booster = xgb.XGBRegressor(learning_rate=0.045, max_depth=5, n_estimators=850, verbosity=0, random_state=1234)
    booster.fit(X_train, y_train)
    xgb_preds = booster.predict(X_test)

    # Hybrid prediction and tuning
    mix_ratio = 0.11
    final_preds = np.clip(mix_ratio * mf_preds + (1 - mix_ratio) * xgb_preds, 1.0, 5.0)

    # Save predictions
    pd.DataFrame({
        "user_id": user_id_seq,
        "business_id": biz_id_seq,
        "prediction": final_preds
    }).to_csv(out_csv, header=['user_id', 'business_id', 'prediction'], index=False)

    # for validation stat
    if 'stars' in test_df.columns:
        y_actual = test_df['stars'].astype(float).values
        diff = np.abs(y_actual - final_preds)
        bins = [0, 1, 2, 3, 4, float('inf')]
        err_counts = [np.sum((diff >= bins[i]) & (diff < bins[i+1])) for i in range(5)]
        tags = [">=0 and <1", ">=1 and <2", ">=2 and <3", ">=3 and <4", ">=4"]
        for t, n in zip(tags, err_counts):
            print(f"{t}: {n}")
        print(f"RMSE: {np.sqrt(np.mean((y_actual - final_preds) ** 2)):.6f}")

    print(f"Run time (s): {time.time() - t0:.2f}")
    sc.stop()