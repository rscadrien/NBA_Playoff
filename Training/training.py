import pandas as pd
from sklearn.model_selection import train_test_split
from Data_loading_preprocessing.data_loader import load_data
from Data_loading_preprocessing.feature_engineering import create_targets, encode_conference, encode_playoff_results, drop_columns
from Data_loading_preprocessing.preprocessing import scale_features
from Model.model import train_model, save_model
from Training.evaluation import evaluate_model

# --- Load and preprocess data ---
df = load_data('./Data/NBA_data.csv')
df = create_targets(df)
df = encode_conference(df, 'encoder_conference.joblib',mode = 'train')
df = encode_playoff_results(df)
df = drop_columns(df)

# --- Scale features ---
scaling_cols = ['Conf. Seed', 'NBA Seed', 'ORtg Rank', 'DRtg Rank']
df = scale_features(df, scaling_cols, 'scaler_seed_rank.joblib', mode = 'train')

# --- Split data ---
X = df.drop(['Champion', 'Finalist', 'Conf_Finalist', 'Conf_SemiFinalist'], axis=1)
y = df[['Champion', 'Finalist', 'Conf_Finalist', 'Conf_SemiFinalist']][['Conf_SemiFinalist', 'Conf_Finalist', 'Finalist', 'Champion']]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- Train and save model ---
model = train_model(X, y)
save_model(model, 'NBA.joblib')

# --- Evaluate model by cross-validation
evaluate_model(model, X, y, ['Semifinalist', 'Conf Finalist', 'Finalist', 'Champion'])