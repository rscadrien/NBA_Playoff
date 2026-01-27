import joblib
import pandas as pd
import numpy as np
from Data_loading_preprocessing.feature_engineering import encode_conference, encode_playoff_results
from Data_loading_preprocessing.preprocessing import scale_features
from Inference.valid_input import get_valid_input_str, get_valid_input_seed, get_valid_input_record

#Ask for Input
Conference = get_valid_input_str("What is the conference of the team? (Est/West): ", ['East', 'West'])
Season_record = get_valid_input_record("What is the season record of the team?")
Conf_seed = get_valid_input_seed('What is the seed of the team in its conference?',15)
NBA_seed = get_valid_input_seed('What is the seed of the team in all the NBA?',30)
record_20 = get_valid_input_record('What is the record of the team in the last 20 games of the regular season?')
ORtg_ranking = get_valid_input_seed('What is the ranking of the team in the offensive rating?',30)
DRtg_ranking = get_valid_input_seed('What is the ranking of the team in the defensive rating?',30)
playoff_options = ['No Playoff', 'First Round','Conference Semi-Final', 'Conference Final','NBA Final', 'NBA Champion']
result_2_season = get_valid_input_str('What was the result of the team 2 season ago?',playoff_options)
result_last_season = get_valid_input_str('What was the result of the team last season?',playoff_options)

#Create a DataFrame with the Input
X=pd.DataFrame([{
    'Conference' : Conference,
    'Season record' : Season_record,
    'Conf. Seed' : Conf_seed,
    'NBA Seed' : NBA_seed,
    'Last 20 Games record' : record_20,
    'ORtg Rank' : ORtg_ranking,
    'DRtg Rank' : DRtg_ranking,
    '2 seasons ago result' : result_2_season,
    'Last season result' : result_last_season
}])

#Preprocessing the data 
X = encode_conference(X, 'encoder_conference.joblib',mode = 'eval')
X = encode_playoff_results(X)
scaling_cols = ['Conf. Seed', 'NBA Seed', 'ORtg Rank', 'DRtg Rank']
X = scale_features(X, scaling_cols, 'scaler_seed_rank.joblib', mode = 'eval')

#Load the pretrained model
model = joblib.load('NBA.joblib')
#Apply the model
y=model.predict(X)
y_clipped = np.clip(y, 0, 1)
#Printing the result
print(f'The playoff strength is: {y_clipped:.2f}')
