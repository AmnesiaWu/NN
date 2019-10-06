# -*-coding:utf8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
kobe = pd.read_csv('data.csv')
kobe = kobe[pd.notnull(kobe['shot_made_flag'])]# shot_made_flag 不为空的全部数据行
predictors = ['shot_distance', 'shot_type', 'combined_shot_type', 'shot_zone_area', 'period', 'playoffs']
drops = ['action_type', 'game_event_id', 'game_id', 'lat', 'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'season', 'shot_zone_basic', 'shot_zone_range', 'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id', 'seconds_remaining']
for drop in drops:
    kobe.drop(drop, axis=1, inplace=True) # 删除列, inplace = True时在原表操作，=False时返回结果，原表不变
kobe.loc[kobe['shot_type'] == '2PT Field Goal', 'shot_type'] = 0
kobe.loc[kobe['shot_type'] == '3PT Field Goal', 'shot_type'] = 1

kobe.loc[kobe['combined_shot_type'] == 'Jump Shot', 'combined_shot_type'] = 0
kobe.loc[kobe['combined_shot_type'] == 'Dunk', 'combined_shot_type'] = 1
kobe.loc[kobe['combined_shot_type'] == 'Layup', 'combined_shot_type'] = 2
kobe.loc[kobe['combined_shot_type'] == 'Tip Shot', 'combined_shot_type'] = 3
kobe.loc[kobe['combined_shot_type'] == 'Hook Shot', 'combined_shot_type'] = 4
kobe.loc[kobe['combined_shot_type'] == 'Bank Shot', 'combined_shot_type'] = 5

kobe.loc[kobe['shot_zone_area'] == 'Left Side(L)', 'shot_zone_area'] = 0
kobe.loc[kobe['shot_zone_area'] == 'Left Side Center(LC)', 'shot_zone_area'] = 1
kobe.loc[kobe['shot_zone_area'] == 'Right Side Center(RC)', 'shot_zone_area'] = 2
kobe.loc[kobe['shot_zone_area'] == 'Center(C)', 'shot_zone_area'] = 3
kobe.loc[kobe['shot_zone_area'] == 'Right Side(R)', 'shot_zone_area'] = 4
kobe.loc[kobe['shot_zone_area'] == 'Back Court(BC)', 'shot_zone_area'] = 5

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=5, min_samples_leaf=1)
scores = model_selection.cross_val_score(alg, kobe[predictors], kobe['shot_made_flag'], cv=6)
print(scores.mean())
