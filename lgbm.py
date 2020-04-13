import os

import warnings as warnings
warnings.filterwarnings(action='ignore')

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', -1)
pd.options.display.float_format = '{:.2f}'.format # 소수점 유효숫자 표기법으로 안쓰기

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from bayes_opt import BayesianOptimization

import matplotlib.pyplot as plt

# from tqdm import tqdm                       # 진행바
from sklearn.model_selection import KFold   # K-fold CV
from functools import partial


# data load
data_folder = 'model_data/'

train_final_ftr_df = pd.read_csv(os.path.join(data_folder,'train_final_ftr_0413.csv'))
test_final_ftr_df = pd.read_csv(os.path.join(data_folder,'test_final_ftr_0413.csv'))

train_final_ftr_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in train_final_ftr_df.columns]
test_final_ftr_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in test_final_ftr_df.columns]

train_final_ftr_df['map_0'] = train_final_ftr_df['map_0'].astype('int').astype('category')
train_final_ftr_df['map_1'] = train_final_ftr_df['map_1'].astype('int').astype('category')
train_final_ftr_df['species_0'] = train_final_ftr_df['species_0'].astype('int').astype('category')
train_final_ftr_df['species_1'] = train_final_ftr_df['species_1'].astype('int').astype('category')

test_final_ftr_df['map_0'] = test_final_ftr_df['map_0'].astype('int').astype('category')
test_final_ftr_df['map_1'] = test_final_ftr_df['map_1'].astype('int').astype('category')
test_final_ftr_df['species_0'] = test_final_ftr_df['species_0'].astype('int').astype('category')
test_final_ftr_df['species_1'] = test_final_ftr_df['species_1'].astype('int').astype('category')



## 전체 데이터 가져오기
_ftr = train_final_ftr_df.columns.tolist()
_ftr.remove('game_id')
_ftr.remove('winner')

X_train, y_train = train_final_ftr_df[_ftr], train_final_ftr_df['winner']

### train
lgb_train_data = lgb.Dataset(X_train, label=y_train)
# lgb_valid_data = lgb.Dataset(X_valid, label=y_valid)

# params = {'boosting_type': 'dart', # gbdt, dart, goss
#          'objective': 'binary',
#          'learning_rate' : 0.005,
# #          'max_depth' : 20,
# #          'feature_fraction' : 0.8,
# #          'scale_pos_weight' : 1.1,
#          'metrics' : 'auc',
#          'verbosity':0}

# |   iter    |  target   | baggin... | featur... | lambda_l1 | lambda_l2 | learni... | max_depth | num_it... | num_le... |
# |  298      |  0.6741   |  0.6347   |  0.4961   |  6.454    |  0.108    |  0.004814 |  15.61    |  7.859e+0 |  26.43    |

# params = {
#     'num_leaves': 26,        # num_leaves,       범위(16~1024)
#     'learning_rate': 0.004814,  # learning_rate,    범위(0.0001~0.1)
#     'num_iterations': 7859,      # n_estimators,     범위(16~1024)
#     'bagging_fraction': 0.6347,             # subsample,        범위(0~1)
#     'feature_fraction': 0.4961,      # colsample_bytree, 범위(0~1)
#     'lambda_l1': 6.454,            # reg_alpha,        범위(0~10)
#     'lambda_l2': 0.108,           # reg_lambda,       범위(0~50)
# }



# |   iter    |  target   | baggin... | featur... | lambda_l1 | lambda_l2 | learni... | num_it... | num_le... |
# |  209      |  0.675    |  0.5809   |  0.5207   |  9.459    |  48.93    |  0.05153  |  751.5    |  62.99    |

# {'target': 0.6768743549172733, 'params': {'bagging_fraction': 1.0, 'feature_fraction': 0.713359337126435, 'lambda_l1': 9.225099197025434, 'lambda_l2': 49.43242484634145, 'learning_rate': 0.06242146704630103, 'num_iterations': 774.4107327332043, 'num_leaves': 90.66078188853349}}

params = {
    'boosting_type': 'dart',
    'n_jobs': 4,
    'objective': 'binary',
    'metrics': 'auc',

    'bagging_fraction': 1.0,
    'feature_fraction': 0.713359337126435,
    'lambda_l1': 9.225099197025434,
    'lambda_l2': 49.43242484634145,
    'learning_rate': 0.06242146704630103,
    'num_iterations': 774,
    'num_leaves': 91
}


# 0.6833
# params = {
#     'num_leaves': 19,        # num_leaves,       범위(16~1024)
#     'learning_rate': 0.02794,  # learning_rate,    범위(0.0001~0.1)
#     'n_estimators': 724,      # n_estimators,     범위(16~1024)
#     'subsample': 0.8579,             # subsample,        범위(0~1)
#     'colsample_bytree': 0.8448,      # colsample_bytree, 범위(0~1)
#     'reg_alpha': 2.816,            # reg_alpha,        범위(0~10)
#     'reg_lambda': 1.524,           # reg_lambda,       범위(0~50)
# }

bst = lgb.train(params, lgb_train_data,
#                valid_sets=[lgb_valid_data],
#                num_boost_round = 10000,
#                early_stopping_rounds=500,
#                categorical_feature=['map_0','map_1','species0_T','species0_P','species0_Z','species1_T','species1_P','species1_Z'],
               categorical_feature=['map_0','map_1','species_0','species_1'],
               verbose_eval=100)




# bst.save_model(os.path.join(data_folder,'baseline_0303.model'))



#### > test_df에서 누락된 game_id 추가 (null인 game_id의 winner는 0.5로)

_ftr = test_final_ftr_df.columns.tolist()
_ftr.remove('game_id')

predict_test = bst.predict(test_final_ftr_df[_ftr])


# print('[n_of_game_id]')
# print('train_df: ', len(train_df.game_id.unique()))
# print('train_ability_feature_df: ', len(train_final_ftr_df.game_id.unique()))
# print('test_df: ', len(test_df.game_id.unique()))
# print('predict_test: ', len(predict_test))

predict_test_df = pd.DataFrame(predict_test).reset_index().rename(columns={0: 'winner'})

test_game_id_df = test_final_ftr_df[['game_id']].reset_index()

predict_test_small_df = test_game_id_df.merge(predict_test_df, how='inner', on='index').drop('index', axis='columns')

# print('[n_of_game_id]')
# print('train_df: ', len(train_df.game_id.unique()))
# print('train_ability_feature_df: ', len(train_final_ftr_df.game_id.unique()))
# print('test_df: ', len(test_df.game_id.unique()))
# print('predict_test_small_df: ', len(predict_test_small_df))

###
test_df = pd.read_csv('data/sample_submission.csv')
###

original_test_game_id_df = pd.DataFrame(test_df.game_id.unique(), columns=['game_id'])

result_df = original_test_game_id_df.merge(predict_test_small_df, how='left', on='game_id').fillna(0.5)


result_df.head()

plt.hist(result_df.winner)

 # Output

result_df.to_csv('submission_baseline_0413_2.csv', index=False)