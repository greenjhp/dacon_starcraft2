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


# from tqdm import tqdm                       # 진행바
from sklearn.model_selection import KFold   # K-fold CV
from functools import partial


# data load
data_folder = 'model_data/'

train_final_ftr_df = pd.read_csv(os.path.join(data_folder,'train_final_ftr_0414_3.csv'))
test_final_ftr_df = pd.read_csv(os.path.join(data_folder,'train_final_ftr_0414_3.csv'))

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


# data set split
# df = train_final_ftr_df[:]

# train_data, test_data = train_test_split(df, test_size=0.2, random_state=1)
# train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=2)

# train_data, valid_data = train_test_split(df, test_size=0.2, random_state=2)

# print('train_data: ', train_data.shape)
# print('valid_data: ', valid_data.shape)
# print('test_data: ', test_data.shape)

# _ftr = train_final_ftr_df.columns.tolist()
# _ftr.remove('game_id')
# _ftr.remove('winner')

# X_train, y_train = train_data[_ftr], train_data['winner']
# X_valid, y_valid = valid_data[_ftr], valid_data['winner']
# X_test, y_test = test_data[_ftr], test_data['winner']

# X_train, y_train = train_data[_ftr], train_data['winner']
# X_valid, y_valid = valid_data[_ftr], valid_data['winner']

# test_final_ftr_df = test_final_ftr_df[_ftr]


# modeling

## 전체 데이터 가져오기
_ftr = train_final_ftr_df.columns.tolist()
_ftr.remove('game_id')
_ftr.remove('winner')

X_train, y_train = train_final_ftr_df[_ftr], train_final_ftr_df['winner']
#
#
# def lgb_cv(num_iterations, learning_rate, num_leaves,
#            feature_fraction, bagging_fraction, max_depth,
#            lambda_l1, lambda_l2,
#            X=X_train, y=y_train
#            ):
#
#     lgb_train_data = lgb.Dataset(X, label=y)
#
#     params = {'boosting_type': 'goss',
#               # 'early_stopping_round':100,
#               'objective': 'binary',
#               'metrics': 'auc'
#              }
#
#     params['num_iterations'] = int(round(num_iterations)),
#     params['learning_rate'] = learning_rate,
#     params['num_leaves'] = int(round(num_leaves)),
#     params['feature_fraction'] = np.clip(feature_fraction, 0, 1),
#     params['bagging_fraction'] = np.clip(bagging_fraction, 0, 1),
#     params['max_depth'] = int(round(max_depth)),
#     params['lambda_l1'] = max(lambda_l1, 0),
#     params['lambda_l2'] = max(lambda_l2, 0),
#
#     print(params)
#
#     cv_result = lgb.cv(params, lgb_train_data,
#                        nfold=5, seed=4321,
#                        # stratified=True, verbose_eval =200,
#                        # feval=lgb_f1_score
#                       )
#     return max(cv_result['auc-mean'])
#
#
# # 베이지안 최적화 범위 설정
# lgbBO = BayesianOptimization(
#     lgb_cv,
#
#     {
#         'num_iterations': (10, 10000),  # num_iterations,     범위(16~1024)
#         'learning_rate': (0.0001, 0.1),  # learning_rate,    범위(0.0001~0.1)
#         'num_leaves': (24, 100),
#         'feature_fraction': (0, 1),
#         'bagging_fraction': (0, 1),
#         'max_depth': (5, 20),
#         'lambda_l1': (0, 10),  # lambda_l1,       범위(0~10)
#         'lambda_l2': (0, 50),  # lambda_l2,       범위(0~50)
#      },
#
#     random_state=4321                    # 시드 고정
# )
# lgbBO.maximize(init_points=5, n_iter=10) # 처음 5회 랜덤 값으로 score 계산 후 30회 최적화


### cv 수동 구현
def lgb_cv(num_iterations, learning_rate, num_leaves,
           feature_fraction, bagging_fraction,
           lambda_l1, lambda_l2,
           min_data_in_leaf,
           x_data=None, y_data=None, n_splits=5, output='score'): # learning_rate, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda,
    score = 0
    kf = KFold(n_splits=n_splits)
    models = []
    for train_index, valid_index in kf.split(x_data):
        x_train, y_train = x_data.iloc[train_index], y_data[train_index]
        x_valid, y_valid = x_data.iloc[valid_index], y_data[valid_index]

        model = lgb.LGBMClassifier(
            boosting_type='dart',
            n_jobs=4,
            objective='binary',

            num_iterations=int(num_iterations),
            learning_rate=learning_rate,
            num_leaves=int(num_leaves),
            feature_fraction=np.clip(feature_fraction, 0, 1),
            bagging_fraction=np.clip(bagging_fraction, 0, 1),
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            min_data_in_leaf=int(min_data_in_leaf)
        )
        model.fit(x_train, y_train, verbose=-1)
        models.append(model)

        pred = model.predict_proba(x_valid)[:, 1]
        true = y_valid
        score += roc_auc_score(true, pred) / n_splits

    if output == 'score':
        return score
    if output == 'model':
        return models


# 모델과 관련없는 변수 고정
func_fixed = partial(lgb_cv, x_data=X_train, y_data=y_train, n_splits=5, output='score')
# 베이지안 최적화 범위 설정
lgbBO = BayesianOptimization(
    func_fixed,

    {
        'num_iterations': (10, 1000),  # num_iterations,     범위(16~1024)
        'learning_rate': (0.001, 0.1),  # learning_rate,    범위(0.0001~0.1)
        'num_leaves': (16, 1024),
        'feature_fraction': (0.5, 1),
        'bagging_fraction': (0.8, 1),
        'lambda_l1': (0, 10),  # lambda_l1,       범위(0~10)
        'lambda_l2': (0, 50),  # lambda_l2,       범위(0~50)
        'min_data_in_leaf': (5, 50)
     },

    # {   'num_leaves': (24, 45),
    #     'feature_fraction': (0.5, 0.9),
    #     'bagging_fraction': (0.8, 1),
    #     'max_depth': (5, 9),
    # },

    # {
    #     'num_leaves': (16, 1024),        # num_leaves,       범위(16~1024)
    #     'learning_rate': (0.0001, 0.1),  # learning_rate,    범위(0.0001~0.1)
    #     'n_estimators': (16, 1024),      # n_estimators,     범위(16~1024)
    #     'subsample': (0, 1),             # subsample,        범위(0~1)
    #     'colsample_bytree': (0, 1),      # colsample_bytree, 범위(0~1)
    #     'reg_alpha': (0, 10),            # reg_alpha,        범위(0~10)
    #     'reg_lambda': (0, 50),           # reg_lambda,       범위(0~50)
    # },
    random_state=4321                    # 시드 고정
)
lgbBO.maximize(init_points=3, n_iter=5) # 처음 5회 랜덤 값으로 score 계산 후 30회 최적화
#
# # 이 예제에서는 7개 하이퍼 파라미터에 대해 30회 조정을 시도했습니다.
# # 다양한 하이퍼 파라미터, 더 많은 iteration을 시도하여 최상의 모델을 얻어보세요!
# # LightGBM Classifier: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html

print(lgbBO.max)

## LGBM Model

### cross validation
def cross_validation(X=X_train, y=y_train):
    lgb_train_data = lgb.Dataset(X, label=y)
    # lgb_valid_data = lgb.Dataset(X, label=y)

    params = {'boosting_type': 'dart', # gbdt, dart, goss
             'objective': 'binary',
             'learning_rate' : 0.005,
    #          'max_depth' : 20,
    #          'feature_fraction' : 0.8,
    #          'scale_pos_weight' : 1.1,
             'metrics' : 'auc',
             'verbosity':0}

    bst = lgb.cv(params, lgb_train_data,
    #                valid_sets=[lgb_valid_data],
                   num_boost_round = 10000,
                   early_stopping_rounds=500,
    #                categorical_feature=['map_0','map_1','species0_T','species0_P','species0_Z','species1_T','species1_P','species1_Z'],
                   categorical_feature=['map_0','map_1','species_0','species_1'],
    #                categorical_feature=name:map_0,map_1,species_0,species_1,
                   verbose_eval=False)

    return bst

# plt.plot(cross_validation(X_train, y_train)['auc-mean'])
# plt.plot(cross_validation(X_train, y_train)['auc-stdv'])



# ### Bayesian Optimizer
# lgb_train_data = lgb.Dataset(X_train, label=y_train)
#
# def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth):
#     # lgb_valid_data = lgb.Dataset(X_valid, label=y_valid)
#
#     params = {'boosting_type': 'dart',  # gbdt, dart, goss
#               'objective': 'binary',
#               'learning_rate': 0.05,
#               'num_boost_round': 500,
#               #              'scale_pos_weight' : 1.1,
#               'metrics': 'auc',
#               'verbosity': 0}
#
#     params['num_leaves'] = int(round(num_leaves))
#     params['feature_fraction'] = max(min(feature_fraction, 1), 0)
#     params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
#     params['max_depth'] = int(round(max_depth))
#
#     bst = lgb.cv(params,
#                  lgb_train_data,
#                  categorical_feature=['map_0', 'map_1', 'species_0', 'species_1']
#                  )
#
#     return bst['auc-mean']
#
# # print(lgb_eval(24,0.5,0.6,5))
#
# lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
#                                         'feature_fraction': (0.1, 0.9),
#                                         'bagging_fraction': (0.8, 1),
#                                         'max_depth': (5, 8.99)}, random_state=0)
#
# lgbBO.maximize(n_iter=5)




# ### train
# lgb_train_data = lgb.Dataset(X_train, label=y_train)
# # lgb_valid_data = lgb.Dataset(X_valid, label=y_valid)
#
# params = {'boosting_type': 'dart', # gbdt, dart, goss
#          'objective': 'binary',
#          'learning_rate' : 0.005,
# #          'max_depth' : 20,
# #          'feature_fraction' : 0.8,
# #          'scale_pos_weight' : 1.1,
#          'metrics' : 'auc',
#          'verbosity':0}
# bst = lgb.train(params, lgb_train_data,
# #                valid_sets=[lgb_valid_data],
#                num_boost_round = 10000,
#                early_stopping_rounds=500,
# #                categorical_feature=['map_0','map_1','species0_T','species0_P','species0_Z','species1_T','species1_P','species1_Z'],
#                categorical_feature=['map_0','map_1','species_0','species_1'],
#                verbose_eval=False)
#
# # predict_test = bst.predict(X_valid)
#
# # auc = roc_auc_score(y_valid, predict_test)
# # print(auc)
#
# # bst.save_model(os.path.join(data_folder,'baseline_0303.model'))



# # #### > test_df에서 누락된 game_id 추가 (null인 game_id의 winner는 0.5로)
#
# # ##### Predict
#
# # In[206]:
#
#
# _ftr = test_final_ftr_df.columns.tolist()
# _ftr.remove('game_id')
#
# # In[207]:
#
#
# predict_test = bst.predict(test_final_ftr_df[_ftr])
#
# # In[208]:
#
#
# predict_test
#
# # In[209]:
#
#
# print('[n_of_game_id]')
# # print('train_df: ', len(train_df.game_id.unique()))
# # print('train_ability_feature_df: ', len(train_final_ftr_df.game_id.unique()))
# # print('test_df: ', len(test_df.game_id.unique()))
# print('predict_test: ', len(predict_test))
#
# # In[210]:
#
#
# predict_test_df = pd.DataFrame(predict_test).reset_index().rename(columns={0: 'winner'})
#
# # In[211]:
#
#
# test_game_id_df = test_final_ftr_df[['game_id']].reset_index()
#
# # In[212]:
#
#
# # test_game_id_df
#
#
# # In[213]:
#
#
# predict_test_small_df = test_game_id_df.merge(predict_test_df, how='inner', on='index').drop('index', axis='columns')
#
# # In[214]:
#
#
# # predict_test_small_df
#
#
# # In[215]:
#
#
# print('[n_of_game_id]')
# # print('train_df: ', len(train_df.game_id.unique()))
# # print('train_ability_feature_df: ', len(train_final_ftr_df.game_id.unique()))
# # print('test_df: ', len(test_df.game_id.unique()))
# print('predict_test_small_df: ', len(predict_test_small_df))
#
# # In[216]:
#
#
# original_test_game_id_df = pd.DataFrame(test_df.game_id.unique(), columns=['game_id'])
#
# # In[217]:
#
#
# # original_test_game_id_df
#
#
# # In[218]:
#
#
# result_df = original_test_game_id_df.merge(predict_test_small_df, how='left', on='game_id').fillna(0.5)
#
# # In[219]:
#
#
# result_df.head()
#
# # In[220]:
#
#
# result_df.shape
#
# # In[221]:
#
#
# result_df[result_df['winner'] == 0.5]
#
# # ## 4.2. Random Forest Model
#
# # In[158]:
#
#
# df = train_final_ftr_df
#
# # train_data, test_data = train_test_split(df, test_size=0.2, random_state=1)
# # train_data, valid_data = train_test_split(train_data, test_size=0.4, random_state=2)
#
# train_data, valid_data = train_test_split(df, test_size=0.3, random_state=2)
#
# print('train_data: ', train_data.shape)
# print('valid_data: ', valid_data.shape)
# # print('test_data: ',test_data.shape)
#
#
# # In[463]:
#
#
# _ftr = train_final_ftr_df.columns.tolist()
# _ftr.remove('game_id')
# _ftr.remove('winner')
#
# # In[464]:
#
#
# len(_ftr)
#
# # In[465]:
#
#
# # X_train, y_train = train_data[_ftr], train_data['winner']
# # X_valid, y_valid = valid_data[_ftr], valid_data['winner']
# # X_test, y_test = test_data[_ftr], test_data['winner']
#
# X_train, y_train = train_data[_ftr], train_data['winner']
# X_valid, y_valid = valid_data[_ftr], valid_data['winner']
#
# # ### 4.2.1. Learning
#
# # ##### 최적 max_depth 탐색
#
# # In[466]:
#
#
# get_ipython().run_cell_magic('time', '',
#                              'x_list = []\ntrain_score_list = []\nvalid_score_list = []\n\nfor i in range(1,51):\n    clf = RandomForestClassifier(max_depth=i, random_state=0)\n    clf = clf.fit(X_train, y_train)\n    \n    x_list.append(i)\n    train_score_list.append(clf.score(X_train, y_train))\n    valid_score_list.append(clf.score(X_valid, y_valid))')
#
# # In[467]:
#
#
# plt.plot(x_list, train_score_list)
#
# # In[468]:
#
#
# plt.plot(x_list, valid_score_list)
#
# # ##### 최적 max_depth로 학습
#
# # 전체 데이터 가져오기
#
# # In[469]:
#
#
# _ftr = train_final_ftr_df.columns.tolist()
# _ftr.remove('game_id')
# _ftr.remove('winner')
#
# # In[470]:
#
#
# X_train, y_train = train_final_ftr_df[_ftr], train_final_ftr_df['winner']
#
# # 전체 데이터로 학습
#
# # In[471]:
#
#
# clf = RandomForestClassifier(max_depth=14, random_state=0)
# clf = clf.fit(X_train, y_train)
#
# # In[472]:
#
#
# clf.score(X_train, y_train)
#
# # ### 4.2.2. Predict
#
# # In[483]:
#
#
# _ftr = test_final_ftr_df.columns.tolist()
# _ftr.remove('game_id')
#
# # In[484]:
#
#
# predict_test = clf.predict(test_final_ftr_df[_ftr])
#
# # In[485]:
#
#
# predict_test
#
# # In[527]:
#
#
# print('[n_of_game_id]')
# # print('train_df: ', len(train_df.game_id.unique()))
# # print('train_ability_feature_df: ', len(train_final_ftr_df.game_id.unique()))
# print('test_df: ', len(test_df.game_id.unique()))
# print('predict_test: ', len(predict_test))
#
# # In[489]:
#
#
# predict_test_df = pd.DataFrame(predict_test).reset_index().rename(columns={0: 'winner'})
#
# # In[490]:
#
#
# # predict_test_df
#
#
# # In[491]:
#
#
# test_game_id_df = test_final_ftr_df[['game_id']].reset_index()
#
# # In[492]:
#
#
# # test_game_id_df
#
#
# # In[493]:
#
#
# predict_test_small_df = test_game_id_df.merge(predict_test_df, how='inner', on='index').drop('index', axis='columns')
#
# # In[494]:
#
#
# # predict_test_small_df
#
#
# # In[496]:
#
#
# original_test_game_id_df = pd.DataFrame(test_df.game_id.unique(), columns=['game_id'])
#
# # In[497]:
#
#
# # original_test_game_id_df
#
#
# # In[498]:
#
#
# result_df = original_test_game_id_df.merge(predict_test_small_df, how='left', on='game_id').fillna(0.5)
#
# # In[179]:
#
#
# result_df.head()
#
# # In[180]:
#
#
# result_df.shape
#
# # In[222]:
#
#
# plt.hist(result_df.winner)
#
# # # Output
#
# # In[223]:
#
#
# result_df.to_csv('submission_baseline_0324_2.csv', index=False)