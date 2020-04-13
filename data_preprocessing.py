import os

import warnings as warnings
warnings.filterwarnings(action='ignore')

import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', -1)
pd.options.display.float_format = '{:.2f}'.format # 소수점 유효숫자 표기법으로 안쓰기

import re


data_folder = 'data/'
output_folder = 'model_data/'


# raw data
# train_df = pd.read_csv(os.path.join(data_folder,'train.csv'))
# test_df = pd.read_csv(os.path.join(data_folder,'test.csv'))


# ability ftr
train_ability_df = pd.read_csv(os.path.join(data_folder,'train_Ability.csv'))
test_ability_df = pd.read_csv(os.path.join(data_folder,'test_Ability.csv'))

print(train_ability_df.game_id.min())
print(train_ability_df.game_id.max())

print(test_ability_df.game_id.min())
print(test_ability_df.game_id.max())


# unit supply & cost ftr
unit_cmp_dict = {'Train':'Train[A-Za-z]*',
                 'Build':'BuildSiegeTank|BuildWidowMine|BuildHellion|BuildThor',
                 'WarpIn':'WarpIn[A-Za-z]*',
                 'WarpSelection':'[A-Za-z]*WarpSelection',
                 'UpgradeTo':'UpgradeToMothership',
                 'Morph':'Morph[A-Za-z]*'}

def add_units(df:pd.DataFrame):
    for k,v in unit_cmp_dict.items():
        cmp = re.compile(v)
        units = df.event_contents.map(cmp.findall).map(lambda x: str(x).replace("[","").replace("]","").replace("'","").replace(" ","").replace("To","").replace("GreaterSpire","").replace("TransportOverlord","").replace(k,""))
        df[k.lower()] = units
        
    return df

train_ability_df = add_units(train_ability_df)
test_ability_df = add_units(test_ability_df)


def unit_column_setting(temp_df:pd.DataFrame):
    temp_df['unit'] = temp_df['train'] + temp_df['build'] + temp_df['warpin'] + temp_df['warpselection'] + temp_df['upgradeto'] + temp_df['morph']
    temp_df = temp_df.drop(['train', 'build', 'warpin', 'warpselection', 'upgradeto', 'morph'], axis='columns')
    
    return temp_df

train_ability_df = unit_column_setting(train_ability_df)
test_ability_df = unit_column_setting(test_ability_df)


###### 유닛 인구수, 자원 정보 붙이기
unit_info_df = pd.read_csv(os.path.join(data_folder, 'unit_info_data.csv'))

train_ability_df = train_ability_df.merge(unit_info_df, how='left', on=['unit', 'species'])
test_ability_df = test_ability_df.merge(unit_info_df, how='left', on=['unit', 'species'])

# del train_df
# del test_df


# duration ftr
def make_duration_feature(df: pd.DataFrame):
    duration_df = df[['game_id', 'time']].groupby(['game_id']).max() - df[['game_id', 'time']].groupby(['game_id']).min()
    duration_df = duration_df.reset_index().rename(columns={'time':'duration'})
    
    return duration_df

def add_duration_feature(df: pd.DataFrame):
    duration_df = make_duration_feature(df)
    df = df.merge(duration_df, how='inner', on=['game_id'])
    
    return df

train_ability_feature_df = add_duration_feature(train_ability_df)
test_ability_feature_df = add_duration_feature(test_ability_df)


# train, build 등 ability 관련
def add_ability_feature(df: pd.DataFrame):
    if 'winner' in df.columns:
        data_set_type = 'train'
    else:
        data_set_type = 'test'
    
    df['event_contents_1'] = df.event_contents.map(lambda x: x.split(';')[0])
    df['event_contents_1_name'] = df.event_contents_1.map(lambda x: x.split('- ')[-1])

    df['event_contents_1_train'] =  df['event_contents_1_name'].str.contains('Train')
    df['event_contents_1_warp'] =  df['event_contents_1_name'].str.contains('Warp')
    df['event_contents_1_morph'] =  df['event_contents_1_name'].str.contains('Morph')
    
    df['event_contents_1_build'] =  df['event_contents_1_name'].str.contains('Build')
    
    df['event_contents_1_upgrade'] =  df['event_contents_1_name'].str.contains('Upgrade')
    df['event_contents_1_research'] =  df['event_contents_1_name'].str.contains('Research')
    df['event_contents_1_evlove'] =  df['event_contents_1_name'].str.contains('Evlove')
    
    df['event_contents_1_attack'] =  df['event_contents_1_name'].str.contains('Attack')

    # train, test set 구분
    df_sum_col_list = ['game_id', 'winner', 'player', 'species', 'duration',
                 'event_contents_1_train', 'event_contents_1_warp', 'event_contents_1_morph', 
                 'event_contents_1_build', 
                 'event_contents_1_upgrade', 'event_contents_1_research', 'event_contents_1_evlove', 
                 'event_contents_1_attack']
    df_sum_groupby_col_list = ['game_id', 'winner', 'player', 'species', 'duration']
    if data_set_type == 'test':
        df_sum_col_list.remove('winner')
        df_sum_groupby_col_list.remove('winner')
    df_sum = df[df_sum_col_list].groupby(df_sum_groupby_col_list).sum()

    df_sum['train_sum'] = df_sum['event_contents_1_train'] + df_sum['event_contents_1_warp'] + df_sum['event_contents_1_morph']
    df_sum['build_sum'] = df_sum['event_contents_1_build']
    df_sum['upgrade_sum'] = df_sum['event_contents_1_upgrade'] + df_sum['event_contents_1_research'] + df_sum['event_contents_1_evlove']
    df_sum['attack_sum'] = df_sum['event_contents_1_attack']
    df_sum = df_sum.reset_index()
    
    df_sum_final_col_list = ['game_id','winner','duration','player','species','train_sum','build_sum','upgrade_sum','attack_sum']
    if data_set_type == 'test':
        df_sum_final_col_list.remove('winner')
    
    df_sum_final = df_sum[df_sum_final_col_list]
    
    return df_sum_final


train_ability_feature_df = add_ability_feature(train_ability_feature_df)
test_ability_feature_df = add_ability_feature(test_ability_feature_df)



## unit 생산 정보
def make_unit_train_feature(df: pd.DataFrame):
    duration_df = df[['game_id', 'player', 'supply', 'minerals', 'gas']].groupby(['game_id', 'player']).sum()
    duration_df = duration_df.reset_index()
    
    return duration_df

def add_unit_train_feature(df: pd.DataFrame):
    if 'winner' in df.columns:
        data_set_type = 'train'
    else:
        data_set_type = 'test'
    
    if data_set_type == 'train':
        unit_train_df = make_unit_train_feature(train_ability_df)
    elif data_set_type == 'test':
        unit_train_df = make_unit_train_feature(test_ability_df)
        
    df = df.merge(unit_train_df, how='inner', on=['game_id', 'player'])
    
    return df

train_ability_feature_df = add_unit_train_feature(train_ability_feature_df)
test_ability_feature_df = add_unit_train_feature(test_ability_feature_df)


# map 정보 가져와서 붙이기
map_ftr_df = pd.read_csv(os.path.join(data_folder, 'map_data.csv'))
map_ftr_df = map_ftr_df[['game_id', 'map']]

train_ability_feature_df = train_ability_feature_df.merge(map_ftr_df, how='left', on=['game_id'])
test_ability_feature_df = test_ability_feature_df.merge(map_ftr_df, how='left', on=['game_id'])


# 0, 1 플레이어 합쳐서 게임 당 한 줄로
def player_flattner(df: pd.DataFrame):
    if 'winner' in df.columns:
        data_set_type = 'train'
    else:
        data_set_type = 'test'
        
    temp_df = df

    temp_df_0 = temp_df.loc[temp_df.player == 0].drop('player', axis='columns')
    temp_df_1 = temp_df.loc[temp_df.player == 1].drop('player', axis='columns')

    key_col_list = ['game_id','winner','duration']
    if data_set_type == 'test':
        key_col_list.remove('winner')
    
    final_df = temp_df_0.merge(temp_df_1, how='inner', on=key_col_list, suffixes = ('_0','_1'))
    
    return final_df

train_ability_feature_df = player_flattner(train_ability_feature_df)
test_ability_feature_df = player_flattner(test_ability_feature_df)


# ability count ftr 비율화
# train_ability_feature_df['train_ratio'] = train_ability_feature_df.train_sum_0 / (train_ability_feature_df.train_sum_0 + train_ability_feature_df.train_sum_1 + 0.00000001)
# train_ability_feature_df['build_ratio'] = train_ability_feature_df.build_sum_0 / (train_ability_feature_df.build_sum_0 + train_ability_feature_df.build_sum_1 + 0.00000001)
# train_ability_feature_df['upgrade_ratio'] = train_ability_feature_df.upgrade_sum_0 / (train_ability_feature_df.upgrade_sum_0 + train_ability_feature_df.upgrade_sum_1 + 0.00000001)
# train_ability_feature_df['attack_ratio'] = train_ability_feature_df.attack_sum_0 / (train_ability_feature_df.attack_sum_0 + train_ability_feature_df.attack_sum_1 + 0.00000001)
# train_ability_feature_df['supply_ratio'] = train_ability_feature_df.supply_0 / (train_ability_feature_df.supply_0 + train_ability_feature_df.supply_1 + 0.00000001)
# train_ability_feature_df['minerals_ratio'] = train_ability_feature_df.minerals_0 / (train_ability_feature_df.minerals_0 + train_ability_feature_df.minerals_1 + 0.00000001)
# train_ability_feature_df['gas_ratio'] = train_ability_feature_df.gas_0 / (train_ability_feature_df.gas_0 + train_ability_feature_df.gas_1 + 0.00000001)

# train_ability_feature_df = train_ability_feature_df[['game_id','winner','duration','species_0','species_1','train_ratio','build_ratio','upgrade_ratio','attack_ratio','supply_ratio','minerals_ratio','gas_ratio']]

# test_ability_feature_df['train_ratio'] = test_ability_feature_df.train_sum_0 / (test_ability_feature_df.train_sum_0 + test_ability_feature_df.train_sum_1 + 0.00000001)
# test_ability_feature_df['build_ratio'] = test_ability_feature_df.build_sum_0 / (test_ability_feature_df.build_sum_0 + test_ability_feature_df.build_sum_1 + 0.00000001)
# test_ability_feature_df['upgrade_ratio'] = test_ability_feature_df.upgrade_sum_0 / (test_ability_feature_df.upgrade_sum_0 + test_ability_feature_df.upgrade_sum_1 + 0.00000001)
# test_ability_feature_df['attack_ratio'] = test_ability_feature_df.attack_sum_0 / (test_ability_feature_df.attack_sum_0 + test_ability_feature_df.attack_sum_1 + 0.00000001)
# test_ability_feature_df['supply_ratio'] = test_ability_feature_df.supply_0 / (test_ability_feature_df.supply_0 + test_ability_feature_df.supply_1 + 0.00000001)
# test_ability_feature_df['minerals_ratio'] = test_ability_feature_df.minerals_0 / (test_ability_feature_df.minerals_0 + test_ability_feature_df.minerals_1 + 0.00000001)
# test_ability_feature_df['gas_ratio'] = test_ability_feature_df.gas_0 / (test_ability_feature_df.gas_0 + test_ability_feature_df.gas_1 + 0.00000001)

# test_ability_feature_df = test_ability_feature_df[['game_id','duration','species_0','species_1','train_ratio','build_ratio','upgrade_ratio','attack_ratio','supply_ratio','minerals_ratio','gas_ratio']]


# 플레이어별 종족 one hot encoding
# def species_one_hot_encoding(df: pd.DataFrame):
#     df['species0_T'] = df['species_0'].map(lambda x: 1 if x=='T' else 0)
#     df['species0_P'] = df['species_0'].map(lambda x: 1 if x=='P' else 0)
#     df['species0_Z'] = df['species_0'].map(lambda x: 1 if x=='Z' else 0)
#     df['species1_T'] = df['species_1'].map(lambda x: 1 if x=='T' else 0)
#     df['species1_P'] = df['species_1'].map(lambda x: 1 if x=='P' else 0)
#     df['species1_Z'] = df['species_1'].map(lambda x: 1 if x=='Z' else 0)
#
#     ability_feature_final_df = df.drop(['species_0', 'species_1'], axis='columns')
#
#     return ability_feature_final_df
#
# train_ability_feature_df = species_one_hot_encoding(train_ability_feature_df)
# train_ability_feature_df[['species0_T','species0_P','species0_Z','species1_T','species1_P','species1_Z']] = train_ability_feature_df[['species0_T','species0_P','species0_Z','species1_T','species1_P','species1_Z']].astype('category')
#
# test_ability_feature_df = species_one_hot_encoding(test_ability_feature_df)
# test_ability_feature_df[['species0_T','species0_P','species0_Z','species1_T','species1_P','species1_Z']] = test_ability_feature_df[['species0_T','species0_P','species0_Z','species1_T','species1_P','species1_Z']].astype('category')


# 플레이어별 종족 label encoding
def species_label_encoding(df: pd.DataFrame):
    df['species_0'] = df['species_0'].map(lambda x: 1 if x=='T' else (2 if x=='P' else (3 if x=='Z' else 0)))
    df['species_1'] = df['species_1'].map(lambda x: 1 if x=='T' else (2 if x=='P' else (3 if x=='Z' else 0)))

    ability_feature_final_df = df

    return ability_feature_final_df

train_ability_feature_df = species_label_encoding(train_ability_feature_df)
test_ability_feature_df = species_label_encoding(test_ability_feature_df)



###### from 태환님
train_top100_unit_counts_ftr_df = pd.read_csv(os.path.join(data_folder, 'ftr_top100_unit_counts.csv'))
test_top100_unit_counts_ftr_df = pd.read_csv(os.path.join(data_folder, 'ftr_top100_unit_counts_test.csv'))

## 일부 컬럼만
# train_top100_unit_counts_ftr_df = train_top100_unit_counts_ftr_df[['game_id','p0_u0','p0_u1','p0_u2','p0_u3','p0_u4','p1_u0','p1_u1','p1_u2','p1_u3','p1_u4']]

## 비율화
# train_top100_unit_counts_ftr_df['u0_ratio'] = train_top100_unit_counts_ftr_df.p0_u0 / (train_top100_unit_counts_ftr_df.p0_u0 + train_top100_unit_counts_ftr_df.p1_u0 + 0.00000001)
# train_top100_unit_counts_ftr_df['u1_ratio'] = train_top100_unit_counts_ftr_df.p0_u1 / (train_top100_unit_counts_ftr_df.p0_u1 + train_top100_unit_counts_ftr_df.p1_u1 + 0.00000001)
# train_top100_unit_counts_ftr_df['u2_ratio'] = train_top100_unit_counts_ftr_df.p0_u2 / (train_top100_unit_counts_ftr_df.p0_u2 + train_top100_unit_counts_ftr_df.p1_u2 + 0.00000001)
# train_top100_unit_counts_ftr_df['u3_ratio'] = train_top100_unit_counts_ftr_df.p0_u3 / (train_top100_unit_counts_ftr_df.p0_u3 + train_top100_unit_counts_ftr_df.p1_u3 + 0.00000001)
# train_top100_unit_counts_ftr_df['u4_ratio'] = train_top100_unit_counts_ftr_df.p0_u4 / (train_top100_unit_counts_ftr_df.p0_u4 + train_top100_unit_counts_ftr_df.p1_u4 + 0.00000001)
# train_top100_unit_counts_ftr_df['u5_ratio'] = train_top100_unit_counts_ftr_df.p0_u5 / (train_top100_unit_counts_ftr_df.p0_u5 + train_top100_unit_counts_ftr_df.p1_u5 + 0.00000001)
# train_top100_unit_counts_ftr_df['u6_ratio'] = train_top100_unit_counts_ftr_df.p0_u6 / (train_top100_unit_counts_ftr_df.p0_u6 + train_top100_unit_counts_ftr_df.p1_u6 + 0.00000001)
# train_top100_unit_counts_ftr_df['u7_ratio'] = train_top100_unit_counts_ftr_df.p0_u7 / (train_top100_unit_counts_ftr_df.p0_u7 + train_top100_unit_counts_ftr_df.p1_u7 + 0.00000001)
# train_top100_unit_counts_ftr_df['u8_ratio'] = train_top100_unit_counts_ftr_df.p0_u8 / (train_top100_unit_counts_ftr_df.p0_u8 + train_top100_unit_counts_ftr_df.p1_u8 + 0.00000001)
# train_top100_unit_counts_ftr_df['u9_ratio'] = train_top100_unit_counts_ftr_df.p0_u9 / (train_top100_unit_counts_ftr_df.p0_u9 + train_top100_unit_counts_ftr_df.p1_u9 + 0.00000001)
# train_top100_unit_counts_ftr_df['u10_ratio'] = train_top100_unit_counts_ftr_df.p0_u10 / (train_top100_unit_counts_ftr_df.p0_u10 + train_top100_unit_counts_ftr_df.p1_u10 + 0.00000001)
# train_top100_unit_counts_ftr_df['u11_ratio'] = train_top100_unit_counts_ftr_df.p0_u11 / (train_top100_unit_counts_ftr_df.p0_u11 + train_top100_unit_counts_ftr_df.p1_u11 + 0.00000001)
# train_top100_unit_counts_ftr_df['u12_ratio'] = train_top100_unit_counts_ftr_df.p0_u12 / (train_top100_unit_counts_ftr_df.p0_u12 + train_top100_unit_counts_ftr_df.p1_u12 + 0.00000001)
# train_top100_unit_counts_ftr_df['u13_ratio'] = train_top100_unit_counts_ftr_df.p0_u13 / (train_top100_unit_counts_ftr_df.p0_u13 + train_top100_unit_counts_ftr_df.p1_u13 + 0.00000001)
# train_top100_unit_counts_ftr_df['u14_ratio'] = train_top100_unit_counts_ftr_df.p0_u14 / (train_top100_unit_counts_ftr_df.p0_u14 + train_top100_unit_counts_ftr_df.p1_u14 + 0.00000001)
# train_top100_unit_counts_ftr_df['u15_ratio'] = train_top100_unit_counts_ftr_df.p0_u15 / (train_top100_unit_counts_ftr_df.p0_u15 + train_top100_unit_counts_ftr_df.p1_u15 + 0.00000001)
# train_top100_unit_counts_ftr_df['u16_ratio'] = train_top100_unit_counts_ftr_df.p0_u16 / (train_top100_unit_counts_ftr_df.p0_u16 + train_top100_unit_counts_ftr_df.p1_u16 + 0.00000001)
# train_top100_unit_counts_ftr_df['u17_ratio'] = train_top100_unit_counts_ftr_df.p0_u17 / (train_top100_unit_counts_ftr_df.p0_u17 + train_top100_unit_counts_ftr_df.p1_u17 + 0.00000001)
# train_top100_unit_counts_ftr_df['u18_ratio'] = train_top100_unit_counts_ftr_df.p0_u18 / (train_top100_unit_counts_ftr_df.p0_u18 + train_top100_unit_counts_ftr_df.p1_u18 + 0.00000001)
# train_top100_unit_counts_ftr_df['u19_ratio'] = train_top100_unit_counts_ftr_df.p0_u19 / (train_top100_unit_counts_ftr_df.p0_u19 + train_top100_unit_counts_ftr_df.p1_u19 + 0.00000001)
# train_top100_unit_counts_ftr_df['u20_ratio'] = train_top100_unit_counts_ftr_df.p0_u20 / (train_top100_unit_counts_ftr_df.p0_u20 + train_top100_unit_counts_ftr_df.p1_u20 + 0.00000001)
# train_top100_unit_counts_ftr_df['u21_ratio'] = train_top100_unit_counts_ftr_df.p0_u21 / (train_top100_unit_counts_ftr_df.p0_u21 + train_top100_unit_counts_ftr_df.p1_u21 + 0.00000001)
# train_top100_unit_counts_ftr_df['u22_ratio'] = train_top100_unit_counts_ftr_df.p0_u22 / (train_top100_unit_counts_ftr_df.p0_u22 + train_top100_unit_counts_ftr_df.p1_u22 + 0.00000001)
# train_top100_unit_counts_ftr_df['u23_ratio'] = train_top100_unit_counts_ftr_df.p0_u23 / (train_top100_unit_counts_ftr_df.p0_u23 + train_top100_unit_counts_ftr_df.p1_u23 + 0.00000001)
# train_top100_unit_counts_ftr_df['u24_ratio'] = train_top100_unit_counts_ftr_df.p0_u24 / (train_top100_unit_counts_ftr_df.p0_u24 + train_top100_unit_counts_ftr_df.p1_u24 + 0.00000001)
# train_top100_unit_counts_ftr_df['u25_ratio'] = train_top100_unit_counts_ftr_df.p0_u25 / (train_top100_unit_counts_ftr_df.p0_u25 + train_top100_unit_counts_ftr_df.p1_u25 + 0.00000001)
# train_top100_unit_counts_ftr_df['u26_ratio'] = train_top100_unit_counts_ftr_df.p0_u26 / (train_top100_unit_counts_ftr_df.p0_u26 + train_top100_unit_counts_ftr_df.p1_u26 + 0.00000001)
# train_top100_unit_counts_ftr_df['u27_ratio'] = train_top100_unit_counts_ftr_df.p0_u27 / (train_top100_unit_counts_ftr_df.p0_u27 + train_top100_unit_counts_ftr_df.p1_u27 + 0.00000001)
# train_top100_unit_counts_ftr_df['u28_ratio'] = train_top100_unit_counts_ftr_df.p0_u28 / (train_top100_unit_counts_ftr_df.p0_u28 + train_top100_unit_counts_ftr_df.p1_u28 + 0.00000001)
# train_top100_unit_counts_ftr_df['u29_ratio'] = train_top100_unit_counts_ftr_df.p0_u29 / (train_top100_unit_counts_ftr_df.p0_u29 + train_top100_unit_counts_ftr_df.p1_u29 + 0.00000001)
# train_top100_unit_counts_ftr_df['u30_ratio'] = train_top100_unit_counts_ftr_df.p0_u30 / (train_top100_unit_counts_ftr_df.p0_u30 + train_top100_unit_counts_ftr_df.p1_u30 + 0.00000001)
# train_top100_unit_counts_ftr_df['u31_ratio'] = train_top100_unit_counts_ftr_df.p0_u31 / (train_top100_unit_counts_ftr_df.p0_u31 + train_top100_unit_counts_ftr_df.p1_u31 + 0.00000001)
# train_top100_unit_counts_ftr_df['u32_ratio'] = train_top100_unit_counts_ftr_df.p0_u32 / (train_top100_unit_counts_ftr_df.p0_u32 + train_top100_unit_counts_ftr_df.p1_u32 + 0.00000001)
# train_top100_unit_counts_ftr_df['u33_ratio'] = train_top100_unit_counts_ftr_df.p0_u33 / (train_top100_unit_counts_ftr_df.p0_u33 + train_top100_unit_counts_ftr_df.p1_u33 + 0.00000001)
# train_top100_unit_counts_ftr_df['u34_ratio'] = train_top100_unit_counts_ftr_df.p0_u34 / (train_top100_unit_counts_ftr_df.p0_u34 + train_top100_unit_counts_ftr_df.p1_u34 + 0.00000001)
# train_top100_unit_counts_ftr_df['u35_ratio'] = train_top100_unit_counts_ftr_df.p0_u35 / (train_top100_unit_counts_ftr_df.p0_u35 + train_top100_unit_counts_ftr_df.p1_u35 + 0.00000001)
# train_top100_unit_counts_ftr_df['u36_ratio'] = train_top100_unit_counts_ftr_df.p0_u36 / (train_top100_unit_counts_ftr_df.p0_u36 + train_top100_unit_counts_ftr_df.p1_u36 + 0.00000001)
# train_top100_unit_counts_ftr_df['u37_ratio'] = train_top100_unit_counts_ftr_df.p0_u37 / (train_top100_unit_counts_ftr_df.p0_u37 + train_top100_unit_counts_ftr_df.p1_u37 + 0.00000001)
# train_top100_unit_counts_ftr_df['u38_ratio'] = train_top100_unit_counts_ftr_df.p0_u38 / (train_top100_unit_counts_ftr_df.p0_u38 + train_top100_unit_counts_ftr_df.p1_u38 + 0.00000001)
# train_top100_unit_counts_ftr_df['u39_ratio'] = train_top100_unit_counts_ftr_df.p0_u39 / (train_top100_unit_counts_ftr_df.p0_u39 + train_top100_unit_counts_ftr_df.p1_u39 + 0.00000001)
# train_top100_unit_counts_ftr_df['u40_ratio'] = train_top100_unit_counts_ftr_df.p0_u40 / (train_top100_unit_counts_ftr_df.p0_u40 + train_top100_unit_counts_ftr_df.p1_u40 + 0.00000001)
# train_top100_unit_counts_ftr_df['u41_ratio'] = train_top100_unit_counts_ftr_df.p0_u41 / (train_top100_unit_counts_ftr_df.p0_u41 + train_top100_unit_counts_ftr_df.p1_u41 + 0.00000001)
# train_top100_unit_counts_ftr_df['u42_ratio'] = train_top100_unit_counts_ftr_df.p0_u42 / (train_top100_unit_counts_ftr_df.p0_u42 + train_top100_unit_counts_ftr_df.p1_u42 + 0.00000001)
# train_top100_unit_counts_ftr_df['u43_ratio'] = train_top100_unit_counts_ftr_df.p0_u43 / (train_top100_unit_counts_ftr_df.p0_u43 + train_top100_unit_counts_ftr_df.p1_u43 + 0.00000001)
# train_top100_unit_counts_ftr_df['u44_ratio'] = train_top100_unit_counts_ftr_df.p0_u44 / (train_top100_unit_counts_ftr_df.p0_u44 + train_top100_unit_counts_ftr_df.p1_u44 + 0.00000001)
# train_top100_unit_counts_ftr_df['u45_ratio'] = train_top100_unit_counts_ftr_df.p0_u45 / (train_top100_unit_counts_ftr_df.p0_u45 + train_top100_unit_counts_ftr_df.p1_u45 + 0.00000001)
# train_top100_unit_counts_ftr_df['u46_ratio'] = train_top100_unit_counts_ftr_df.p0_u46 / (train_top100_unit_counts_ftr_df.p0_u46 + train_top100_unit_counts_ftr_df.p1_u46 + 0.00000001)
# train_top100_unit_counts_ftr_df['u47_ratio'] = train_top100_unit_counts_ftr_df.p0_u47 / (train_top100_unit_counts_ftr_df.p0_u47 + train_top100_unit_counts_ftr_df.p1_u47 + 0.00000001)
# train_top100_unit_counts_ftr_df['u48_ratio'] = train_top100_unit_counts_ftr_df.p0_u48 / (train_top100_unit_counts_ftr_df.p0_u48 + train_top100_unit_counts_ftr_df.p1_u48 + 0.00000001)
# train_top100_unit_counts_ftr_df['u49_ratio'] = train_top100_unit_counts_ftr_df.p0_u49 / (train_top100_unit_counts_ftr_df.p0_u49 + train_top100_unit_counts_ftr_df.p1_u49 + 0.00000001)
# train_top100_unit_counts_ftr_df['u50_ratio'] = train_top100_unit_counts_ftr_df.p0_u50 / (train_top100_unit_counts_ftr_df.p0_u50 + train_top100_unit_counts_ftr_df.p1_u50 + 0.00000001)
# train_top100_unit_counts_ftr_df['u51_ratio'] = train_top100_unit_counts_ftr_df.p0_u51 / (train_top100_unit_counts_ftr_df.p0_u51 + train_top100_unit_counts_ftr_df.p1_u51 + 0.00000001)
# train_top100_unit_counts_ftr_df['u52_ratio'] = train_top100_unit_counts_ftr_df.p0_u52 / (train_top100_unit_counts_ftr_df.p0_u52 + train_top100_unit_counts_ftr_df.p1_u52 + 0.00000001)
# train_top100_unit_counts_ftr_df['u53_ratio'] = train_top100_unit_counts_ftr_df.p0_u53 / (train_top100_unit_counts_ftr_df.p0_u53 + train_top100_unit_counts_ftr_df.p1_u53 + 0.00000001)
# train_top100_unit_counts_ftr_df['u54_ratio'] = train_top100_unit_counts_ftr_df.p0_u54 / (train_top100_unit_counts_ftr_df.p0_u54 + train_top100_unit_counts_ftr_df.p1_u54 + 0.00000001)
# train_top100_unit_counts_ftr_df['u55_ratio'] = train_top100_unit_counts_ftr_df.p0_u55 / (train_top100_unit_counts_ftr_df.p0_u55 + train_top100_unit_counts_ftr_df.p1_u55 + 0.00000001)
# train_top100_unit_counts_ftr_df['u56_ratio'] = train_top100_unit_counts_ftr_df.p0_u56 / (train_top100_unit_counts_ftr_df.p0_u56 + train_top100_unit_counts_ftr_df.p1_u56 + 0.00000001)
# train_top100_unit_counts_ftr_df['u57_ratio'] = train_top100_unit_counts_ftr_df.p0_u57 / (train_top100_unit_counts_ftr_df.p0_u57 + train_top100_unit_counts_ftr_df.p1_u57 + 0.00000001)
# train_top100_unit_counts_ftr_df['u58_ratio'] = train_top100_unit_counts_ftr_df.p0_u58 / (train_top100_unit_counts_ftr_df.p0_u58 + train_top100_unit_counts_ftr_df.p1_u58 + 0.00000001)
# train_top100_unit_counts_ftr_df['u59_ratio'] = train_top100_unit_counts_ftr_df.p0_u59 / (train_top100_unit_counts_ftr_df.p0_u59 + train_top100_unit_counts_ftr_df.p1_u59 + 0.00000001)
# train_top100_unit_counts_ftr_df['u60_ratio'] = train_top100_unit_counts_ftr_df.p0_u60 / (train_top100_unit_counts_ftr_df.p0_u60 + train_top100_unit_counts_ftr_df.p1_u60 + 0.00000001)
# train_top100_unit_counts_ftr_df['u61_ratio'] = train_top100_unit_counts_ftr_df.p0_u61 / (train_top100_unit_counts_ftr_df.p0_u61 + train_top100_unit_counts_ftr_df.p1_u61 + 0.00000001)
# train_top100_unit_counts_ftr_df['u62_ratio'] = train_top100_unit_counts_ftr_df.p0_u62 / (train_top100_unit_counts_ftr_df.p0_u62 + train_top100_unit_counts_ftr_df.p1_u62 + 0.00000001)
# train_top100_unit_counts_ftr_df['u63_ratio'] = train_top100_unit_counts_ftr_df.p0_u63 / (train_top100_unit_counts_ftr_df.p0_u63 + train_top100_unit_counts_ftr_df.p1_u63 + 0.00000001)
# train_top100_unit_counts_ftr_df['u64_ratio'] = train_top100_unit_counts_ftr_df.p0_u64 / (train_top100_unit_counts_ftr_df.p0_u64 + train_top100_unit_counts_ftr_df.p1_u64 + 0.00000001)
# train_top100_unit_counts_ftr_df['u65_ratio'] = train_top100_unit_counts_ftr_df.p0_u65 / (train_top100_unit_counts_ftr_df.p0_u65 + train_top100_unit_counts_ftr_df.p1_u65 + 0.00000001)
# train_top100_unit_counts_ftr_df['u66_ratio'] = train_top100_unit_counts_ftr_df.p0_u66 / (train_top100_unit_counts_ftr_df.p0_u66 + train_top100_unit_counts_ftr_df.p1_u66 + 0.00000001)
# train_top100_unit_counts_ftr_df['u67_ratio'] = train_top100_unit_counts_ftr_df.p0_u67 / (train_top100_unit_counts_ftr_df.p0_u67 + train_top100_unit_counts_ftr_df.p1_u67 + 0.00000001)
# train_top100_unit_counts_ftr_df['u68_ratio'] = train_top100_unit_counts_ftr_df.p0_u68 / (train_top100_unit_counts_ftr_df.p0_u68 + train_top100_unit_counts_ftr_df.p1_u68 + 0.00000001)
# train_top100_unit_counts_ftr_df['u69_ratio'] = train_top100_unit_counts_ftr_df.p0_u69 / (train_top100_unit_counts_ftr_df.p0_u69 + train_top100_unit_counts_ftr_df.p1_u69 + 0.00000001)
# train_top100_unit_counts_ftr_df['u70_ratio'] = train_top100_unit_counts_ftr_df.p0_u70 / (train_top100_unit_counts_ftr_df.p0_u70 + train_top100_unit_counts_ftr_df.p1_u70 + 0.00000001)
# train_top100_unit_counts_ftr_df['u71_ratio'] = train_top100_unit_counts_ftr_df.p0_u71 / (train_top100_unit_counts_ftr_df.p0_u71 + train_top100_unit_counts_ftr_df.p1_u71 + 0.00000001)
# train_top100_unit_counts_ftr_df['u72_ratio'] = train_top100_unit_counts_ftr_df.p0_u72 / (train_top100_unit_counts_ftr_df.p0_u72 + train_top100_unit_counts_ftr_df.p1_u72 + 0.00000001)
# train_top100_unit_counts_ftr_df['u73_ratio'] = train_top100_unit_counts_ftr_df.p0_u73 / (train_top100_unit_counts_ftr_df.p0_u73 + train_top100_unit_counts_ftr_df.p1_u73 + 0.00000001)
# train_top100_unit_counts_ftr_df['u74_ratio'] = train_top100_unit_counts_ftr_df.p0_u74 / (train_top100_unit_counts_ftr_df.p0_u74 + train_top100_unit_counts_ftr_df.p1_u74 + 0.00000001)
# train_top100_unit_counts_ftr_df['u75_ratio'] = train_top100_unit_counts_ftr_df.p0_u75 / (train_top100_unit_counts_ftr_df.p0_u75 + train_top100_unit_counts_ftr_df.p1_u75 + 0.00000001)
# train_top100_unit_counts_ftr_df['u76_ratio'] = train_top100_unit_counts_ftr_df.p0_u76 / (train_top100_unit_counts_ftr_df.p0_u76 + train_top100_unit_counts_ftr_df.p1_u76 + 0.00000001)
# train_top100_unit_counts_ftr_df['u77_ratio'] = train_top100_unit_counts_ftr_df.p0_u77 / (train_top100_unit_counts_ftr_df.p0_u77 + train_top100_unit_counts_ftr_df.p1_u77 + 0.00000001)
# train_top100_unit_counts_ftr_df['u78_ratio'] = train_top100_unit_counts_ftr_df.p0_u78 / (train_top100_unit_counts_ftr_df.p0_u78 + train_top100_unit_counts_ftr_df.p1_u78 + 0.00000001)
# train_top100_unit_counts_ftr_df['u79_ratio'] = train_top100_unit_counts_ftr_df.p0_u79 / (train_top100_unit_counts_ftr_df.p0_u79 + train_top100_unit_counts_ftr_df.p1_u79 + 0.00000001)
# train_top100_unit_counts_ftr_df['u80_ratio'] = train_top100_unit_counts_ftr_df.p0_u80 / (train_top100_unit_counts_ftr_df.p0_u80 + train_top100_unit_counts_ftr_df.p1_u80 + 0.00000001)
# train_top100_unit_counts_ftr_df['u81_ratio'] = train_top100_unit_counts_ftr_df.p0_u81 / (train_top100_unit_counts_ftr_df.p0_u81 + train_top100_unit_counts_ftr_df.p1_u81 + 0.00000001)
# train_top100_unit_counts_ftr_df['u82_ratio'] = train_top100_unit_counts_ftr_df.p0_u82 / (train_top100_unit_counts_ftr_df.p0_u82 + train_top100_unit_counts_ftr_df.p1_u82 + 0.00000001)
# train_top100_unit_counts_ftr_df['u83_ratio'] = train_top100_unit_counts_ftr_df.p0_u83 / (train_top100_unit_counts_ftr_df.p0_u83 + train_top100_unit_counts_ftr_df.p1_u83 + 0.00000001)
# train_top100_unit_counts_ftr_df['u84_ratio'] = train_top100_unit_counts_ftr_df.p0_u84 / (train_top100_unit_counts_ftr_df.p0_u84 + train_top100_unit_counts_ftr_df.p1_u84 + 0.00000001)
# train_top100_unit_counts_ftr_df['u85_ratio'] = train_top100_unit_counts_ftr_df.p0_u85 / (train_top100_unit_counts_ftr_df.p0_u85 + train_top100_unit_counts_ftr_df.p1_u85 + 0.00000001)
# train_top100_unit_counts_ftr_df['u86_ratio'] = train_top100_unit_counts_ftr_df.p0_u86 / (train_top100_unit_counts_ftr_df.p0_u86 + train_top100_unit_counts_ftr_df.p1_u86 + 0.00000001)
# train_top100_unit_counts_ftr_df['u87_ratio'] = train_top100_unit_counts_ftr_df.p0_u87 / (train_top100_unit_counts_ftr_df.p0_u87 + train_top100_unit_counts_ftr_df.p1_u87 + 0.00000001)
# train_top100_unit_counts_ftr_df['u88_ratio'] = train_top100_unit_counts_ftr_df.p0_u88 / (train_top100_unit_counts_ftr_df.p0_u88 + train_top100_unit_counts_ftr_df.p1_u88 + 0.00000001)
# train_top100_unit_counts_ftr_df['u89_ratio'] = train_top100_unit_counts_ftr_df.p0_u89 / (train_top100_unit_counts_ftr_df.p0_u89 + train_top100_unit_counts_ftr_df.p1_u89 + 0.00000001)
# train_top100_unit_counts_ftr_df['u90_ratio'] = train_top100_unit_counts_ftr_df.p0_u90 / (train_top100_unit_counts_ftr_df.p0_u90 + train_top100_unit_counts_ftr_df.p1_u90 + 0.00000001)
# train_top100_unit_counts_ftr_df['u91_ratio'] = train_top100_unit_counts_ftr_df.p0_u91 / (train_top100_unit_counts_ftr_df.p0_u91 + train_top100_unit_counts_ftr_df.p1_u91 + 0.00000001)
# train_top100_unit_counts_ftr_df['u92_ratio'] = train_top100_unit_counts_ftr_df.p0_u92 / (train_top100_unit_counts_ftr_df.p0_u92 + train_top100_unit_counts_ftr_df.p1_u92 + 0.00000001)
# train_top100_unit_counts_ftr_df['u93_ratio'] = train_top100_unit_counts_ftr_df.p0_u93 / (train_top100_unit_counts_ftr_df.p0_u93 + train_top100_unit_counts_ftr_df.p1_u93 + 0.00000001)
# train_top100_unit_counts_ftr_df['u94_ratio'] = train_top100_unit_counts_ftr_df.p0_u94 / (train_top100_unit_counts_ftr_df.p0_u94 + train_top100_unit_counts_ftr_df.p1_u94 + 0.00000001)
# train_top100_unit_counts_ftr_df['u95_ratio'] = train_top100_unit_counts_ftr_df.p0_u95 / (train_top100_unit_counts_ftr_df.p0_u95 + train_top100_unit_counts_ftr_df.p1_u95 + 0.00000001)
# train_top100_unit_counts_ftr_df['u96_ratio'] = train_top100_unit_counts_ftr_df.p0_u96 / (train_top100_unit_counts_ftr_df.p0_u96 + train_top100_unit_counts_ftr_df.p1_u96 + 0.00000001)
# train_top100_unit_counts_ftr_df['u97_ratio'] = train_top100_unit_counts_ftr_df.p0_u97 / (train_top100_unit_counts_ftr_df.p0_u97 + train_top100_unit_counts_ftr_df.p1_u97 + 0.00000001)
# train_top100_unit_counts_ftr_df['u98_ratio'] = train_top100_unit_counts_ftr_df.p0_u98 / (train_top100_unit_counts_ftr_df.p0_u98 + train_top100_unit_counts_ftr_df.p1_u98 + 0.00000001)
# train_top100_unit_counts_ftr_df['u99_ratio'] = train_top100_unit_counts_ftr_df.p0_u99 / (train_top100_unit_counts_ftr_df.p0_u99 + train_top100_unit_counts_ftr_df.p1_u99 + 0.00000001)

# test_top100_unit_counts_ftr_df['u0_ratio'] =  test_top100_unit_counts_ftr_df.p0_u0 / (test_top100_unit_counts_ftr_df.p0_u0 + test_top100_unit_counts_ftr_df.p1_u0 + 0.00000001)
# test_top100_unit_counts_ftr_df['u1_ratio'] =  test_top100_unit_counts_ftr_df.p0_u1 / (test_top100_unit_counts_ftr_df.p0_u1 + test_top100_unit_counts_ftr_df.p1_u1 + 0.00000001)
# test_top100_unit_counts_ftr_df['u2_ratio'] =  test_top100_unit_counts_ftr_df.p0_u2 / (test_top100_unit_counts_ftr_df.p0_u2 + test_top100_unit_counts_ftr_df.p1_u2 + 0.00000001)
# test_top100_unit_counts_ftr_df['u3_ratio'] =  test_top100_unit_counts_ftr_df.p0_u3 / (test_top100_unit_counts_ftr_df.p0_u3 + test_top100_unit_counts_ftr_df.p1_u3 + 0.00000001)
# test_top100_unit_counts_ftr_df['u4_ratio'] =  test_top100_unit_counts_ftr_df.p0_u4 / (test_top100_unit_counts_ftr_df.p0_u4 + test_top100_unit_counts_ftr_df.p1_u4 + 0.00000001)
# test_top100_unit_counts_ftr_df['u5_ratio'] =  test_top100_unit_counts_ftr_df.p0_u5 / (test_top100_unit_counts_ftr_df.p0_u5 + test_top100_unit_counts_ftr_df.p1_u5 + 0.00000001)
# test_top100_unit_counts_ftr_df['u6_ratio'] =  test_top100_unit_counts_ftr_df.p0_u6 / (test_top100_unit_counts_ftr_df.p0_u6 + test_top100_unit_counts_ftr_df.p1_u6 + 0.00000001)
# test_top100_unit_counts_ftr_df['u7_ratio'] =  test_top100_unit_counts_ftr_df.p0_u7 / (test_top100_unit_counts_ftr_df.p0_u7 + test_top100_unit_counts_ftr_df.p1_u7 + 0.00000001)
# test_top100_unit_counts_ftr_df['u8_ratio'] =  test_top100_unit_counts_ftr_df.p0_u8 / (test_top100_unit_counts_ftr_df.p0_u8 + test_top100_unit_counts_ftr_df.p1_u8 + 0.00000001)
# test_top100_unit_counts_ftr_df['u9_ratio'] =  test_top100_unit_counts_ftr_df.p0_u9 / (test_top100_unit_counts_ftr_df.p0_u9 + test_top100_unit_counts_ftr_df.p1_u9 + 0.00000001)
# test_top100_unit_counts_ftr_df['u10_ratio'] = test_top100_unit_counts_ftr_df.p0_u10 / (test_top100_unit_counts_ftr_df.p0_u10 + test_top100_unit_counts_ftr_df.p1_u10 + 0.00000001)
# test_top100_unit_counts_ftr_df['u11_ratio'] = test_top100_unit_counts_ftr_df.p0_u11 / (test_top100_unit_counts_ftr_df.p0_u11 + test_top100_unit_counts_ftr_df.p1_u11 + 0.00000001)
# test_top100_unit_counts_ftr_df['u12_ratio'] = test_top100_unit_counts_ftr_df.p0_u12 / (test_top100_unit_counts_ftr_df.p0_u12 + test_top100_unit_counts_ftr_df.p1_u12 + 0.00000001)
# test_top100_unit_counts_ftr_df['u13_ratio'] = test_top100_unit_counts_ftr_df.p0_u13 / (test_top100_unit_counts_ftr_df.p0_u13 + test_top100_unit_counts_ftr_df.p1_u13 + 0.00000001)
# test_top100_unit_counts_ftr_df['u14_ratio'] = test_top100_unit_counts_ftr_df.p0_u14 / (test_top100_unit_counts_ftr_df.p0_u14 + test_top100_unit_counts_ftr_df.p1_u14 + 0.00000001)
# test_top100_unit_counts_ftr_df['u15_ratio'] = test_top100_unit_counts_ftr_df.p0_u15 / (test_top100_unit_counts_ftr_df.p0_u15 + test_top100_unit_counts_ftr_df.p1_u15 + 0.00000001)
# test_top100_unit_counts_ftr_df['u16_ratio'] = test_top100_unit_counts_ftr_df.p0_u16 / (test_top100_unit_counts_ftr_df.p0_u16 + test_top100_unit_counts_ftr_df.p1_u16 + 0.00000001)
# test_top100_unit_counts_ftr_df['u17_ratio'] = test_top100_unit_counts_ftr_df.p0_u17 / (test_top100_unit_counts_ftr_df.p0_u17 + test_top100_unit_counts_ftr_df.p1_u17 + 0.00000001)
# test_top100_unit_counts_ftr_df['u18_ratio'] = test_top100_unit_counts_ftr_df.p0_u18 / (test_top100_unit_counts_ftr_df.p0_u18 + test_top100_unit_counts_ftr_df.p1_u18 + 0.00000001)
# test_top100_unit_counts_ftr_df['u19_ratio'] = test_top100_unit_counts_ftr_df.p0_u19 / (test_top100_unit_counts_ftr_df.p0_u19 + test_top100_unit_counts_ftr_df.p1_u19 + 0.00000001)
# test_top100_unit_counts_ftr_df['u20_ratio'] = test_top100_unit_counts_ftr_df.p0_u20 / (test_top100_unit_counts_ftr_df.p0_u20 + test_top100_unit_counts_ftr_df.p1_u20 + 0.00000001)
# test_top100_unit_counts_ftr_df['u21_ratio'] = test_top100_unit_counts_ftr_df.p0_u21 / (test_top100_unit_counts_ftr_df.p0_u21 + test_top100_unit_counts_ftr_df.p1_u21 + 0.00000001)
# test_top100_unit_counts_ftr_df['u22_ratio'] = test_top100_unit_counts_ftr_df.p0_u22 / (test_top100_unit_counts_ftr_df.p0_u22 + test_top100_unit_counts_ftr_df.p1_u22 + 0.00000001)
# test_top100_unit_counts_ftr_df['u23_ratio'] = test_top100_unit_counts_ftr_df.p0_u23 / (test_top100_unit_counts_ftr_df.p0_u23 + test_top100_unit_counts_ftr_df.p1_u23 + 0.00000001)
# test_top100_unit_counts_ftr_df['u24_ratio'] = test_top100_unit_counts_ftr_df.p0_u24 / (test_top100_unit_counts_ftr_df.p0_u24 + test_top100_unit_counts_ftr_df.p1_u24 + 0.00000001)
# test_top100_unit_counts_ftr_df['u25_ratio'] = test_top100_unit_counts_ftr_df.p0_u25 / (test_top100_unit_counts_ftr_df.p0_u25 + test_top100_unit_counts_ftr_df.p1_u25 + 0.00000001)
# test_top100_unit_counts_ftr_df['u26_ratio'] = test_top100_unit_counts_ftr_df.p0_u26 / (test_top100_unit_counts_ftr_df.p0_u26 + test_top100_unit_counts_ftr_df.p1_u26 + 0.00000001)
# test_top100_unit_counts_ftr_df['u27_ratio'] = test_top100_unit_counts_ftr_df.p0_u27 / (test_top100_unit_counts_ftr_df.p0_u27 + test_top100_unit_counts_ftr_df.p1_u27 + 0.00000001)
# test_top100_unit_counts_ftr_df['u28_ratio'] = test_top100_unit_counts_ftr_df.p0_u28 / (test_top100_unit_counts_ftr_df.p0_u28 + test_top100_unit_counts_ftr_df.p1_u28 + 0.00000001)
# test_top100_unit_counts_ftr_df['u29_ratio'] = test_top100_unit_counts_ftr_df.p0_u29 / (test_top100_unit_counts_ftr_df.p0_u29 + test_top100_unit_counts_ftr_df.p1_u29 + 0.00000001)
# test_top100_unit_counts_ftr_df['u30_ratio'] = test_top100_unit_counts_ftr_df.p0_u30 / (test_top100_unit_counts_ftr_df.p0_u30 + test_top100_unit_counts_ftr_df.p1_u30 + 0.00000001)
# test_top100_unit_counts_ftr_df['u31_ratio'] = test_top100_unit_counts_ftr_df.p0_u31 / (test_top100_unit_counts_ftr_df.p0_u31 + test_top100_unit_counts_ftr_df.p1_u31 + 0.00000001)
# test_top100_unit_counts_ftr_df['u32_ratio'] = test_top100_unit_counts_ftr_df.p0_u32 / (test_top100_unit_counts_ftr_df.p0_u32 + test_top100_unit_counts_ftr_df.p1_u32 + 0.00000001)
# test_top100_unit_counts_ftr_df['u33_ratio'] = test_top100_unit_counts_ftr_df.p0_u33 / (test_top100_unit_counts_ftr_df.p0_u33 + test_top100_unit_counts_ftr_df.p1_u33 + 0.00000001)
# test_top100_unit_counts_ftr_df['u34_ratio'] = test_top100_unit_counts_ftr_df.p0_u34 / (test_top100_unit_counts_ftr_df.p0_u34 + test_top100_unit_counts_ftr_df.p1_u34 + 0.00000001)
# test_top100_unit_counts_ftr_df['u35_ratio'] = test_top100_unit_counts_ftr_df.p0_u35 / (test_top100_unit_counts_ftr_df.p0_u35 + test_top100_unit_counts_ftr_df.p1_u35 + 0.00000001)
# test_top100_unit_counts_ftr_df['u36_ratio'] = test_top100_unit_counts_ftr_df.p0_u36 / (test_top100_unit_counts_ftr_df.p0_u36 + test_top100_unit_counts_ftr_df.p1_u36 + 0.00000001)
# test_top100_unit_counts_ftr_df['u37_ratio'] = test_top100_unit_counts_ftr_df.p0_u37 / (test_top100_unit_counts_ftr_df.p0_u37 + test_top100_unit_counts_ftr_df.p1_u37 + 0.00000001)
# test_top100_unit_counts_ftr_df['u38_ratio'] = test_top100_unit_counts_ftr_df.p0_u38 / (test_top100_unit_counts_ftr_df.p0_u38 + test_top100_unit_counts_ftr_df.p1_u38 + 0.00000001)
# test_top100_unit_counts_ftr_df['u39_ratio'] = test_top100_unit_counts_ftr_df.p0_u39 / (test_top100_unit_counts_ftr_df.p0_u39 + test_top100_unit_counts_ftr_df.p1_u39 + 0.00000001)
# test_top100_unit_counts_ftr_df['u40_ratio'] = test_top100_unit_counts_ftr_df.p0_u40 / (test_top100_unit_counts_ftr_df.p0_u40 + test_top100_unit_counts_ftr_df.p1_u40 + 0.00000001)
# test_top100_unit_counts_ftr_df['u41_ratio'] = test_top100_unit_counts_ftr_df.p0_u41 / (test_top100_unit_counts_ftr_df.p0_u41 + test_top100_unit_counts_ftr_df.p1_u41 + 0.00000001)
# test_top100_unit_counts_ftr_df['u42_ratio'] = test_top100_unit_counts_ftr_df.p0_u42 / (test_top100_unit_counts_ftr_df.p0_u42 + test_top100_unit_counts_ftr_df.p1_u42 + 0.00000001)
# test_top100_unit_counts_ftr_df['u43_ratio'] = test_top100_unit_counts_ftr_df.p0_u43 / (test_top100_unit_counts_ftr_df.p0_u43 + test_top100_unit_counts_ftr_df.p1_u43 + 0.00000001)
# test_top100_unit_counts_ftr_df['u44_ratio'] = test_top100_unit_counts_ftr_df.p0_u44 / (test_top100_unit_counts_ftr_df.p0_u44 + test_top100_unit_counts_ftr_df.p1_u44 + 0.00000001)
# test_top100_unit_counts_ftr_df['u45_ratio'] = test_top100_unit_counts_ftr_df.p0_u45 / (test_top100_unit_counts_ftr_df.p0_u45 + test_top100_unit_counts_ftr_df.p1_u45 + 0.00000001)
# test_top100_unit_counts_ftr_df['u46_ratio'] = test_top100_unit_counts_ftr_df.p0_u46 / (test_top100_unit_counts_ftr_df.p0_u46 + test_top100_unit_counts_ftr_df.p1_u46 + 0.00000001)
# test_top100_unit_counts_ftr_df['u47_ratio'] = test_top100_unit_counts_ftr_df.p0_u47 / (test_top100_unit_counts_ftr_df.p0_u47 + test_top100_unit_counts_ftr_df.p1_u47 + 0.00000001)
# test_top100_unit_counts_ftr_df['u48_ratio'] = test_top100_unit_counts_ftr_df.p0_u48 / (test_top100_unit_counts_ftr_df.p0_u48 + test_top100_unit_counts_ftr_df.p1_u48 + 0.00000001)
# test_top100_unit_counts_ftr_df['u49_ratio'] = test_top100_unit_counts_ftr_df.p0_u49 / (test_top100_unit_counts_ftr_df.p0_u49 + test_top100_unit_counts_ftr_df.p1_u49 + 0.00000001)
# test_top100_unit_counts_ftr_df['u50_ratio'] = test_top100_unit_counts_ftr_df.p0_u50 / (test_top100_unit_counts_ftr_df.p0_u50 + test_top100_unit_counts_ftr_df.p1_u50 + 0.00000001)
# test_top100_unit_counts_ftr_df['u51_ratio'] = test_top100_unit_counts_ftr_df.p0_u51 / (test_top100_unit_counts_ftr_df.p0_u51 + test_top100_unit_counts_ftr_df.p1_u51 + 0.00000001)
# test_top100_unit_counts_ftr_df['u52_ratio'] = test_top100_unit_counts_ftr_df.p0_u52 / (test_top100_unit_counts_ftr_df.p0_u52 + test_top100_unit_counts_ftr_df.p1_u52 + 0.00000001)
# test_top100_unit_counts_ftr_df['u53_ratio'] = test_top100_unit_counts_ftr_df.p0_u53 / (test_top100_unit_counts_ftr_df.p0_u53 + test_top100_unit_counts_ftr_df.p1_u53 + 0.00000001)
# test_top100_unit_counts_ftr_df['u54_ratio'] = test_top100_unit_counts_ftr_df.p0_u54 / (test_top100_unit_counts_ftr_df.p0_u54 + test_top100_unit_counts_ftr_df.p1_u54 + 0.00000001)
# test_top100_unit_counts_ftr_df['u55_ratio'] = test_top100_unit_counts_ftr_df.p0_u55 / (test_top100_unit_counts_ftr_df.p0_u55 + test_top100_unit_counts_ftr_df.p1_u55 + 0.00000001)
# test_top100_unit_counts_ftr_df['u56_ratio'] = test_top100_unit_counts_ftr_df.p0_u56 / (test_top100_unit_counts_ftr_df.p0_u56 + test_top100_unit_counts_ftr_df.p1_u56 + 0.00000001)
# test_top100_unit_counts_ftr_df['u57_ratio'] = test_top100_unit_counts_ftr_df.p0_u57 / (test_top100_unit_counts_ftr_df.p0_u57 + test_top100_unit_counts_ftr_df.p1_u57 + 0.00000001)
# test_top100_unit_counts_ftr_df['u58_ratio'] = test_top100_unit_counts_ftr_df.p0_u58 / (test_top100_unit_counts_ftr_df.p0_u58 + test_top100_unit_counts_ftr_df.p1_u58 + 0.00000001)
# test_top100_unit_counts_ftr_df['u59_ratio'] = test_top100_unit_counts_ftr_df.p0_u59 / (test_top100_unit_counts_ftr_df.p0_u59 + test_top100_unit_counts_ftr_df.p1_u59 + 0.00000001)
# test_top100_unit_counts_ftr_df['u60_ratio'] = test_top100_unit_counts_ftr_df.p0_u60 / (test_top100_unit_counts_ftr_df.p0_u60 + test_top100_unit_counts_ftr_df.p1_u60 + 0.00000001)
# test_top100_unit_counts_ftr_df['u61_ratio'] = test_top100_unit_counts_ftr_df.p0_u61 / (test_top100_unit_counts_ftr_df.p0_u61 + test_top100_unit_counts_ftr_df.p1_u61 + 0.00000001)
# test_top100_unit_counts_ftr_df['u62_ratio'] = test_top100_unit_counts_ftr_df.p0_u62 / (test_top100_unit_counts_ftr_df.p0_u62 + test_top100_unit_counts_ftr_df.p1_u62 + 0.00000001)
# test_top100_unit_counts_ftr_df['u63_ratio'] = test_top100_unit_counts_ftr_df.p0_u63 / (test_top100_unit_counts_ftr_df.p0_u63 + test_top100_unit_counts_ftr_df.p1_u63 + 0.00000001)
# test_top100_unit_counts_ftr_df['u64_ratio'] = test_top100_unit_counts_ftr_df.p0_u64 / (test_top100_unit_counts_ftr_df.p0_u64 + test_top100_unit_counts_ftr_df.p1_u64 + 0.00000001)
# test_top100_unit_counts_ftr_df['u65_ratio'] = test_top100_unit_counts_ftr_df.p0_u65 / (test_top100_unit_counts_ftr_df.p0_u65 + test_top100_unit_counts_ftr_df.p1_u65 + 0.00000001)
# test_top100_unit_counts_ftr_df['u66_ratio'] = test_top100_unit_counts_ftr_df.p0_u66 / (test_top100_unit_counts_ftr_df.p0_u66 + test_top100_unit_counts_ftr_df.p1_u66 + 0.00000001)
# test_top100_unit_counts_ftr_df['u67_ratio'] = test_top100_unit_counts_ftr_df.p0_u67 / (test_top100_unit_counts_ftr_df.p0_u67 + test_top100_unit_counts_ftr_df.p1_u67 + 0.00000001)
# test_top100_unit_counts_ftr_df['u68_ratio'] = test_top100_unit_counts_ftr_df.p0_u68 / (test_top100_unit_counts_ftr_df.p0_u68 + test_top100_unit_counts_ftr_df.p1_u68 + 0.00000001)
# test_top100_unit_counts_ftr_df['u69_ratio'] = test_top100_unit_counts_ftr_df.p0_u69 / (test_top100_unit_counts_ftr_df.p0_u69 + test_top100_unit_counts_ftr_df.p1_u69 + 0.00000001)
# test_top100_unit_counts_ftr_df['u70_ratio'] = test_top100_unit_counts_ftr_df.p0_u70 / (test_top100_unit_counts_ftr_df.p0_u70 + test_top100_unit_counts_ftr_df.p1_u70 + 0.00000001)
# test_top100_unit_counts_ftr_df['u71_ratio'] = test_top100_unit_counts_ftr_df.p0_u71 / (test_top100_unit_counts_ftr_df.p0_u71 + test_top100_unit_counts_ftr_df.p1_u71 + 0.00000001)
# test_top100_unit_counts_ftr_df['u72_ratio'] = test_top100_unit_counts_ftr_df.p0_u72 / (test_top100_unit_counts_ftr_df.p0_u72 + test_top100_unit_counts_ftr_df.p1_u72 + 0.00000001)
# test_top100_unit_counts_ftr_df['u73_ratio'] = test_top100_unit_counts_ftr_df.p0_u73 / (test_top100_unit_counts_ftr_df.p0_u73 + test_top100_unit_counts_ftr_df.p1_u73 + 0.00000001)
# test_top100_unit_counts_ftr_df['u74_ratio'] = test_top100_unit_counts_ftr_df.p0_u74 / (test_top100_unit_counts_ftr_df.p0_u74 + test_top100_unit_counts_ftr_df.p1_u74 + 0.00000001)
# test_top100_unit_counts_ftr_df['u75_ratio'] = test_top100_unit_counts_ftr_df.p0_u75 / (test_top100_unit_counts_ftr_df.p0_u75 + test_top100_unit_counts_ftr_df.p1_u75 + 0.00000001)
# test_top100_unit_counts_ftr_df['u76_ratio'] = test_top100_unit_counts_ftr_df.p0_u76 / (test_top100_unit_counts_ftr_df.p0_u76 + test_top100_unit_counts_ftr_df.p1_u76 + 0.00000001)
# test_top100_unit_counts_ftr_df['u77_ratio'] = test_top100_unit_counts_ftr_df.p0_u77 / (test_top100_unit_counts_ftr_df.p0_u77 + test_top100_unit_counts_ftr_df.p1_u77 + 0.00000001)
# test_top100_unit_counts_ftr_df['u78_ratio'] = test_top100_unit_counts_ftr_df.p0_u78 / (test_top100_unit_counts_ftr_df.p0_u78 + test_top100_unit_counts_ftr_df.p1_u78 + 0.00000001)
# test_top100_unit_counts_ftr_df['u79_ratio'] = test_top100_unit_counts_ftr_df.p0_u79 / (test_top100_unit_counts_ftr_df.p0_u79 + test_top100_unit_counts_ftr_df.p1_u79 + 0.00000001)
# test_top100_unit_counts_ftr_df['u80_ratio'] = test_top100_unit_counts_ftr_df.p0_u80 / (test_top100_unit_counts_ftr_df.p0_u80 + test_top100_unit_counts_ftr_df.p1_u80 + 0.00000001)
# test_top100_unit_counts_ftr_df['u81_ratio'] = test_top100_unit_counts_ftr_df.p0_u81 / (test_top100_unit_counts_ftr_df.p0_u81 + test_top100_unit_counts_ftr_df.p1_u81 + 0.00000001)
# test_top100_unit_counts_ftr_df['u82_ratio'] = test_top100_unit_counts_ftr_df.p0_u82 / (test_top100_unit_counts_ftr_df.p0_u82 + test_top100_unit_counts_ftr_df.p1_u82 + 0.00000001)
# test_top100_unit_counts_ftr_df['u83_ratio'] = test_top100_unit_counts_ftr_df.p0_u83 / (test_top100_unit_counts_ftr_df.p0_u83 + test_top100_unit_counts_ftr_df.p1_u83 + 0.00000001)
# test_top100_unit_counts_ftr_df['u84_ratio'] = test_top100_unit_counts_ftr_df.p0_u84 / (test_top100_unit_counts_ftr_df.p0_u84 + test_top100_unit_counts_ftr_df.p1_u84 + 0.00000001)
# test_top100_unit_counts_ftr_df['u85_ratio'] = test_top100_unit_counts_ftr_df.p0_u85 / (test_top100_unit_counts_ftr_df.p0_u85 + test_top100_unit_counts_ftr_df.p1_u85 + 0.00000001)
# test_top100_unit_counts_ftr_df['u86_ratio'] = test_top100_unit_counts_ftr_df.p0_u86 / (test_top100_unit_counts_ftr_df.p0_u86 + test_top100_unit_counts_ftr_df.p1_u86 + 0.00000001)
# test_top100_unit_counts_ftr_df['u87_ratio'] = test_top100_unit_counts_ftr_df.p0_u87 / (test_top100_unit_counts_ftr_df.p0_u87 + test_top100_unit_counts_ftr_df.p1_u87 + 0.00000001)
# test_top100_unit_counts_ftr_df['u88_ratio'] = test_top100_unit_counts_ftr_df.p0_u88 / (test_top100_unit_counts_ftr_df.p0_u88 + test_top100_unit_counts_ftr_df.p1_u88 + 0.00000001)
# test_top100_unit_counts_ftr_df['u89_ratio'] = test_top100_unit_counts_ftr_df.p0_u89 / (test_top100_unit_counts_ftr_df.p0_u89 + test_top100_unit_counts_ftr_df.p1_u89 + 0.00000001)
# test_top100_unit_counts_ftr_df['u90_ratio'] = test_top100_unit_counts_ftr_df.p0_u90 / (test_top100_unit_counts_ftr_df.p0_u90 + test_top100_unit_counts_ftr_df.p1_u90 + 0.00000001)
# test_top100_unit_counts_ftr_df['u91_ratio'] = test_top100_unit_counts_ftr_df.p0_u91 / (test_top100_unit_counts_ftr_df.p0_u91 + test_top100_unit_counts_ftr_df.p1_u91 + 0.00000001)
# test_top100_unit_counts_ftr_df['u92_ratio'] = test_top100_unit_counts_ftr_df.p0_u92 / (test_top100_unit_counts_ftr_df.p0_u92 + test_top100_unit_counts_ftr_df.p1_u92 + 0.00000001)
# test_top100_unit_counts_ftr_df['u93_ratio'] = test_top100_unit_counts_ftr_df.p0_u93 / (test_top100_unit_counts_ftr_df.p0_u93 + test_top100_unit_counts_ftr_df.p1_u93 + 0.00000001)
# test_top100_unit_counts_ftr_df['u94_ratio'] = test_top100_unit_counts_ftr_df.p0_u94 / (test_top100_unit_counts_ftr_df.p0_u94 + test_top100_unit_counts_ftr_df.p1_u94 + 0.00000001)
# test_top100_unit_counts_ftr_df['u95_ratio'] = test_top100_unit_counts_ftr_df.p0_u95 / (test_top100_unit_counts_ftr_df.p0_u95 + test_top100_unit_counts_ftr_df.p1_u95 + 0.00000001)
# test_top100_unit_counts_ftr_df['u96_ratio'] = test_top100_unit_counts_ftr_df.p0_u96 / (test_top100_unit_counts_ftr_df.p0_u96 + test_top100_unit_counts_ftr_df.p1_u96 + 0.00000001)
# test_top100_unit_counts_ftr_df['u97_ratio'] = test_top100_unit_counts_ftr_df.p0_u97 / (test_top100_unit_counts_ftr_df.p0_u97 + test_top100_unit_counts_ftr_df.p1_u97 + 0.00000001)
# test_top100_unit_counts_ftr_df['u98_ratio'] = test_top100_unit_counts_ftr_df.p0_u98 / (test_top100_unit_counts_ftr_df.p0_u98 + test_top100_unit_counts_ftr_df.p1_u98 + 0.00000001)
# test_top100_unit_counts_ftr_df['u99_ratio'] = test_top100_unit_counts_ftr_df.p0_u99 / (test_top100_unit_counts_ftr_df.p0_u99 + test_top100_unit_counts_ftr_df.p1_u99 + 0.00000001)

# train_top100_unit_counts_ftr_df = train_top100_unit_counts_ftr_df[['game_id','u0_ratio','u1_ratio','u2_ratio','u3_ratio','u4_ratio','u5_ratio','u6_ratio','u7_ratio','u8_ratio','u9_ratio','u10_ratio','u11_ratio','u12_ratio','u13_ratio','u14_ratio','u15_ratio','u16_ratio','u17_ratio','u18_ratio','u19_ratio','u20_ratio','u21_ratio','u22_ratio','u23_ratio','u24_ratio','u25_ratio','u26_ratio','u27_ratio','u28_ratio','u29_ratio','u30_ratio','u31_ratio','u32_ratio','u33_ratio','u34_ratio','u35_ratio','u36_ratio','u37_ratio','u38_ratio','u39_ratio','u40_ratio','u41_ratio','u42_ratio','u43_ratio','u44_ratio','u45_ratio','u46_ratio','u47_ratio','u48_ratio','u49_ratio','u50_ratio','u51_ratio','u52_ratio','u53_ratio','u54_ratio','u55_ratio','u56_ratio','u57_ratio','u58_ratio','u59_ratio','u60_ratio','u61_ratio','u62_ratio','u63_ratio','u64_ratio','u65_ratio','u66_ratio','u67_ratio','u68_ratio','u69_ratio','u70_ratio','u71_ratio','u72_ratio','u73_ratio','u74_ratio','u75_ratio','u76_ratio','u77_ratio','u78_ratio','u79_ratio','u80_ratio','u81_ratio','u82_ratio','u83_ratio','u84_ratio','u85_ratio','u86_ratio','u87_ratio','u88_ratio','u89_ratio','u90_ratio','u91_ratio','u92_ratio','u93_ratio','u94_ratio','u95_ratio','u96_ratio','u97_ratio','u98_ratio','u99_ratio']]
# test_top100_unit_counts_ftr_df = test_top100_unit_counts_ftr_df[['game_id','u0_ratio','u1_ratio','u2_ratio','u3_ratio','u4_ratio','u5_ratio','u6_ratio','u7_ratio','u8_ratio','u9_ratio','u10_ratio','u11_ratio','u12_ratio','u13_ratio','u14_ratio','u15_ratio','u16_ratio','u17_ratio','u18_ratio','u19_ratio','u20_ratio','u21_ratio','u22_ratio','u23_ratio','u24_ratio','u25_ratio','u26_ratio','u27_ratio','u28_ratio','u29_ratio','u30_ratio','u31_ratio','u32_ratio','u33_ratio','u34_ratio','u35_ratio','u36_ratio','u37_ratio','u38_ratio','u39_ratio','u40_ratio','u41_ratio','u42_ratio','u43_ratio','u44_ratio','u45_ratio','u46_ratio','u47_ratio','u48_ratio','u49_ratio','u50_ratio','u51_ratio','u52_ratio','u53_ratio','u54_ratio','u55_ratio','u56_ratio','u57_ratio','u58_ratio','u59_ratio','u60_ratio','u61_ratio','u62_ratio','u63_ratio','u64_ratio','u65_ratio','u66_ratio','u67_ratio','u68_ratio','u69_ratio','u70_ratio','u71_ratio','u72_ratio','u73_ratio','u74_ratio','u75_ratio','u76_ratio','u77_ratio','u78_ratio','u79_ratio','u80_ratio','u81_ratio','u82_ratio','u83_ratio','u84_ratio','u85_ratio','u86_ratio','u87_ratio','u88_ratio','u89_ratio','u90_ratio','u91_ratio','u92_ratio','u93_ratio','u94_ratio','u95_ratio','u96_ratio','u97_ratio','u98_ratio','u99_ratio']]

train_unit_select_raw_ftr_df = pd.read_csv(os.path.join(data_folder, 'unit_select_raw_train.csv'))
test_unit_select_raw_ftr_df = pd.read_csv(os.path.join(data_folder, 'unit_select_raw_test.csv'))

train_camera_homeaway_ftr_df = pd.read_csv(os.path.join(data_folder, 'camera_homeaway_train.csv'))
test_camera_homeaway_ftr_df = pd.read_csv(os.path.join(data_folder, 'camera_homeaway_test.csv'))

train_camera_center_moves_ftr_df = pd.read_csv(os.path.join(data_folder, 'camera_center_moves_train.csv'))
test_camera_center_moves_ftr_df = pd.read_csv(os.path.join(data_folder, 'camera_center_moves_test.csv'))

train_user_activity_ability_df = pd.read_csv(os.path.join(data_folder, 'user_activity_ability_train.csv'))
test_user_activity_ability_df = pd.read_csv(os.path.join(data_folder, 'user_activity_ability_test.csv'))

camera_moving_stats_train_df = pd.read_csv(os.path.join(data_folder, 'camera_moving_stats_train.csv'))
camera_moving_stats_test_df = pd.read_csv(os.path.join(data_folder, 'camera_moving_stats_test.csv'))

attack_units_cnt_train_df = pd.read_csv(os.path.join(data_folder, 'attack_units_cnt_train.csv'))
attack_units_cnt_test_df = pd.read_csv(os.path.join(data_folder, 'attack_units_cnt_test.csv'))

###### from 상혁님

# camera
train_camera_ftr_df = pd.read_csv(os.path.join(data_folder, 'train_Camera_ftr.csv'))
test_camera_ftr_df = pd.read_csv(os.path.join(data_folder, 'test_Camera_ftr.csv'))

## 비율화
# train_camera_ftr_df['player0_ratio'] = train_camera_ftr_df.player0_near0 / (train_camera_ftr_df.player0_near0 + train_camera_ftr_df.player0_near1 + 0.00000001)
# train_camera_ftr_df['player1_ratio'] = train_camera_ftr_df.player1_near1 / (train_camera_ftr_df.player1_near1 + train_camera_ftr_df.player1_near0 + 0.00000001)

# test_camera_ftr_df['player0_ratio'] = test_camera_ftr_df.player0_near0 / (test_camera_ftr_df.player0_near0 + test_camera_ftr_df.player0_near1 + 0.00000001)
# test_camera_ftr_df['player1_ratio'] = test_camera_ftr_df.player1_near1 / (test_camera_ftr_df.player1_near1 + test_camera_ftr_df.player1_near0 + 0.00000001)

# train_camera_ftr_df = train_camera_ftr_df.drop(['player0_near0','player0_near1','player1_near1','player1_near0'], axis='columns')
# test_camera_ftr_df = test_camera_ftr_df.drop(['player0_near0','player0_near1','player1_near1','player1_near0'], axis='columns')


# rightclick
train_rightclick_ftr_df = pd.read_csv(os.path.join(data_folder, 'train_Rightclick_ftr.csv'))
test_rightclick_ftr_df = pd.read_csv(os.path.join(data_folder, 'test_Rightclick_ftr.csv'))

## 비율화
# train_rightclick_ftr_df['player0_click_ratio'] = train_rightclick_ftr_df.player0_click_near0 / (train_rightclick_ftr_df.player0_click_near0 + train_rightclick_ftr_df.player0_click_near1 + 0.00000001)
# train_rightclick_ftr_df['player1_click_ratio'] = train_rightclick_ftr_df.player1_click_near1 / (train_rightclick_ftr_df.player1_click_near1 + train_rightclick_ftr_df.player1_click_near0 + 0.00000001)

# test_rightclick_ftr_df['player0_click_ratio'] = test_rightclick_ftr_df.player0_click_near0 / (test_rightclick_ftr_df.player0_click_near0 + train_rightclick_ftr_df.player0_click_near1 + 0.00000001)
# test_rightclick_ftr_df['player1_click_ratio'] = test_rightclick_ftr_df.player1_click_near1 / (test_rightclick_ftr_df.player1_click_near1 + train_rightclick_ftr_df.player1_click_near0 + 0.00000001)

# train_rightclick_ftr_df = train_rightclick_ftr_df.drop(['player0_click_near0','player0_click_near1','player1_click_near1','player1_click_near0'], axis='columns')
# test_rightclick_ftr_df = test_rightclick_ftr_df.drop(['player0_click_near0','player0_click_near1','player1_click_near1','player1_click_near0'], axis='columns')



# 파트합체
## train
# temp_df = train_top100_unit_counts_ftr_df.merge(train_ability_feature_df, how='left', on='game_id')
temp_df = train_ability_feature_df

temp_df = temp_df.merge(train_unit_select_raw_ftr_df, how='left', on='game_id')
temp_df = temp_df.merge(train_camera_ftr_df, how='left', on='game_id')
temp_df = temp_df.merge(train_rightclick_ftr_df, how='left', on='game_id')
temp_df = temp_df.merge(train_camera_homeaway_ftr_df, how='left', on='game_id')
temp_df = temp_df.merge(train_camera_center_moves_ftr_df, how='left', on='game_id')
temp_df = temp_df.merge(train_user_activity_ability_df, how='left', on='game_id')
temp_df = temp_df.merge(camera_moving_stats_train_df, how='left', on='game_id')
temp_df = temp_df.merge(attack_units_cnt_train_df, how='left', on='game_id')

train_final_ftr_df = temp_df.fillna(0)

train_final_ftr_df['map_0'] = train_final_ftr_df['map_0'].astype('int').astype('category')
train_final_ftr_df['map_1'] = train_final_ftr_df['map_1'].astype('int').astype('category')
train_final_ftr_df['species_0'] = train_final_ftr_df['species_0'].astype('int').astype('category')
train_final_ftr_df['species_1'] = train_final_ftr_df['species_1'].astype('int').astype('category')

print('train_ability_feature_df:', train_ability_feature_df.shape)

print('train_unit_select_raw_ftr_df:', train_unit_select_raw_ftr_df.shape)
print('train_camera_ftr_df:', train_camera_ftr_df.shape)
print('train_rightclick_ftr_df:', train_rightclick_ftr_df.shape)
print('train_camera_homeaway_ftr_df:', train_camera_homeaway_ftr_df.shape)
print('train_camera_center_moves_ftr_df:', train_camera_center_moves_ftr_df.shape)
print('train_user_activity_ability_df:', train_user_activity_ability_df.shape)
print('camera_moving_stats_train_df:', camera_moving_stats_train_df.shape)
print('attack_units_cnt_train_df:', attack_units_cnt_train_df.shape)

print('train_final_ftr_df:', train_final_ftr_df.shape)

print(train_final_ftr_df.columns)

## test
# temp_df = test_top100_unit_counts_ftr_df.merge(test_ability_feature_df, how='left', on='game_id')
temp_df = test_ability_feature_df

temp_df = temp_df.merge(test_unit_select_raw_ftr_df, how='left', on='game_id')
temp_df = temp_df.merge(test_camera_ftr_df, how='left', on='game_id')
temp_df = temp_df.merge(test_rightclick_ftr_df, how='left', on='game_id')
temp_df = temp_df.merge(test_camera_homeaway_ftr_df, how='left', on='game_id')
temp_df = temp_df.merge(test_camera_center_moves_ftr_df, how='left', on='game_id')
temp_df = temp_df.merge(test_user_activity_ability_df, how='left', on='game_id')
temp_df = temp_df.merge(camera_moving_stats_test_df, how='left', on='game_id')
temp_df = temp_df.merge(attack_units_cnt_test_df, how='left', on='game_id')

test_final_ftr_df = temp_df.fillna(0)

test_final_ftr_df['map_0'] = test_final_ftr_df['map_0'].astype('int').astype('category')
test_final_ftr_df['map_1'] = test_final_ftr_df['map_1'].astype('int').astype('category')
test_final_ftr_df['species_0'] = test_final_ftr_df['species_0'].astype('int').astype('category')
test_final_ftr_df['species_1'] = test_final_ftr_df['species_1'].astype('int').astype('category')

print('test_ability_feature_df:', test_ability_feature_df.shape)

print('test_unit_select_raw_ftr_df:', test_unit_select_raw_ftr_df.shape)
print('test_camera_ftr_df:', test_camera_ftr_df.shape)
print('test_rightclick_ftr_df:', test_rightclick_ftr_df.shape)
print('test_camera_homeaway_ftr_df:', test_camera_homeaway_ftr_df.shape)
print('test_camera_center_moves_ftr_df:', test_camera_center_moves_ftr_df.shape)
print('test_user_activity_ability_df:', test_user_activity_ability_df.shape)
print('camera_moving_stats_test_df:', camera_moving_stats_test_df.shape)
print('attack_units_cnt_test_df:', attack_units_cnt_test_df.shape)

print('test_final_ftr_df:', test_final_ftr_df.shape)


## output
train_final_ftr_df.to_csv(os.path.join(output_folder,'train_final_ftr_0413.csv'))
test_final_ftr_df.to_csv(os.path.join(output_folder,'test_final_ftr_0413.csv'))


# del train_ability_df
# del test_ability_df
#
# del train_ability_feature_df
# del train_camera_ftr_df
# del train_top100_unit_counts_ftr_df
# del train_rightclick_ftr_df
#
# del test_ability_feature_df
# del test_camera_ftr_df
# del test_top100_unit_counts_ftr_df
# del test_rightclick_ftr_df


