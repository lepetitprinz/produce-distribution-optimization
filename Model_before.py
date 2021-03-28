import pandas as pd
import numpy as np
import random
import sys
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras import optimizers
from keras.models import model_from_json
import warnings
warnings.filterwarnings("ignore")


ga = pd.read_csv('C:\\Users\\shchoi\\PycharmProjects\\pipeline_tomato\\mafra\\가락\\garak.csv', engine='python', encoding='cp949')
dae = pd.read_csv("C:\\Users\\shchoi\\PycharmProjects\\pipeline_tomato\\mafra\\대구\\daegu.csv")

mod = sys.modules[__name__]
random.seed(0)  # fix seed
rseed = 3

del ga['DELNG_DE']
del dae['DELNG_DE']

data_set = ga.values
ga_X = data_set[:, 0:51]
ga_Y = data_set[:, 51]

data_set = dae.values
dae_X = data_set[:, 0:51]
dae_Y = data_set[:, 51]

scaler = MinMaxScaler()
ga_X_scaled=scaler.fit_transform(ga_X)
dae_X_scaled=scaler.fit_transform(dae_X)


def ga_minmax(x: pd.DataFrame):
    # x = pd.DataFrame(x)
    gd = ga.drop('PRC', axis=1)
    # gd['PRC'] = np.nan
    # gd['MART'] = np.nan
    # gd['DELNG_QY_new'] = np.nan
    # a = (np.array(gd.describe().iloc[7])-np.array(gd.describe().iloc[3]))
    # y = np.array(x-np.array(gd.describe().iloc[3])) / a
    gd_max = np.array(gd.max())
    gd_min = np.array(gd.min())
    a = gd_max - gd_min
    a[np.where(a == 0)] = 1
    # if len(gd_min) != 51:
    #     gd_min = np.delete(gd_min, np.s_[51:], axis=0)
    if x.shape[1] != 51:
        x = x.drop(['PRC', 'MART', 'DELNG_QY_new'], axis=1)
    y = (x - gd_min) / a
    # y = y.drop(['PRC', 'MART', 'DELNG_QY_new'], axis=1)

    return y

def dae_minmax(x: pd.DataFrame) :
    gd = dae.drop('PRC', axis=1)
    # gd['PRC'] = np.nan
    # gd['MART'] = np.nan
    # gd['DELNG_QY_new'] = np.nan
    # a = (np.array(gd.describe().iloc[7]) - np.array(gd.describe().iloc[3]))
    gd_max = np.array(gd.max())
    gd_min = np.array(gd.min())
    a = gd_max - gd_min
    a[np.where(a == 0)] = 1
    # y = np.array(x - np.array(gd.describe().iloc[3])) / a
    if x.shape[1] != 51:
        x = x.drop(['PRC', 'MART', 'DELNG_QY_new'], axis=1)
    y = (x - gd_min) / a
    # y = y.drop(['PRC', 'MART', 'DELNG_QY_new'], axis=1)

    return y

def ga_rev_minmax(x: np.array):
    x_df = pd.DataFrame(x)
    gd = ga.drop('PRC', axis=1)
    # y = x * (np.array(gd.describe().iloc[7]) - np.array(gd.describe().iloc[3])) + np.array(gd.describe().iloc[3])
    gd_max = np.array(gd.max())
    gd_min = np.array(gd.min())
    y = x_df * (np.array(gd_max) - gd_min) + gd_min
    y.columns = ['AUC', 'CPR_1', 'CPR_2', 'CPR_3', 'CPR_4', 'CPR_5', 'KIND_1', 'KIND_2',
       'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7', 'SHIPMNT_1',
       'SHIPMNT_2', 'SHIPMNT_3', 'QLITY_1', 'QLITY_2', 'QLITY_3', 'QLITY_4',
       'QLITY_5', 'QLITY_6', 'QLITY_7', 'MONTH_1', 'MONTH_2', 'MONTH_3',
       'MONTH_4', 'MONTH_5', 'MONTH_6', 'MONTH_7', 'MONTH_8', 'MONTH_9',
       'MONTH_10', 'MONTH_11', 'MONTH_12', 'SUBTOTAL_1', 'SUBTOTAL_2',
       'SUBTOTAL_3', 'SUBTOTAL_4', 'SUBTOTAL_5', 'SUBTOTAL_6', 'SUBTOTAL_7',
       'CPR_SUB_1', 'CPR_SUB_2', 'CPR_SUB_3', 'CPR_SUB_4', 'CPR_SUB_5',
       'CPR_SUB_6', 'CPR_SUB_7', 'DELNG_QY', 'ex_PRC']

    return y


def dae_rev_minmax(x: np.array):
    x_df = pd.DataFrame(x)
    gd = dae.drop('PRC', axis=1)
    # y = x*(np.array(gd.describe().iloc[7]) - np.array(gd.describe().iloc[3])) + np.array(gd.describe().iloc[3])
    gd_max = np.array(gd.max())
    gd_min = np.array(gd.min())
    y = x_df * (np.array(gd_max) - gd_min) + gd_min
    y.columns = ['AUC', 'CPR_1', 'CPR_2', 'CPR_3', 'CPR_4', 'CPR_5', 'KIND_1', 'KIND_2',
       'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7', 'SHIPMNT_1',
       'SHIPMNT_2', 'SHIPMNT_3', 'QLITY_1', 'QLITY_2', 'QLITY_3', 'QLITY_4',
       'QLITY_5', 'QLITY_6', 'QLITY_7', 'MONTH_1', 'MONTH_2', 'MONTH_3',
       'MONTH_4', 'MONTH_5', 'MONTH_6', 'MONTH_7', 'MONTH_8', 'MONTH_9',
       'MONTH_10', 'MONTH_11', 'MONTH_12', 'SUBTOTAL_1', 'SUBTOTAL_2',
       'SUBTOTAL_3', 'SUBTOTAL_4', 'SUBTOTAL_5', 'SUBTOTAL_6', 'SUBTOTAL_7',
       'CPR_SUB_1', 'CPR_SUB_2', 'CPR_SUB_3', 'CPR_SUB_4', 'CPR_SUB_5',
       'CPR_SUB_6', 'CPR_SUB_7', 'DELNG_QY', 'ex_PRC']

    return y


ga_x_set, ga_x_test, ga_y_set, ga_y_test = train_test_split(ga_X_scaled, ga_Y, test_size=0.2, random_state=rseed)
ga_x_train, ga_x_val, ga_y_train, ga_y_val = train_test_split(ga_x_set, ga_y_set, test_size=0.2, random_state=rseed)

dae_x_set, dae_x_test, dae_y_set, dae_y_test = train_test_split(dae_X_scaled, dae_Y, test_size=0.2, random_state=rseed)
dae_x_train, dae_x_val, dae_y_train, dae_y_val = train_test_split(dae_x_set, dae_y_set, test_size=0.2, random_state=rseed)


ga_model = Sequential(name='sequential_1')
ga_model.add(Dense(name='dense_1',
                   trainable=True,
                   batch_input_shape=(None, 51),
                   dtype='float32',
                   units=2048,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                   kernel_regularizer={'class_name':'L1L2',
                                      'config': {'l1':0, 'l2':0.009999999776482582}},
                   bias_regularizer={'class_name':'L1L2',
                                      'config': {'l1':0, 'l2':0.009999999776482582}},
                  ))
ga_model.add(Dropout(name='dropout_1',
                     trainable=True,
                     rate=0.5))
ga_model.add(Dense(name='dense_2',
                   trainable=True,
                   units=256,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
ga_model.add(Dense(name='dense_3',
                   trainable=True,
                   units=256,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
ga_model.add(Dense(name='dense_4',
                   trainable=True,
                   units=128,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
ga_model.add(Dense(name='dense_5',
                   trainable=True,
                   units=128,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
ga_model.add(Dense(name='dense_6',
                   trainable=True,
                   units=128,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
ga_model.add(Dense(name='dense_7',
                   trainable=True,
                   units=64,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
ga_model.add(Dense(name='dense_8',
                   trainable=True,
                   units=32,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
ga_model.add(Dense(name='dense_9',
                   trainable=True,
                   units=8,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
ga_model.add(Dense(name='dense_10',
                   trainable=True,
                   units=1,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))


# g_json_file = open("/home/istream/pkg/worker/src/notebook/ba_data/mafra/가락/garak_model.json", "r")
# g_model_json = g_json_file.read()
# g_json_file.close()
# ga_model = model_from_json(g_model_json)

#adam = optimizers.Adam(lr=0.01, decay=1e-6)
adam = optimizers.Adam(lr=0.001)
#model.compile(loss='mean_squared_error', optimizer=sgd)
ga_model.compile(loss='mean_absolute_error', optimizer=adam)
ga_model.load_weights("C:\\Users\\shchoi\\PycharmProjects\\pipeline_tomato\\mafra\\가락\\mlp_0103_1710.h5")


dae_model = Sequential(name='sequential_2')
dae_model.add(Dense(name='dense_11',
                   trainable=True,
                   batch_input_shape=(None, 51),
                   dtype='float32',
                   units=2048,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                   kernel_regularizer={'class_name':'L1L2',
                                      'config': {'l1':0, 'l2':0.009999999776482582}},
                   bias_regularizer={'class_name':'L1L2',
                                      'config': {'l1':0, 'l2':0.009999999776482582}},
                  ))
dae_model.add(Dropout(name='dropout_7',
                     trainable=True,
                     rate=0.5))
dae_model.add(Dense(name='dense_12',
                   trainable=True,
                   units=1024,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
dae_model.add(Dropout(name='dropout_8',
                     trainable=True,
                     rate=0.5))
dae_model.add(Dense(name='dense_13',
                   trainable=True,
                   units=256,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
dae_model.add(Dense(name='dense_14',
                   trainable=True,
                   units=128,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
dae_model.add(Dense(name='dense_15',
                   trainable=True,
                   units=128,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
dae_model.add(Dense(name='dense_16',
                   trainable=True,
                   units=128,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
dae_model.add(Dense(name='dense_17',
                   trainable=True,
                   units=64,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
dae_model.add(Dense(name='dense_18',
                   trainable=True,
                   units=32,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
dae_model.add(Dense(name='dense_19',
                   trainable=True,
                   units=8,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))
dae_model.add(Dense(name='dense_20',
                   trainable=True,
                   units=1,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer={'class_name':'VarianceScaling',
                                       'config':{'scale':1, 'mode': 'fan_avg',
                                                 'distribution':'uniform', 'seed':None}},
                   bias_initializer={'class_name':'Zeros',
                                     'config': {}},
                  ))


# d_json_file = open("/home/istream/pkg/worker/src/notebook/ba_data/mafra/대구/MLP_0213_1700_deagu_model.json", "r")
# d_model_json = d_json_file.read()
# d_json_file.close()
# dae_model = model_from_json(d_model_json)

#adam = optimizers.Adam(lr=0.01, decay=1e-6)
adam = optimizers.Adam(lr=0.001)
#model.compile(loss='mean_squared_error', optimizer=sgd)
dae_model.compile(loss='mean_absolute_error', optimizer=adam)
dae_model.load_weights("C:\\Users\\shchoi\\PycharmProjects\\pipeline_tomato\\mafra\\대구\\MLP_0213_1700.h5")


ga_model.evaluate(ga_x_test, ga_y_test)
dae_model.evaluate(dae_x_test, dae_y_test)


# 입력값
pred_0 = ga_X_scaled[72520:72593]  ### 2018-12-31

model_0 = ga_model
minmax_0 = ga_minmax
rev_minmax_0 = ga_rev_minmax

pred_1 = dae_X_scaled[75341:75410]  ### 2018-12-31

model_1 = dae_model
minmax_1 = dae_minmax
rev_minmax_1 = dae_rev_minmax

alpha=0.05
profit_list = []
num_market = 2
patience = 64
blind = 0


# 입력값 계산 함수 정의
def subtotal(x):
    x_sub = x[['KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7', 'DELNG_QY']]
    x_sub_groupby = x_sub.groupby(['KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7'],
                                  as_index=False).sum()

    for i in range(1, 8):
        try:
            x['SUBTOTAL_{}'.format(i)] = x_sub_groupby[x_sub_groupby['KIND_{}'.format(i)] == 1]['DELNG_QY'].values[0]
        except:
            continue

    return x

def cpr(x):
    x_cpr_sub = x[
        ['KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7', 'CPR_1', 'CPR_2', 'CPR_3', 'CPR_4',
         'CPR_5', 'DELNG_QY']]
    x_cpr_groupby = x_cpr_sub.groupby(
        ['KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7', 'CPR_1', 'CPR_2', 'CPR_3', 'CPR_4',
         'CPR_5'], as_index=False).sum()

    for j in range(1, 8):
        for i in range(len(x)):
            if x['CPR_1'][i] == 1:
                a = x_cpr_groupby[(x_cpr_groupby['CPR_1'] == 1) & (x_cpr_groupby['KIND_{}'.format(j)] == 1)]['DELNG_QY']
                try:
                    x['CPR_SUB_{}'.format(j)][i] = np.array(a)[0]
                except:
                    break
            elif x['CPR_2'][i] == 1:
                a = x_cpr_groupby[(x_cpr_groupby['CPR_2'] == 1) & (x_cpr_groupby['KIND_{}'.format(j)] == 1)]['DELNG_QY']
                try:
                    x['CPR_SUB_{}'.format(j)][i] = np.array(a)[0]
                except:
                    break
            elif x['CPR_3'][i] == 1:
                a = x_cpr_groupby[(x_cpr_groupby['CPR_3'] == 1) & (x_cpr_groupby['KIND_{}'.format(j)] == 1)]['DELNG_QY']
                try:
                    x['CPR_SUB_{}'.format(j)][i] = np.array(a)[0]
                except:
                    break
            elif x['CPR_4'][i] == 1:
                a = x_cpr_groupby[(x_cpr_groupby['CPR_4'] == 1) & (x_cpr_groupby['KIND_{}'.format(j)] == 1)]['DELNG_QY']
                try:
                    x['CPR_SUB_{}'.format(j)][i] = np.array(a)[0]
                except:
                    break
            else:
                a = x_cpr_groupby[(x_cpr_groupby['CPR_5'] == 1) & (x_cpr_groupby['KIND_{}'.format(j)] == 1)]['DELNG_QY']
                try:
                    x['CPR_SUB_{}'.format(j)][i] = np.array(a)[0]
                except:
                    break

    return x


model_list = [model_0, model_1]
minmax_list = [minmax_0, minmax_1]
rev_minmax_list = [rev_minmax_0, rev_minmax_1]
pred_list = [pred_0, pred_1]


def pred(model_list: list, rev_minmax_list: list, pred_list:list):
    pred_price_dict = {}
    df_pred_dict = {}
    sel_dict = {}
    test_dict = {}
    best_DELNG_dict = {}
    new_pred_dict = {}

    profit_list = []
    for cnt in range(100):
        for i in range(num_market):
            pred_price_val = model_list[i].predict(pred_list[i])
            pred_price_dict.update({i: pred_price_val})
            df_pred_val = rev_minmax_list[i](pred_list[i])
            df_pred_val['PRC'] = pred_price_val
            df_pred_val['MART'] = i
            df_pred_dict.update({i: df_pred_val})

            sel_dict_val = df_pred_val[['AUC', 'KIND_1', 'KIND_2','KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7',
                                           'SHIPMNT_1', 'SHIPMNT_2', 'SHIPMNT_3', 'QLITY_1', 'QLITY_2', 'QLITY_3', 'QLITY_4',
                                           'QLITY_5', 'QLITY_6', 'QLITY_7','DELNG_QY', 'PRC', 'MART']]
            sel_dict.update({i: sel_dict_val})


        concat_list = []
        for i in range(num_market):
            concat_list.append(sel_dict[i])

        total = pd.concat(concat_list)
        cols = ['AUC', 'KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7',
                'SHIPMNT_1', 'SHIPMNT_2', 'SHIPMNT_3', 'QLITY_1', 'QLITY_2', 'QLITY_3', 'QLITY_4',
                'QLITY_5', 'QLITY_6', 'QLITY_7', 'DELNG_QY', 'PRC', 'MART']
        t_sel = pd.DataFrame(columns=cols)

        profit = 0
        for k in range(2):
            auc = total[(total['AUC'] == k)]
            for l in range(1, 4):
                ship = auc[(auc['SHIPMNT_{}'.format(l)] == 1)]
                for i in range(1, 8):
                    KIND = ship[(ship['KIND_{}'.format(i)] == 1)]
                    for j in range(1, 8):
                        QLITY = KIND[(KIND['QLITY_{}'.format(j)] == 1)]
                        if QLITY['DELNG_QY'].sum() != 0:
                            del_sum = QLITY['DELNG_QY'].sum()
                            r = QLITY['DELNG_QY'] / del_sum
                            profit += np.matmul(np.array(QLITY['DELNG_QY']), QLITY['PRC'])

                            new_r = r + alpha * (QLITY['PRC'] / QLITY['PRC'].mean() - 1)
                            new_r = np.array(new_r)
                            nega = 0
                            for i in range(len(new_r)):    ## 음수 비율 제거
                                if new_r[i] <=0:
                                    nega += new_r[i]
                                    new_r[i] = 0
                            new_r[np.argmax(new_r)] += nega
                            QLITY['DELNG_QY'] = np.round(del_sum * new_r)
                            t_sel = t_sel.append(QLITY)
                            # t_sel = pd.concat([t_sel, QLITY])

        t_sel.rename(columns={"DELNG_QY": "DELNG_QY_new"}, inplace=True)

        for i in range(num_market):
            test_dict_val = pd.merge(df_pred_dict[i], t_sel[t_sel['MART'] == i], how='left')
            # test_dict_val= pd.concat([t_sel[t_sel['MART']==i], df_pred_dict[i]])
            test_dict_val['DELNG_QY'] = test_dict_val['DELNG_QY_new']
            test_dict_val.drop('DELNG_QY_new', axis=1)
            test_dict_val.drop(['PRC', 'MART'], axis=1)
            test_dict.update({i: test_dict_val})


        print('Total profit : ', profit)
        if len(profit_list) > 0 and profit_list[-1] > profit:
            break
        else:
            profit_list.append(profit)
            for i in range(num_market):
                best_DELNG_val = pred_list[i]
                best_DELNG_dict.update({i: best_DELNG_val})

        for i in range(num_market):
            value = test_dict[i]
            value = value.reset_index(drop=True)
            value = subtotal(value)
            value = cpr(value)
            test_dict.update({i: value})
            pred_list[i] = minmax_list[i](test_dict[i])
            new_pred_dict.update({i: minmax_list[i](test_dict[i])})

        for i in range(num_market):
            pred_price_dict[i] = model_list[i].predict(new_pred_dict[i])
            new_df_pred_val = rev_minmax_list[i](new_pred_dict[i])
            new_df_pred_val['PRC'] = pred_price_dict[i]
            new_df_pred_val['MART'] = i
            df_pred_dict[i]['new_PRC'] = new_df_pred_val['PRC']
            new_col = ['AUC','KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7',
                       'SHIPMNT_1','SHIPMNT_2', 'SHIPMNT_3', 'QLITY_1', 'QLITY_2', 'QLITY_3', 'QLITY_4',
                       'QLITY_5', 'QLITY_6', 'QLITY_7','DELNG_QY', 'PRC','MART','new_PRC']
            sel_dict[i] = df_pred_dict[i][new_col]

        concat_list = []
        for i in range(num_market):
            concat_list.append(sel_dict[i])

        total = pd.concat(concat_list)
        total['include'] = 1
        t_sel = pd.DataFrame(columns=['AUC', 'KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7',
                                      'SHIPMNT_1','SHIPMNT_2', 'SHIPMNT_3', 'QLITY_1', 'QLITY_2', 'QLITY_3', 'QLITY_4',
                                      'QLITY_5', 'QLITY_6', 'QLITY_7','DELNG_QY', 'PRC','MART','new_PRC','include'])

        for k in range(2):
            auc = total[(total['AUC'] == k) ]
            for l in range(1, 4):
                ship = auc[(auc['SHIPMNT_{}'.format(l)] == 1)]
                for i in range(1, 8):
                    KIND = ship[(ship['KIND_{}'.format(i)] == 1)]
                    for j in range(1, 8):
                        QLITY = KIND[ (KIND['QLITY_{}'.format(j)] == 1)]
                        QLITY = QLITY.reset_index(drop=True)
                        if len(QLITY) == 1:
                            t_sel=t_sel.append(QLITY)
                        elif QLITY['DELNG_QY'].sum() != 0:
                            idx_list = QLITY[QLITY['PRC'] >= QLITY['PRC'].mean()].index
                            for idx in idx_list:
                                if QLITY['PRC'][idx] < QLITY['new_PRC'][idx]:
                                    QLITY['include'][idx] = 0
                            del_sum=QLITY[QLITY['include'] == 1]['DELNG_QY'].sum()

                            r = QLITY[QLITY['include'] == 1]['DELNG_QY'] / del_sum
                            new_r = r + alpha * (QLITY[QLITY['include'] == 1]['PRC'] / QLITY[QLITY['include'] == 1]['PRC'].mean()-1)
                            new_r = np.array(new_r)
                            nega = 0
                            for i in range(len(new_r)):
                                if new_r[i] <= 0:
                                    nega += new_r[i]
                                    new_r[i] = 0
                            new_r[np.argmax(new_r)] += nega
                            asdf = QLITY[QLITY['include'] == 1]
                            asdf['DELNG_QY'] = np.round(del_sum * new_r)
                            QLITY = asdf.append(QLITY[QLITY['include'] == 0])
                            t_sel = t_sel.append(QLITY)
                            # t_sel = pd.concat([t_sel, QLITY])

        t_sel.rename(columns={"DELNG_QY": "DELNG_QY_new"}, inplace=True)

        for i in range(num_market):
            test_dict[i] = pd.merge(df_pred_dict[i], t_sel[t_sel['MART'] == i], how='left')
            # test_dict_val= pd.concat([t_sel[t_sel['MART']==i], df_pred_dict[i]])
            test_dict[i]['DELNG_QY'] = test_dict[i]['DELNG_QY_new']
            test_dict[i] = test_dict[i].drop('DELNG_QY_new', axis=1)
            idx = test_dict[i][test_dict[i]['DELNG_QY'] == 0].index
            test_dict[i] = test_dict[i].drop(idx)
            test_dict[i] = test_dict[i].drop(['PRC', 'MART', 'new_PRC', 'include'], axis=1)

        for i in range(num_market):
            test_dict[i] = test_dict[i].reset_index(drop=True)
            test_dict[i] = subtotal(test_dict[i])
            test_dict[i] = cpr(test_dict[i])
            pred_list[i] = minmax_list[i](test_dict[i])

    return best_DELNG_dict


best_DELNG_dict = pred(model_list=model_list, rev_minmax_list=rev_minmax_list, pred_list=pred_list)

# 최적의 물동량
ga_opt = ga_rev_minmax(best_DELNG_dict[0])
dae_opt = dae_rev_minmax(best_DELNG_dict[1])

# 조절 이전 물동량
ga_befor=ga_rev_minmax(ga_X_scaled[72520:72593])
dae_befor=dae_rev_minmax(dae_X_scaled[75341:75410])


print("\n[경매, 협동계통, 토마토 , 10kg , 특]")

ga_befor_val = ga_befor[(ga_befor['AUC'] == 0) & (ga_befor['KIND_1']==1) & (ga_befor['SHIPMNT_1'] == 1) & (ga_befor['QLITY_1'] == 1)][['CPR_1','CPR_2','CPR_3','CPR_4','CPR_5','KIND_1','DELNG_QY']]
print(f'가락 조절 이전 : {ga_befor_val}')
dae_befor_val = dae_befor[(dae_befor['AUC'] == 0) & (dae_befor['KIND_1']==1) & (dae_befor['SHIPMNT_1'] == 1) & (dae_befor['QLITY_1'] == 1)][['CPR_1','CPR_2','CPR_3','CPR_4','CPR_5','KIND_1','DELNG_QY']]
print(f'대구 조절 이전 : {dae_befor_val}')
ga_opt_val = ga_opt[(ga_opt['AUC'] == 0) & (ga_opt['KIND_1']==1) & (ga_opt['SHIPMNT_1'] == 1) & (ga_opt['QLITY_1'] == 1)][['CPR_1','CPR_2','CPR_3','CPR_4','CPR_5','KIND_1','DELNG_QY']]
print(f'가락 조절 이후 : {ga_opt_val}')
dae_opt_val = dae_opt[(dae_opt['AUC'] == 0) & (dae_opt['KIND_1']==1) & (dae_opt['SHIPMNT_1'] == 1) & (dae_opt['QLITY_1'] == 1)][['CPR_1','CPR_2','CPR_3','CPR_4','CPR_5','KIND_1','DELNG_QY']]
print(f'대구 조절 이후 : {dae_opt_val}')

print("\n[경매, 협동계통, 원형방울토마토 , 5kg , 특]")

ga_befor_val = ga_befor[(ga_befor['AUC'] == 0) & (ga_befor['KIND_4']==1) & (ga_befor['SHIPMNT_1'] == 1) & (ga_befor['QLITY_1'] == 1)][['CPR_1','CPR_2','CPR_3','CPR_4','CPR_5','KIND_2','DELNG_QY']]
print(f'가락 조절 이전 : {ga_befor_val}')
dae_befor_val = dae_befor[(dae_befor['AUC'] == 0) & (dae_befor['KIND_4']==1) & (dae_befor['SHIPMNT_1'] == 1) & (dae_befor['QLITY_1'] == 1)][['CPR_1','CPR_2','CPR_3','CPR_4','CPR_5','KIND_2','DELNG_QY']]
print(f'대구 조절 이전 : {dae_befor_val}')
ga_opt_val = ga_opt[(ga_opt['AUC'] == 0) & (ga_opt['KIND_4']==1) & (ga_opt['SHIPMNT_1'] == 1) & (ga_opt['QLITY_1'] == 1)][['CPR_1','CPR_2','CPR_3','CPR_4','CPR_5','KIND_2','DELNG_QY']]
print(f'가락 조절 이후 : {ga_opt_val}')
dae_opt_val = dae_opt[(dae_opt['AUC'] == 0) & (dae_opt['KIND_4']==1) & (dae_opt['SHIPMNT_1'] == 1) & (dae_opt['QLITY_1'] == 1)][['CPR_1','CPR_2','CPR_3','CPR_4','CPR_5','KIND_1','DELNG_QY']]
print(f'대구 조절 이후 : {dae_opt_val}')


# profit before update
print('가락시장 조절 이전 수익 :')
ga_pred_x=ga_minmax(ga_X[72520:72593])
pred_price_x=ga_model.predict(ga_pred_x)
np.matmul(np.array(ga[72520:72593]['DELNG_QY']),pred_price_x.flatten())

# profit after update
print('가락시장 조절 이후 수익 :')
ga_pred_price=ga_model.predict(best_DELNG_0)
np.matmul(np.array(ga_rev_minmax(best_DELNG_0)['DELNG_QY']),ga_pred_price.flatten())

# profit before update
print('대구시장 조절 이전 수익 :')
dae_pred_x=dae_minmax(dae_X[72520:72593])
pred_price_x=dae_model.predict(dae_pred_x)
np.matmul(np.array(dae[72520:72593]['DELNG_QY']),pred_price_x.flatten())

# profit after update
print('대구시장 조절 이후 수익 :')
dae_pred_price=dae_model.predict(best_DELNG_1)
np.matmul(np.array(dae_rev_minmax(best_DELNG_1)['DELNG_QY']),dae_pred_price.flatten())
