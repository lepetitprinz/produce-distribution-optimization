# -*-coding: utf-8-*-

import os
import copy
import numpy as np
import pandas as pd

from itertools import product
from sklearn.preprocessing import OneHotEncoder

from Dao import Dao


class Preprocessing(object):

    # Categorical 변수 인코딩 사전
    KIND_VALUES = [1, 2, 3, 4, 5, 6, 7]     # 현재 방울토마토에 대한 값 목록
    SHIPMENT_MAP = {'협동': 1, '계통': 1, '개별': 2, '상인': 2, '수입': 3}
    QLITY_MAP = {'특': 1, '상': 2, '보통': 3, '무등급': 4, '등외': 5, '무농약': 6,
                 '4등': 7, '5등': 7, '6등': 7, '7등': 7, '8등': 7}
    AUC_MAP = {'경매': 0, '정가수의': 1, '매수도매': 2, '자가계산': 3}
    SAVE_PATH = 'preprocessed_data.csv'

    # Distribution
    QLITY_CODE_MAP = {'11': 1, '12': 2, '13': 3, '1Z': 4, '19': 5, '1C': 6,
                 '14': 7, '15': 7, '16': 7, '17': 7, '18': 7}
    SHIPMENT_CODE_MAP = {'1': 1, '4': 1, '5': 2, '3': 2, '2': 3}

    def __init__(self):
        self._dir: str = ''
        self.org_data: pd.DataFrame = None

    # def set_dir(self, path: str):
    #     self._dir = path

    # def set_org_data(self, data: pd.DataFrame):
    #     self.org_data = data

    # def load_finished(self, name: str):
    #     return pd.read_csv(self._find_csv_path(name))
    #
    # def _find_csv_path(self, name: str):
    #     csvs: list = []
    #     for walk in os.walk(self._dir):
    #         if os.path.basename(walk[0]) != name:
    #             continue
    #         for filename in walk[2]:
    #             if os.path.splitext(filename)[-1].lower() == '.csv':
    #                 csvs.append(os.path.join(walk[0], filename))
    #
    #     if not csvs:
    #         return ''
    #
    #     csv_path: str = csvs[0]
    #     return csv_path

    def run(self, input_df: pd.DataFrame):

        # Data Access Object ( to D/B )
        dao: Dao = Dao.instance()

        # 전처리 - 1. 필터링 처리 : 토마토 데이터만 필터링
        df_filtered: pd.DataFrame = self.get_prod_filtered(data=input_df)

        # 전처리 - 2. 문자열 카테고리 변수를 숫자 값으로 인코딩
        corp_map: dict = dao.get_corp_to_num_map()
        df_categorized: pd.DataFrame = self.category_to_number(data=df_filtered, corp_map=corp_map)

        # 전처리 - 3.
        df_std: pd.DataFrame = self.feat_engr_1(data=df_categorized)

        # 전처리 - 4.
        df_engred_2 = self.feat_engr_2(data_std=df_std)

        # 전처리 - 5.
        df_engred_3 = self.feat_engr_3(data_std=df_std, data_engred_2=df_engred_2)

        # 전처리 - 6.
        df_std = self.feat_engr_4(data_std=df_std, corp_map=corp_map)

        # 전처리 - 7.
        df_engred_5, encoded_col_list = self.feat_engr_5(data_std=df_std, data_engred_3=df_engred_3)

        # 전처리 - 8. 최종 전처리 산출물 정리
        df_final = self.feat_engr_final(data_engred_5=df_engred_5, encoded_col_list=encoded_col_list)

        return df_final

    def get_prod_filtered(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame 필터링 처리
            - 전체 데이터에서 토마토만
        :param data:
        :return:
        """

        idx_1 = data[(data['STD_PRDLST_NEW_NM'] == '토마토') & (data['DELNG_PRUT'] == 10)].index
        idx_2 = data[(data['STD_PRDLST_NEW_NM'] == '토마토') & (data['DELNG_PRUT'] == 5)].index
        idx_3 = data[(data['STD_PRDLST_NEW_NM'] == '토마토') & (data['DELNG_PRUT'] == 4)].index
        idx_4 = data[(data['STD_PRDLST_NEW_NM'] == '방울토마토') & (data['STD_SPCIES_NEW_NM'] != '대추방울') & (
                    data['DELNG_PRUT'] == 5)].index
        idx_5 = data[(data['STD_PRDLST_NEW_NM'] == '방울토마토') & (data['STD_SPCIES_NEW_NM'] != '대추방울') & (
                    data['DELNG_PRUT'] == 2)].index
        idx_6 = data[(data['STD_PRDLST_NEW_NM'] == '방울토마토') & (data['STD_SPCIES_NEW_NM'] == '대추방울') & (
                    data['DELNG_PRUT'] == 3)].index
        idx_7 = data[(data['STD_PRDLST_NEW_NM'] == '방울토마토') & (data['STD_SPCIES_NEW_NM'] == '대추방울') & (
                    data['DELNG_PRUT'] == 2)].index

        temp = pd.concat([data.loc[idx_1], data.loc[idx_2], data.loc[idx_3],
                          data.loc[idx_4], data.loc[idx_5], data.loc[idx_6], data.loc[idx_7]])
        drop_idx = temp[(temp['AUC_SE_NM'] != '경매') & (temp['AUC_SE_NM'] != '정가수의')].index
        temp = temp.drop(index=drop_idx)
        temp = temp.reset_index(drop=True)

        return temp

    def category_to_number(self, data: pd.DataFrame, corp_map) -> pd.DataFrame:
        """
        Category 데이터 넘버링 변환
        :param data:
        :param corp_map:
        :return:
        """

        # 법인
        data['CPR'] = data['INSTT_NEW_NM'].map(corp_map)

        # 품목
        kind = []
        data['KIND'] = np.nan
        for i in range(len(data)):
            if data['STD_PRDLST_NEW_NM'][i] == '토마토':
                if data['DELNG_PRUT'][i] == 10:
                    kind.append(1)
                    # data['kind'] = 1
                elif data['DELNG_PRUT'][i] == 5:
                    kind.append(2)
                    # data['kind'] = 2
                elif data['DELNG_PRUT'][i] == 4:
                    kind.append(3)
                    # data['kind'] = 3
            elif data['STD_PRDLST_NEW_NM'][i] == '방울토마토':
                if data['STD_SPCIES_NEW_NM'][i] != '대추방울':
                    if data['DELNG_PRUT'][i] == 5:
                        kind.append(4)
                        # data['kind'] = 4
                    elif data['DELNG_PRUT'][i] == 2:
                        kind.append(5)
                        # data['kind'] = 5
                else:
                    if data['DELNG_PRUT'][i] == 3:
                        kind.append(6)
                        # data['kind'] = 6
                    elif data['DELNG_PRUT'][i] == 2:
                        kind.append(7)
                        # data['kind'] = 7

        data['KIND'] = kind
        # kind_frame = pd.DataFrame(kind)
        # kind_frame.columns = ['KIND']
        # data['KIND'] = kind_frame['KIND']

        # 출하구분
        data['SHIPMNT'] = data['SHIPMNT_SE_NM'].map(self.SHIPMENT_MAP)

        # 등급
        data['QULITY'] = data['STD_QLITY_NEW_NM'].map(self.QLITY_MAP)

        # 경매구분
        data['AUC'] = data['AUC_SE_NM'].map(self.AUC_MAP)

        return data

    def feat_engr_1(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        파생 Data의 기준이 되는 데이터 생성
            -
        :return:
        """

        data = data.dropna(how='any')
        data = data.reset_index(drop=True)
        data['DELNG_QY'] = data['DELNG_QY'].abs()

        select_feature = ['DELNG_DE', 'AUC', 'CPR', 'KIND', 'QULITY', 'SHIPMNT', 'DELNG_QY', 'SBID_PRIC']
        data_std = data[select_feature]
        temp = copy.deepcopy(data_std)

        del data_std['SBID_PRIC']
        data_std = data_std.groupby(['DELNG_DE', 'AUC', 'CPR', 'KIND', 'QULITY', 'SHIPMNT'], as_index=False).sum()
        # print(data_std.shape)
        temp_grouped = temp.groupby(['DELNG_DE', 'AUC', 'CPR', 'KIND', 'QULITY', 'SHIPMNT'], as_index=False)
        weighted_avg = lambda x: np.average(x['SBID_PRIC'], weights=x['DELNG_QY'])
        data_std['PRC'] = list(temp_grouped.apply(weighted_avg))

        return data_std

    def feat_engr_2(self, data_std: pd.DataFrame):
        temp = data_std[['DELNG_DE', 'KIND', 'DELNG_QY']]
        data_1 = temp.groupby(['DELNG_DE', 'KIND'], as_index=False).sum()
        data_1 = data_1.rename(columns={"DELNG_QY": "subtotal"})

        temp_1 = pd.DataFrame(columns=['DELNG_DE'])
        temp_1['DELNG_DE'] = temp.groupby(['DELNG_DE'], as_index=False).sum()['DELNG_DE']

        columns = ['SUBTOTAL_1', 'SUBTOTAL_2', 'SUBTOTAL_3', 'SUBTOTAL_4', 'SUBTOTAL_5', 'SUBTOTAL_6', 'SUBTOTAL_7']
        temp_2 = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

        data_2 = pd.concat([temp_1, temp_2], axis=1)
        data_2.fillna(0, inplace=True)

        data_1_days = data_1['DELNG_DE']
        data_2_days = data_2['DELNG_DE']

        for i, data_2_day in enumerate(data_2_days):
            for j, data_1_day in enumerate(data_1_days):
                if data_2_day == data_1_day:
                    if data_1['KIND'][j] == 1:
                        data_2['SUBTOTAL_1'][i] = data_1['subtotal'][j]
                    elif data_1['KIND'][j] == 2:
                        data_2['SUBTOTAL_2'][i] = data_1['subtotal'][j]
                    elif data_1['KIND'][j] == 3:
                        data_2['SUBTOTAL_3'][i] = data_1['subtotal'][j]
                    elif data_1['KIND'][j] == 4:
                        data_2['SUBTOTAL_4'][i] = data_1['subtotal'][j]
                    elif data_1['KIND'][j] == 5:
                        data_2['SUBTOTAL_5'][i] = data_1['subtotal'][j]
                    elif data_1['KIND'][j] == 6:
                        data_2['SUBTOTAL_6'][i] = data_1['subtotal'][j]
                    elif data_1['KIND'][j] == 7:
                        data_2['SUBTOTAL_7'][i] = data_1['subtotal'][j]

        data_renew = pd.merge(data_std, data_2, how='left', on='DELNG_DE')
        return data_renew

    def feat_engr_3(self, data_std: pd.DataFrame, data_engred_2: pd.DataFrame):
        temp = data_std[['DELNG_DE', 'KIND', 'CPR', 'DELNG_QY']]
        df_1 = temp.groupby(['DELNG_DE', 'KIND', 'CPR'], as_index=False).sum()
        df_1 = df_1.rename(columns={"DELNG_QY": "subtotal"})

        temp_1 = df_1[['DELNG_DE', 'KIND', 'CPR']]
        columns = ['CPR_SUB_1', 'CPR_SUB_2', 'CPR_SUB_3', 'CPR_SUB_4', 'CPR_SUB_5', 'CPR_SUB_6', 'CPR_SUB_7']
        temp_2 = pd.DataFrame(np.zeros((temp_1.shape[0], len(columns))), columns=columns)
        df_2 = pd.concat([temp_1, temp_2], axis=1)

        df_1_days = df_1['DELNG_DE']
        df_2_days = df_2['DELNG_DE']

        for i, df_2_day in enumerate(df_2_days):
            for j, df_1_day in enumerate(df_1_days):
                if df_2_day == df_1_day:
                    if df_1['KIND'][j] == 1 and df_1['CPR'][j] == df_2['CPR'][i]:
                        df_2['CPR_SUB_1'][i] = df_1['subtotal'][j]
                    elif df_1['KIND'][j] == 2 and df_1['CPR'][j] == df_2['CPR'][i]:
                        df_2['CPR_SUB_2'][i] = df_1['subtotal'][j]
                    elif df_1['KIND'][j] == 3 and df_1['CPR'][j] == df_2['CPR'][i]:
                        df_2['CPR_SUB_3'][i] = df_1['subtotal'][j]
                    elif df_1['KIND'][j] == 4 and df_1['CPR'][j] == df_2['CPR'][i]:
                        df_2['CPR_SUB_4'][i] = df_1['subtotal'][j]
                    elif df_1['KIND'][j] == 5 and df_1['CPR'][j] == df_2['CPR'][i]:
                        df_2['CPR_SUB_5'][i] = df_1['subtotal'][j]
                    elif df_1['KIND'][j] == 6 and df_1['CPR'][j] == df_2['CPR'][i]:
                        df_2['CPR_SUB_6'][i] = df_1['subtotal'][j]
                    elif df_1['KIND'][j] == 7 and df_1['CPR'][j] == df_2['CPR'][i]:
                        df_2['CPR_SUB_7'][i] = df_1['subtotal'][j]

        data_engred_3 = pd.merge(data_engred_2, df_2, how='left', on=['DELNG_DE', 'KIND', 'CPR'])

        return data_engred_3

    def feat_engr_4(self, data_std: pd.DataFrame, corp_map):
        kind = np.arange(1, 8, 1)
        cpr = np.arange(1, len(corp_map) + 1, 1)
        prd = list(product(kind, cpr))
        ex = pd.DataFrame(prd, columns=['KIND', 'CPR'])
        ex['ex'] = 0

        temp_1 = data_std.groupby(['DELNG_DE', 'CPR', 'KIND'], as_index=False).sum()
        temp_1 = temp_1[['DELNG_DE', 'CPR', 'KIND']]

        temp_2 = copy.deepcopy(data_std)
        temp_2_group = temp_2.groupby(['DELNG_DE', 'CPR', 'KIND'], as_index=False)
        weighted_avg_func = lambda x: np.average(x['PRC'], weights=x['DELNG_QY'])
        temp_1['PRC'] = list(temp_2_group.apply(weighted_avg_func))
        data_std['ex_PRC'] = 0

        for i in range(len(data_std)):
            temp_3 = ex[(ex['KIND'] == data_std['KIND'][i]) & (ex['CPR'] == data_std['CPR'][i])]['ex']
            data_std['ex_PRC'][i] = temp_3

            try:
                if data_std['KIND'][i] != data_std['KIND'][i + 1]:
                    temp = temp_1[(temp_1['DELNG_DE'] == data_std['DELNG_DE'][i]) &
                                  (temp_1['CPR'] == data_std['CPR'][i])
                                  & (temp_1['KIND'] == data_std['KIND'][i])]['PRC']
                    idx = ex[(ex['KIND'] == data_std['KIND'][i]) & (ex['CPR'] == data_std['CPR'][i])].index
                    ex['ex'][idx] = temp
            except:
                continue

        return data_std

    def feat_engr_5(self, data_std: pd.DataFrame, data_engred_3: pd.DataFrame):
        """
        One-Hot Encoder
        :param data_std:
        :param data_engred_3:
        :return:
        """

        drop_idx = data_std[data_std['ex_PRC'] == 0].index

        # Original Data
        data_std = data_std.drop(drop_idx)
        data_std = data_std.reset_index(drop=True)

        data_engred_3 = data_engred_3.drop(drop_idx)
        data_engred_3 = data_engred_3.reset_index(drop=True)
        data_engred_3['ex_PRC'] = data_std['ex_PRC']

        # Onehot Encoding
        encoded_col_name = []
        onehot_encoder = OneHotEncoder(sparse=False)

        # Corporation
        cpr_list = data_engred_3['CPR'].values.reshape(len(data_engred_3), 1)
        cpr_encoded = onehot_encoder.fit_transform(cpr_list)
        cpr_cols = ['CPR_' + str(i + 1) for i in range(cpr_encoded.shape[1])]
        encoded_col_name.extend(cpr_cols)
        cpr_encoded_df = pd.DataFrame(cpr_encoded, columns=cpr_cols)
        data_engred_3 = pd.concat([data_engred_3, cpr_encoded_df], axis=1, sort=False)
        # print('cpr_encoded: ', cpr_encoded.shape)

        # Kind
        kind_list = data_engred_3['KIND'].values.reshape(len(data_engred_3), 1)
        kind_encoded = onehot_encoder.fit_transform(kind_list)
        # kind_encoded = self.one_hot_kind(kind_list)
        kind_cols = ['KIND_' + str(i + 1) for i in range(kind_encoded.shape[1])]
        encoded_col_name.extend(kind_cols)
        kind_encoded_df = pd.DataFrame(kind_encoded, columns=kind_cols)
        data_engred_3 = pd.concat([data_engred_3, kind_encoded_df], axis=1, sort=False)
        #  print('kind_encoded: ', kind_encoded.shape)

        # Ship
        ship_list = data_engred_3['SHIPMNT'].values.reshape(len(data_engred_3), 1)
        ship_encoded = onehot_encoder.fit_transform(ship_list)
        ship_cols = ['SHIPMNT_' + str(i + 1) for i in range(ship_encoded.shape[1])]
        encoded_col_name.extend(ship_cols)
        ship_encoded_df = pd.DataFrame(ship_encoded, columns=ship_cols)
        data_engred_3 = pd.concat([data_engred_3, ship_encoded_df], axis=1, sort=False)
        # print('ship_encoded: ', ship_encoded.shape)

        # Quality
        qlity_list = data_engred_3['QULITY'].values.reshape(len(data_engred_3), 1)
        qlity_encoded = onehot_encoder.fit_transform(qlity_list)
        qlity_cols = ['QLITY_' + str(i + 1) for i in range(qlity_encoded.shape[1])]
        encoded_col_name.extend(qlity_cols)
        qlity_encoded_df = pd.DataFrame(qlity_encoded, columns=qlity_cols)
        data_engred_3 = pd.concat([data_engred_3, qlity_encoded_df], axis=1, sort=False)
        # print('qlity_encoded: ', qlity_encoded.shape)

        # Month
        data_engred_3['MONTH'] = pd.to_numeric(data_engred_3['DELNG_DE'].str.slice(4, 6))
        month_list = data_engred_3['MONTH'].values.reshape(len(data_engred_3), 1)
        month_encoded = onehot_encoder.fit_transform(month_list)
        month_cols = ['MONTH_' + str(month) for month in np.unique(data_engred_3['MONTH'].values)]
        encoded_col_name.extend(month_cols)
        month_encoded_df = pd.DataFrame(month_encoded, columns=month_cols)
        data_engred_3 = pd.concat([data_engred_3, month_encoded_df], axis=1, sort=False)
        # print('month_encoded: ', month_encoded.shape)

        return data_engred_3, encoded_col_name

    def feat_engr_final(self, data_engred_5: pd.DataFrame, encoded_col_list: list):
        fixed_col = ['DELNG_DE', 'AUC', 'DELNG_QY', 'ex_PRC', 'PRC',
                     'SUBTOTAL_1', 'SUBTOTAL_2', 'SUBTOTAL_3', 'SUBTOTAL_4', 'SUBTOTAL_5', 'SUBTOTAL_6', 'SUBTOTAL_7',
                     'CPR_SUB_1', 'CPR_SUB_2', 'CPR_SUB_3', 'CPR_SUB_4', 'CPR_SUB_5', 'CPR_SUB_6', 'CPR_SUB_7']
        fixed_col.extend(encoded_col_list)

        return data_engred_5[fixed_col]

    def save_preprocessing_data(self, data: pd.DataFrame):
        data.to_csv(self.SAVE_PATH)

    def load_preprocessing_data(self):
        return pd.read_csv(self.SAVE_PATH)

    ####################################################
    # Distrinbution Preprocessing
    ####################################################

    def get_prdt_remap(self, data: pd.DataFrame):
        temp = data
        # temp = self.get_dist_prdt_filtered(temp)
        temp['QULITY'] = temp['STD_QLITY_NEW_CODE'].map(self.QLITY_CODE_MAP)
        temp['SHIPMNT'] = temp['SHIPMNT_SE_CODE'].map(self.SHIPMENT_CODE_MAP)
        temp['SHIPMNT'] = temp['SHIPMNT'].astype('int32')
        temp = self.prdlst_code_to_number(data=temp)

        return temp

    def prdlst_code_to_number(self, data: pd.DataFrame):
        # 품목
        kind = []
        data['KIND'] = np.nan
        for i in range(len(data)):
            if data['STD_PRDLST_NEW_CODE'][i] == '803':
                if data['DELNG_PRUT'][i] == 10:
                    kind.append(1)
                    # data['kind'] = 1
                elif data['DELNG_PRUT'][i] == 5:
                    kind.append(2)
                    # data['kind'] = 2
                elif data['DELNG_PRUT'][i] == 4:
                    kind.append(3)
                    # data['kind'] = 3
            elif data['STD_PRDLST_NEW_CODE'][i] == '806':
                if data['STD_SPCIES_NEW_CODE'][i] != '080603':
                    if data['DELNG_PRUT'][i] == 5:
                        kind.append(4)
                        # data['kind'] = 4
                    elif data['DELNG_PRUT'][i] == 2:
                        kind.append(5)
                        # data['kind'] = 5
                else:
                    if data['DELNG_PRUT'][i] == 3:
                        kind.append(6)
                        # data['kind'] = 6
                    elif data['DELNG_PRUT'][i] == 2:
                        kind.append(7)
                        # data['kind'] = 7
        data['KIND'] = kind

        return data

    def one_hot_kind(self, data: list):
        kind = []
        for i in range(len(data)):
            if data[i][0] == 1:
                kind.append([1, 0, 0, 0, 0, 0, 0])
            elif data[i][0] == 2:
                kind.append([0, 1, 0, 0, 0, 0, 0])
            elif data[i][0] == 3:
                kind.append([0, 0, 1, 0, 0, 0, 0])
            elif data[i][0] == 4:
                kind.append([0, 0, 0, 1, 0, 0, 0])
            elif data[i][0] == 5:
                kind.append([0, 0, 0, 0, 1, 0, 0])
            elif data[i][0] == 6:
                kind.append([0, 0, 0, 0, 0, 1, 0])
            elif data[i][0] == 7:
                kind.append([0, 0, 0, 0, 0, 0, 1])

        kind_frame = pd.DataFrame(kind, columns=['KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7'])
        pd.concat(data, kind_frame, axis=1)