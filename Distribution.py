import math
import numpy as np
import pandas as pd

from itertools import product

from Preprocessing import Preprocessing

class Distribution(object):
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def calc_rev_std(self, best_DELNG: np.array, data: pd.DataFrame):
        """
        ga_rev_minmax function
        :param best_DELNG:
        :param data:
        :return:
        """
        best = pd.DataFrame(best_DELNG)

        temp_max = np.array(data.max())
        temp_min = np.array(data.min())
        result = best * (temp_max - temp_min) + temp_min

        return result

    def pred(self, market_dict: dict, max_iteration: int = 100):
        """
        가격 예측
        :param market_dict
        {
            market_name: {
                MODEL: Model Instance
                INPUT: Input DataFrame
                # rev_std: rev function
            }
        }
        :param max_iteration: 최대 Iteration 수
        :return:
        """

        # 콘솔 출력을 위한 MaxIteration 횟수의 자릿수
        # ex: 100 - 3자리
        # ex: 2000 - 4자리
        nof_digit: int = int(math.floor(math.log10(max_iteration))) + 1

        # 반복적으로 참조되는 Constant 정보
        sel_cols: list = [
        'AUC', 'KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7',
        'SHIPMNT_1', 'SHIPMNT_2', 'SHIPMNT_3', 'QLITY_1', 'QLITY_2', 'QLITY_3', 'QLITY_4',
        'QLITY_5', 'QLITY_6', 'QLITY_7', 'DELNG_QY', 'PRC', 'MART'
        ]
        tsel_cols: list = [
        'AUC', 'KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7',
        'SHIPMNT_1', 'SHIPMNT_2', 'SHIPMNT_3', 'QLITY_1', 'QLITY_2', 'QLITY_3', 'QLITY_4',
        'QLITY_5', 'QLITY_6', 'QLITY_7', 'DELNG_QY', 'PRC', 'MART'
        ]
        new_tsel_cols: list = tsel_cols + ['new_PRC', 'include']
        drop_cols: list = ['DELNG_QY_new', 'PRC', 'MART', 'new_PRC', 'include']
        value_range: product = product(
        Preprocessing.AUC_MAP.values(),         # AUC
        Preprocessing.SHIPMENT_MAP.values(),    # SHIPMNT
        Preprocessing.KIND_VALUES,              # KIND
        Preprocessing.QLITY_MAP.values()        # QLITY
        )

        # 상태 변수 초기화
        pred_price_dict = {}
        df_pred_dict = {}
        sel_dict = {}
        test_dict = {}
        best_DELNG_dict = {}
        new_pred_dict = {}

        # Iteration
        profit_list = []
        for iter_count in range(max_iteration):
            for market_name, market_info in market_dict.items():
                market_model = market_info['MODEL']
                market_input = market_info['INPUT']

                pred_price_val = market_model.predict(market_input)
                df_pred_val = self.calc_rev_std(market_input)
                df_pred_val['PRC'] = pred_price_val
                df_pred_val['MART'] = market_name
                sel_dict_val = df_pred_val[sel_cols]

                pred_price_dict.update({market_name: pred_price_val})
                df_pred_dict.update({market_name: df_pred_val})
                sel_dict.update({market_name: sel_dict_val})

        concat_list = list(sel_dict.values())
        total = pd.concat(concat_list)
        t_sel = pd.DataFrame(columns=tsel_cols)

        profit = 0
        for auc, shipment, kind, quality in value_range:
            row_idxs: list = \
                total['AUC'] == auc and \
                total[f'SHIPMNT_{shipment}'] == 1 and \
                total[f'KIND_{kind}'] == 1 and \
                total[f'QLITY_{quality}'] == 1
            total_sub: pd.DataFrame = total[row_idxs]

            del_sum = total_sub['DELNG_QY'].sum()
            if del_sum == 0:
                continue

            r = total_sub['DELNG_QY'] / del_sum
            profit += np.matmul(np.array(total_sub['DELNG_QY']), total_sub['PRC'])

            new_r = r + self.ALPHA * (total_sub['PRC'] / total_sub['PRC'].sum())
            new_r = np.array(new_r)
            nega = 0
            for i in range(len(new_r)):
                if new_r[i] <= 0:
                    nega += new_r[i]
                    new_r[i] = 0

            new_r[np.argmax(new_r)] += nega
            total_sub['DELNG_QY'] = np.round(del_sum * new_r)
            t_sel.append(total_sub)

        t_sel.rename(columns={"DELNG_QY": "DELNG_QY_new"}, inplace=True)

        for market_name, market_info in market_dict.items():
            test_dict_val = pd.merge(df_pred_dict[market_name], t_sel[t_sel['MART'] == market_name], how='left')
            test_dict_val['DELNG_QY'] = test_dict_val['DELNG_QY_new']
            test_dict_val.drop(['DELNG_QY_new', 'PRC', 'MART'], axis=1)
            test_dict.update({market_name: test_dict_val})

            print(f"{'%0{}d'.format(nof_digit) % (iter_count + 1)} "
                  f"\tTotal Profit = {profit}")

            if len(profit_list) > 0 and profit_list[-1] > profit:
                break

        profit_list.append(profit)
        for market_name, market_info in market_dict.items():
            best_DELNG_dict.update({market_name: market_info['INPUT']})

        for market_name, market_info in market_dict.items():
            value = test_dict[market_name]
            market_input = market_info['INPUT']

            value.reset_index(drop=True, inplace=True)
            value = self._subtotal(value)
            value = self._cpr(value)
            market_input_new = self.calc_std(value)

            test_dict.update({market_name: value})
            market_dict[market_name].update({'INPUT': market_input_new})
            new_pred_dict.update({market_name: market_input_new})

        for market_name, market_info in market_dict.items():
            market_model = market_info['MODEL']
            new_pred = new_pred_dict['MARKET_NAME']

            pred_price_dict.update({market_name: market_model.predict(new_pred)})
            new_df_pred_val = self.calc_rev_std(new_pred)
            new_df_pred_val['PRC'] = pred_price_dict[market_name]
            new_df_pred_val['MART'] = market_name
            df_pred_dict[market_name]['new_PRC'] = new_df_pred_val['PRC']
            sel_dict[market_name] = df_pred_dict[market_name][sel_cols]

        concat_list = list(sel_dict.values())
        total = pd.concat(concat_list)
        total['include'] = 1
        t_sel = pd.DataFrame(columns=new_tsel_cols)

        for auc, shipment, kind, quality in value_range:
            row_idxs: list = \
                total['AUC'] == auc and \
                total[f'SHIPMNT_{shipment}'] == 1 and \
                total[f'KIND_{kind}'] == 1 and \
                total[f'QLITY_{quality}'] == 1
            total_sub: pd.DataFrame = total[row_idxs]
            total_sub.reset_index(drop=True)

            if len(total_sub) == 1:
                t_sel.append(total_sub)
            elif t_sel['DELNG_QY'].sum() != 0:
                prc_mean = total_sub['PRC'].mean()
                idxs: list = total_sub[total_sub['PRC'] >= prc_mean].index
                for idx in idxs:
                    if total_sub['PRC'][idx] < total_sub['new_PRC'][idx]:
                        total_sub['include'][idx] = 0

                del_sum = total_sub[total_sub['include'] == 1]['DELNG_QY'].sum()

                r = total_sub[total_sub['include'] == 1]['DELNG_QY'] / del_sum
                new_r = r + \
                        self.ALPHA * (
                                total_sub[total_sub['include'] == 1]['PRC'] / total_sub[total_sub['include'] == 1]['PRC'].mean() - 1)
                new_r = np.array(new_r)
                nega = 0
                for i in range(len(new_r)):
                    if new_r[i] <= 0:
                        nega += new_r[i]
                        new_r[i] = 0
                new_r[np.argmax(new_r)] += nega
                asdf = total_sub[total_sub['include'] == 1]     # Fixme: 변수명 정정 요청 상태
                asdf['DELNG_QY'] = np.round(del_sum * new_r)
                total_sub = asdf.append(total_sub[total_sub['include'] == 0])
                t_sel = t_sel.append(total_sub)
        t_sel.rename(columns={"DELNG_QY": "DELNG_QY_new"}, inplace=True)

        for market_name, market_info in market_dict.items():
            value = pd.merge(df_pred_dict[market_name], t_sel[t_sel['MART'] == market_name], how='left')
            value['DELNG_QY'] = value['DELNG_QY_new']
            idx = value[value['DELNG_QY'] == 0].index
            value.drop(idx, inplace=True)
            value.drop(drop_cols, axis=1, inplace=True)
            test_dict.update({market_name: value})

        for market_name, market_info in market_dict.items():
            value = test_dict[market_name]
            value = self._subtotal(value)
            value = self._cpr(value)
            test_dict.update({market_name: value})
            market_info.update({'INPUT': self.calc_std(value)})

        return best_DELNG_dict

    def _subtotal(self, x):
        x_sub = x[['KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7', 'DELNG_QY']]
        x_sub_groupby = x_sub.groupby(['KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7'],
                                      as_index=False).sum()

        for i in range(1, 8):
            try:
                x['SUBTOTAL_{}'.format(i)] = \
                    x_sub_groupby[x_sub_groupby['KIND_{}'.format(i)] == 1]['DELNG_QY'].values[0]
            except:
                continue

        return x

    def _cpr(self, x):
        x_cpr_sub = x[
            ['KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7', 'CPR_1', 'CPR_2', 'CPR_3', 'CPR_4',
             'CPR_5', 'DELNG_QY']]
        x_cpr_groupby = x_cpr_sub.groupby(
            ['KIND_1', 'KIND_2', 'KIND_3', 'KIND_4', 'KIND_5', 'KIND_6', 'KIND_7', 'CPR_1', 'CPR_2', 'CPR_3', 'CPR_4',
             'CPR_5'], as_index=False).sum()

        for j in range(1, 8):
            for i in range(len(x)):
                if x['CPR_1'][i] == 1:
                    a = x_cpr_groupby[(x_cpr_groupby['CPR_1'] == 1) & (x_cpr_groupby['KIND_{}'.format(j)] == 1)][
                        'DELNG_QY']
                    try:
                        x['CPR_SUB_{}'.format(j)][i] = np.array(a)[0]
                    except:
                        break
                elif x['CPR_2'][i] == 1:
                    a = x_cpr_groupby[(x_cpr_groupby['CPR_2'] == 1) & (x_cpr_groupby['KIND_{}'.format(j)] == 1)][
                        'DELNG_QY']
                    try:
                        x['CPR_SUB_{}'.format(j)][i] = np.array(a)[0]
                    except:
                        break
                elif x['CPR_3'][i] == 1:
                    a = x_cpr_groupby[(x_cpr_groupby['CPR_3'] == 1) & (x_cpr_groupby['KIND_{}'.format(j)] == 1)][
                        'DELNG_QY']
                    try:
                        x['CPR_SUB_{}'.format(j)][i] = np.array(a)[0]
                    except:
                        break
                elif x['CPR_4'][i] == 1:
                    a = x_cpr_groupby[(x_cpr_groupby['CPR_4'] == 1) & (x_cpr_groupby['KIND_{}'.format(j)] == 1)][
                        'DELNG_QY']
                    try:
                        x['CPR_SUB_{}'.format(j)][i] = np.array(a)[0]
                    except:
                        break
                else:
                    a = x_cpr_groupby[(x_cpr_groupby['CPR_5'] == 1) & (x_cpr_groupby['KIND_{}'.format(j)] == 1)][
                        'DELNG_QY']
                    try:
                        x['CPR_SUB_{}'.format(j)][i] = np.array(a)[0]
                    except:
                        break

        return x