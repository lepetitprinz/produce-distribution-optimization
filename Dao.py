# -*-coding: utf-8-*-

import os
import cx_Oracle as Oci
import pandas as pd
import warnings
from typing import List, Dict

from SingletonInstance import SingletonInstance

warnings.filterwarnings('ignore')
os.environ["NLS_LANG"] = ".AL32UTF8"
os.putenv('NLS_LANG', 'KOREAN_KOREA.KO16MSWIN949')

class Dao(SingletonInstance):

    # D/B 접속 정보
    HOST = '211.62.179.111'
    SID = 'orcl'
    ID = 'MASTER'
    PASSWORD = 'ROOT1234'

    # MARKET_CODE = '1005601'     # 가락
    MARKET_CODE = '1041401'    # 대구

    TRAIN_START_DATE = '20190501'   # Train Data: Start Date
    TRAIN_END_DATE = '20200228'     # Test Data: End Date

    DATE  = '20200622'      # 임시처리

    def __init__(self):
        self.conn: Oci.Connection = Oci.connect(self.ID, self.PASSWORD, self.HOST + '/' + self.SID)
        # self.market_code = self.get_user_select_market()
        # self.market_code = '1005601'  # 가락동농수산물시장 Fixme: 임시처리
        self.market_code = self.MARKET_CODE  # 가락동농수산물시장 Fixme: 임시처리
        self.train_start_date = self.TRAIN_START_DATE
        self.train_end_date = self.TRAIN_END_DATE
        self.date = self.DATE

    def get_user_select_market(self):
        query = """SELECT WHSAL_MRKT_NEW_CODE
                     FROM TBL_MRKT_DIST_INFO"""

        return pd.read_sql(query, con=self.conn)

    # User가 선택한 시장의 데이터를 받아오는 query
    def get_data(self) -> pd.DataFrame:
        query = """ SELECT DELNG_DE, AUC_SE_NM, INSTT_NEW_NM, STD_PRDLST_NEW_NM, STD_SPCIES_NEW_NM, DELNG_PRUT, 
                          STD_QLITY_NEW_NM, SHIPMNT_SE_NM,STD_MTC_NEW_NM, DELNG_QY, SBID_PRIC
                     FROM TBL_DATA_MART 
                    WHERE 1=1
                      AND WHSAL_MRKT_NEW_CODE = '{}'   -- Market
                      AND STD_PRDLST_NEW_CODE = '0908' -- Tomato
                      AND DELNG_PRUT IN (4,5,10) -- Weight
                      AND (DELNG_DE >= '{}' AND DELNG_DE <= '{}') -- Date
                    UNION ALL
                    SELECT DELNG_DE, AUC_SE_NM, INSTT_NEW_NM, STD_PRDLST_NEW_NM, STD_SPCIES_NEW_NM, DELNG_PRUT, 
                          STD_QLITY_NEW_NM, SHIPMNT_SE_NM,STD_MTC_NEW_NM, DELNG_QY, SBID_PRIC
                     FROM TBL_DATA_MART 
                    WHERE 1=1
                      AND WHSAL_MRKT_NEW_CODE = '{}'  -- Market
                      AND STD_PRDLST_NEW_CODE = '911' -- Tomato
                      AND STD_SPCIES_NEW_CODE <> '091103'
                      AND DELNG_PRUT IN (2,5) -- Weight 
                      AND (DELNG_DE >= '{}' AND DELNG_DE <= '{}') -- Date
                    UNION ALL
                    SELECT DELNG_DE, AUC_SE_NM, INSTT_NEW_NM, STD_PRDLST_NEW_NM, STD_SPCIES_NEW_NM, DELNG_PRUT, 
                          STD_QLITY_NEW_NM, SHIPMNT_SE_NM,STD_MTC_NEW_NM, DELNG_QY, SBID_PRIC
                     FROM TBL_DATA_MART 
                    WHERE 1=1
                      AND WHSAL_MRKT_NEW_CODE = '{}'  -- Market
                      AND STD_PRDLST_NEW_CODE = '911' -- Tomato
                      AND STD_SPCIES_NEW_CODE = '091103'
                      AND DELNG_PRUT IN (2,3) -- Weight
                      AND (DELNG_DE >= '{}' AND DELNG_DE <= '{}') -- Date""".format(self.market_code, self.train_start_date, self.train_end_date,
                                                                                self.market_code, self.train_start_date, self.train_end_date,
                                                                                self.market_code, self.train_start_date, self.train_end_date)

        return pd.read_sql(query, con=self.conn)

    def get_prdt_data(self) -> pd.DataFrame:
        query = """SELECT DATE_INFO
                        , STD_MTC_NEW_CODE      --산지코드
                        , STD_PRDLST_NEW_CODE   -- 품목
                        , STD_SPCIES_NEW_CODE   -- 품종
                        , DELNG_PRUT            --단량
                        , STD_QLITY_NEW_CODE    -- 등급
                        , SHIPMNT_SE_CODE       -- 출하구분
                        , SUM(ORDER_QUANTITY) AS ORDER_QUANTITY -- 입고량
                     FROM TBL_PRDT_INFO
                    WHERE 1=1
                     AND STD_PRDLST_NEW_CODE = '803'
                     AND DELNG_PRUT IN (4,5,10)
                     AND SHIPMNT_SE_CODE IS NOT NULL
                    GROUP BY DATE_INFO
                        , STD_MTC_NEW_CODE
                        , STD_PRDLST_NEW_CODE
                        , STD_SPCIES_NEW_CODE
                        , DELNG_PRUT
                        , STD_QLITY_NEW_CODE
                        , SHIPMNT_SE_CODE
                UNION ALL
                SELECT DATE_INFO
                        , STD_MTC_NEW_CODE      --산지코드
                        , STD_PRDLST_NEW_CODE   -- 품목
                        , STD_SPCIES_NEW_CODE   -- 품종
                        , DELNG_PRUT            --단량
                        , STD_QLITY_NEW_CODE    -- 등급
                        , SHIPMNT_SE_CODE       -- 출하구분
                        , SUM(ORDER_QUANTITY) AS ORDER_QUANTITY -- 입고량
                     FROM TBL_PRDT_INFO
                    WHERE 1=1
                     AND STD_PRDLST_NEW_CODE = '806'
                     AND STD_SPCIES_NEW_CODE <> '080603'
                     AND DELNG_PRUT IN (2, 5)
                     AND SHIPMNT_SE_CODE IS NOT NULL
                    GROUP BY DATE_INFO
                        , STD_MTC_NEW_CODE
                        , STD_PRDLST_NEW_CODE
                        , STD_SPCIES_NEW_CODE
                        , DELNG_PRUT
                        , STD_QLITY_NEW_CODE
                        , SHIPMNT_SE_CODE
                UNION ALL
                SELECT DATE_INFO
                        , STD_MTC_NEW_CODE      --산지코드
                        , STD_PRDLST_NEW_CODE   -- 품목
                        , STD_SPCIES_NEW_CODE   -- 품종
                        , DELNG_PRUT            --단량
                        , STD_QLITY_NEW_CODE    -- 등급
                        , SHIPMNT_SE_CODE       -- 출하구분
                        , SUM(ORDER_QUANTITY) AS ORDER_QUANTITY -- 입고량
                     FROM TBL_PRDT_INFO
                    WHERE 1=1
                     AND STD_PRDLST_NEW_CODE = '806'
                     AND STD_SPCIES_NEW_CODE = '080603'
                     AND DELNG_PRUT IN (2, 3)
                     AND SHIPMNT_SE_CODE IS NOT NULL
                    GROUP BY DATE_INFO
                        , STD_MTC_NEW_CODE
                        , STD_PRDLST_NEW_CODE
                        , STD_SPCIES_NEW_CODE
                        , DELNG_PRUT
                        , STD_QLITY_NEW_CODE
                        , SHIPMNT_SE_CODE"""

        return pd.read_sql(query, con=self.conn)

    # 시장 이름별 시장 code mapping dictionary
    def get_market_map(self):
        query = """SELECT UNIQUE WHSAL_MRKT_NEW_NM
                        , WHSAL_MRKT_NEW_CODE
                     FROM TBL_NH_DATA_MART
                    ORDER BY WHSAL_MRKT_NEW_CODE"""

        market = pd.read_sql(query, con=self.conn)
        market_map = {row[0]: row[1] for i, row in market.iterrows()}

        return market_map

    # 법인 List를 받아오는 query
    def get_corp_to_num_map(self) -> dict:
        query = """SELECT UNIQUE INSTT_NEW_NM
                     FROM TBL_NH_DATA_MART 
                    WHERE WHSAL_MRKT_NEW_CODE = '{}' 
                    ORDER BY INSTT_NEW_NM
                    """.format(self.market_code)
        corp_list = pd.read_sql(query, con=self.conn)
        corp_map = {corp: i + 1 for i, corp in enumerate(corp_list['INSTT_NEW_NM'].values)}

        return corp_map

    def insert(self, sql: str, data_list: list) -> None:

        try:
            cursor = self.conn.cursor()
            cursor.executemany(sql, data_list)
            self.conn.commit()
        except Oci.DatabaseError as e:
            error, _ = e.args
            error_code = error.code
            raise e

    def insert_whsd_info(self, data_list: list) -> None:
        sql = """
        INSERT INTO TBL_WHSD_INFO(
            WHSAL_MRKT_NEW_CODE, DATE_INFO, NUM, STD_PRDLST_NEW_CODE, STD_MTC_NEW_CODE, PRO_NAME, SEQ, 
            EXP_DIST, EXP_PROFIT_PRICE, INSTT_NEW_NM, BIGO, REG_DTTM, REG_ID, REG_IP, UPD_DTTM, UPD_ID, UPD_IP 
        ) values (
            {}
        )
        """.format(', '.join(list(map(lambda x: f':{x}', range(1, 1 + 17)))))
        try:
            self.insert(sql, data_list)
        except Oci.DatabaseError as e:
            raise e

    def get_custom_date(self) -> pd.DataFrame:
        query = """쿼리 입력"""
        return pd.read_sql(query, con=self.conn)

    def update_data(self, data_list: list) -> None:
        sql = """INSERT INTO TABLE(
            col_1, col_2, .... col_n
            )values(:1, :2, ...., :n)
            """
        try:
            self.insert(sql, data_list)
        except Oci.DatabaseError as e:
            raise e
