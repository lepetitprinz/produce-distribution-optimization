# -*-coding: utf-8-*-

import numpy as np
import pandas as pd

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

from Dao import Dao
from Preprocessing import Preprocessing
from Model import Model
from Distribution import Distribution

# MARKET_KOR = '가락'
# MARKET_ENG = 'garak'

MARKET_KOR = '대구'
MARKET_ENG = 'daegu'

def main():
    """
    Main Function for Application Launcher
    :return:
    """

    ############################
    # Loading Data
    ############################
    # Data Access Object ( to D/B )
    dao: Dao = Dao.instance()

    # Query Data
    df: pd.DataFrame = dao.get_data()

    # ############################
    # # PreProcessing
    # ############################

    # 전처리 수행 인스턴스 초기화
    preprocessing: Preprocessing = Preprocessing()
    # preprocessing.set_dir('./mafra')

    # 전처리
    df_final = preprocessing.run(df)
    preprocessing.save_preprocessing_data(data=df_final)

    ############################
    # Modeling
    ############################
    # 전처리 완료 데이터 로드
    preprocessing: Preprocessing = Preprocessing()
    df_preprocessing = preprocessing.load_preprocessing_data()

    # Initialize Model Object for Modeling
    model = Model()

    # # Drop Columns
    df_new = model.drop_columns(data=df_preprocessing)

    # X, Y Split
    input_data, output_data = model.get_input_output(data=df_new)

    # Min-Max Scaling
    input_scaled = model.data_scaling(data=input_data)

    # Train, Test Split
    input_set, input_test, output_set, output_test = model.train_test_split(x=input_scaled, y=output_data, kind='test')
    input_train, input_val, output_train, output_val = model.train_test_split(x=np.array(input_set), y=np.array(output_set), kind='validation')

    # Model Preparation
    model_set = model.build_model_from_json(MARKET_KOR, input_shape=input_set.shape[1])            # Hard Coding

    ###################
    # Model Training
    ###################
    opt = optimizers.Adam(lr=0.001)
    model_set.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mse'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    cp = ModelCheckpoint(filepath='./checkpoint/mlp_' + MARKET_ENG + '.h5', monitor='val_loss', save_best_only=True)
    model_set.fit(input_train, output_train, validation_data=(input_val, output_val),
                  epochs=10000, batch_size=16384, callbacks=[es, cp])

    # 사전에 학습한 weight 불러오기
    model.load_weights_from_h5(model_set, '가락')
    model_set.load_weights('./checkpoint/mlp_garak.h5')

    # Evaluation w.t. Test Dataset  - Todo: 모델 Input 변수 갯수 동적 할당
    model_set.evaluate(input_test, output_test)  # Todo: 칼럼 수 차이 보정

    ############################
    # Distribution(묻동량 분배)
    ############################

    distribution = Distribution()

    #
    pred_0 = input_scaled[:]    # Fixme: 조건문 필요 ( 원본 하드코딩 )
    model_0 = model_set
    minmax_0 = callable(object())
    rev_minmax_0 = callable(object())

    # 선택된 시장별 필요 정보 사전
    market_dict: dict = {
        '가락': {
            'MODEL': model_set,
            'INPUT': input_scaled[:]    # Fixme: 조건문 필요 ( 원본 하드코딩 )
        }
    }
    #
    # 예측 실행
    best_DELNG_dict: dict = model.pred(market_dict)

    # 조절 이후 최적 물동량
    optimized_result = distribution.calc_rev_std(best_DELNG_dict)

    # 조절 이전 물동량
    distribution.calc_rev_std(input_scaled[:])     # Fixme: 조건문 필요 ( 원본 하드코딩 )

    print("Debug Point")

if __name__ == '__main__':
    """
        Call Main Function
    """
    main()
