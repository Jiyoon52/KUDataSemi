import os
import sys
import torch

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../..")
from KETIToolDL.CLUSTTool.common import p1_integratedDataSaving as p1

def setParameter(integration_method, window_size, sliding_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device}" " is available.")

    ## split&transform parameter
    splitInterval = {0:3, 1:3, 2:3} # 분할한 각 데이터의 컬럼 개수 : 분할하고 싶은 데이터 개수별로 각 데터 별 분할 간격(컬럼 개수)을 입력
    freqTransformMode = 'sampling' # 주기 변환 방법 : sampling / drop 중 에서 선택

    ## integration parameter
    integration_freq_sec = 4 # 통합 기준 주기
    #integration_method = "ML" # 통합 하는 방법 : meta/ML 중 에서 선택 (ML은 RNN AE 를 활용해 Alignment를 진행)
    integration_duration_criteria = 'total' # 통합 기간 : total / common 중 선택

    ## split&transform&integration 필요한 parameter
    original_freq = '1S'

    transformParam = {}
    DataSaveMode = 'CSV'

    ## RNN AE Param
    if integration_method == "ML":
        transformParam = {
            "model": 'RNN_AE',
            "model_parameter": {
                # window_size : 모델의 input sequence 길이(int값)지만, FC를 활용할때는 inpuData Length-1 값을 넣어야 한다. 
                "window_size": window_size, # 모델의 input sequence 길이, int(default: 10, 범위: 0 이상 & 원래 데이터의 sequence 길이 이하)
                "emb_dim": 32, # 변환할 데이터의 차원, int(범위: 16~256)
                "num_epochs": 100, # 학습 epoch 횟수, int(범위: 1 이상, 수렴 여부 확인 후 적합하게 설정)
                "batch_size": 512, # batch 크기, int(범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
                "learning_rate": 0.0001, # learning rate, float(default: 0.0001, 범위: 0.1 이하)
                "device": device, # 학습 환경, ["cuda", "cpu"] 중 선택
                "sliding_size" : sliding_size # get_loders에서 range의 step 사이즈, int(default : 1, 범위 : 1~window_size)
                }
            }

    ## preprocess parameter
    cleanParam = "NoClean"
    processParam = p1.getProcessParam(cleanParam)


    ## 최종 활용 Parameter
    parameter = {
        "splitInterval" : splitInterval, 
        "freqTransformMode" : freqTransformMode, 
        "integration_freq_sec" : integration_freq_sec,
        "integration_method" : integration_method,
        "integration_duration_criteria" : integration_duration_criteria, 
        "original_freq" : original_freq,
        "transformParam" : transformParam,
        "DataSaveMode" : DataSaveMode,
        "cleanParam" : cleanParam,
        "processParam" : processParam
    }

    return parameter