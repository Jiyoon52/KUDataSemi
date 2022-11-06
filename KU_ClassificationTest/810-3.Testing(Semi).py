import sys, os
import pandas as pd
import numpy as np
import setting
import pathSetting
import torch

sys.path.append("../../..")

from KETIToolDL.CLUSTTool.common import p1_integratedDataSaving as p1
from KETIToolDL.CLUSTTool.common import p2_dataSelection as p2
from KETIToolDL.CLUSTTool.common import p3_training as p3
from KETIToolDL.CLUSTTool.common import p4_testing as p4
from KETIToolDL.CLUSTTool.Semi import p4_testing as p4S


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

# Test Data & Model
# 1-1. 저장 Data 확인
DataMeta = p1.readJsonData(pathSetting.DataMetaPath)
dataList =  list(DataMeta.keys())
dataList

# 1-2. Select Test Data
## dataX
dataName_X = dataList[2]
## datay
dataName_y = dataList[3]
dataFolderName = "data_integrated_result"

# 1-3. 저장 Model 확인
ModelMeta =p1.readJsonData(pathSetting.trainModelMetaFilePath)
modelList = list(ModelMeta.keys())
modelList

# 1-4. select Model 
modelName = modelList[3]
windows = DataMeta[dataName_X]["integrationInfo"]["dataInfo"]["windows"]

# 2. Testing
# 2-1. Read Parameter and model information and Testing
# load data 부분이 DataMeta를 통해 불러오지 않고, 간단히 raw 데이터를 불러서 getTestResult에 넣어주는 형식 으로 우선 변경

folderAddress = './data/'
model_list = ["SemiTime_cf", "SMATE_cf"]
model_method = model_list[0]
split_ratio = 0.2
scaler_x_path = './scaler/HAR_temp_minmax_scaler_x_semi.pkl'
train_x, train_y,test_x, test_y = setting.load_data(folderAddress, model_method)
test_x, x_scaler = setting.get_test_data(test_x, scaler_x_path)

# SMATE 알고리즘에서 train_x, train_y 필요
df_result, result_metrics, acc, auroc = p4S.getTestResult(train_x, train_y, test_x, test_y, modelName, DataMeta, ModelMeta, dataFolderName, device, windows)
# df_result, result_metrics, acc= p4C.getTestResult(dataName_X, dataName_y, modelName, DataMeta, ModelMeta, dataFolderName, device, windows)

print('acc :' , acc)

# 3. Save Result
import json
save_path = './modelResult'
os.makedirs(save_path, exist_ok=True)

with open('./modelResult/{}_result.json'.format(modelName),'w') as f:
  json.dump(result_metrics, f, ensure_ascii=False, indent=4)


