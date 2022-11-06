import sys, os
import pandas as pd
import numpy as np
import setting
import pathSetting
sys.path.append("../../..")

from KETIToolDL.CLUSTTool.common import p1_integratedDataSaving as p1
from KETIToolDL.CLUSTTool.common import p2_dataSelection as p2
from KETIToolDL.CLUSTTool.common import p3_training as p3

import torch
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

# 1. Train Data
# 1-1. 저장 Data 확인
DataMeta = p1.readJsonData(pathSetting.DataMetaPath)
dataList =  list(DataMeta.keys())

# 1-2. Select Train Data
## dataX
dataName_X = dataList[2]
dataSaveMode_X = DataMeta[dataName_X]["integrationInfo"]["DataSaveMode"]
windows = DataMeta[dataName_X]["integrationInfo"]["dataInfo"]["windows"]

## datay
dataName_y = dataList[3]
dataSaveMode_y = DataMeta[dataName_y]["integrationInfo"]["DataSaveMode"]

# 1-3. Read Train Data
## CSV 로 Local 에 데이터 저장되어 있을 경우
dataFolderName = "data_integrated_result"
current = os.getcwd()
dataFolderPath = os.path.join(current, dataFolderName)

dataX = p2.getSavedIntegratedData(dataSaveMode_X, dataName_X, dataFolderPath)
datay = p2.getSavedIntegratedData(dataSaveMode_y, dataName_y, dataFolderPath)
integration_freq_sec = DataMeta[dataName_X]["integrationInfo"]["integration_freq_sec"]

print(dataX.shape)
print(datay.shape)

# UCI HAR Data Read
# 원래 아래부분도 from KETIToolDL.CLUSTTool.common import p3_training as p3 등에 녹아있어야 하나, p1이 아직 완전한 형태가 아니어서 일단은 분리해서 사용
folderAddress = './data/'
model_list = ["SemiTime_cf", "SMATE_cf"]
model_name = model_list[1]
split_ratio = 0.2
scaler_x_path = './scaler/HAR_temp_minmax_scaler_x_semi.pkl'
train_x, train_y,test_x, test_y = setting.load_data(folderAddress, model_name)

if model_name == 'SemiTime_cf':
    # split labeled data/unlabeled data in train_x
    train_labeled_x, train_labeled_y, train_unlabeled_x = setting.labeled_unlabeled_split(train_x, train_y)
    train_labeled_x, valid_x, train_labeled_y, valid_y = train_test_split(train_labeled_x, train_labeled_y, test_size=split_ratio, shuffle=True)
    # normalization
    train_labeled_x, train_unlabeled_x, valid_x = setting.get_train_val_data([train_labeled_x, train_unlabeled_x, valid_x], scaler_x_path)
    train_labeled_x, train_labeled_y, train_unlabeled_x, valid_x, valid_y
    data_dict = {'train_labeled_x': train_labeled_x,
                 'train_labeled_y' : train_labeled_y,
                 'train_unlabeled_x': train_unlabeled_x,
                 'val_x': valid_x,
                 'val_y': valid_y}
elif model_name == 'SMATE_cf':
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=split_ratio, shuffle=True)
    train_x, valid_x = setting.get_train_val_data([train_x, valid_x], scaler_x_path)
    data_dict = {'train_x': train_x,
                 'train_y' : train_y,
                 'val_x': valid_x,
                 'val_y': valid_y}
else:
    print('Choose the model correctly')


# 3. Set Training Parameter
# 3-1.
model_list = ["SemiTime_cf","SMATE_cf"]
model_method = model_list[1]

n_epochs = 50 # 학습 epoch 횟수, int(default: 1000, 범위: 1 이상)
batch_size = 512  # batch 크기, int(default: 16, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
num_classes = 6 # class 개수

trainParameter = setting.modelConfig[model_method]
trainParameter['device']  = device
trainParameter['num_classes'] = num_classes

modelTags =["action", "sensor", "classification", "pattern", dataName_X, model_method]
trainDataType = "timeseries"
modelPurpose = "classification"

trainDataInfo = DataMeta[dataName_X]['integrationInfo']

# 3-2. 모델을 저장할 파일 패스를 생성
from KETIPreDataTransformation.general_transformation.dataScaler import encodeHashStyle
trainParameter_encode =  encodeHashStyle(str(trainParameter))

ModelName = dataName_X+"_"+model_method

trainDataPathList = [ModelName, dataName_X, trainParameter_encode]
modelFilePath = p3.getModelFilePath(trainDataPathList, model_method)

# 4. Training
from KETIToolDL.TrainTool.Semi.trainer_copy import ClassificationML as CML

cml = CML(model_method, trainParameter)
cml.processInputData(data_dict, batch_size, windows) #labeled, unlabeled 는 따로 처리 
model = cml.getModel()
best_model, timeElapsed = cml.trainModel(model, modelFilePath, n_epochs)