import sys
import numpy as np
import pandas as pd
import time
import datetime

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../..")

import pathSetting

from KETIToolDL.CLUSTTool.common import p1_integratedDataSaving as p1
from KETIPreDataTransformation.dataFormatTransformation import NPArrayToDF as nptodf #import trans2NPtoDF
from KETIPreDataTransformation.trans_for_purpose.transformForDataSplit import getSplitAndTransformDataByFrequency

class getIntegrationData():
    def __init__(self, originalDataset, startTime, splitNum, transformFreqList, **Parameter):
        self.originalDataset = originalDataset
        self.startTime = startTime
        self.splitNum = splitNum
        self.transformFreqList = transformFreqList
        self.Parameter = Parameter
        self.integration_method = self.Parameter["integration_method"]
        self.splitInterval = self.Parameter["splitInterval"]
        self.freqTransformMode = self.Parameter["freqTransformMode"]
        self.transformParam = self.Parameter["transformParam"]
        self.processParam = self.Parameter["processParam"]
        self.integration_duration_criteria = self.Parameter["integration_duration_criteria"]
        self.integration_freq_sec = self.Parameter["integration_freq_sec"]
        self.original_freq = self.Parameter["original_freq"]
        
    def getIntegrationDataByMethod(self, mode = None, data_y = None):
        self.since = time.time()
        if self.integration_method == "ML":
            self.intDataSetX, self.DataSety, self.window = self.getIntegratedDatabyML(data_y)
            self.getTimeElapsed()
            return self.intDataSetX, self.DataSety, self.window
        else:
            if (mode == "trainX")|(mode == "testX"):
                self.intDataSet, self.window = self.getIntegratedDataXbyMeta()
            elif (mode == "trainy")|(mode == "testy"):
                self.intDataSet, self.window = self.getIntegratedDatay() # integration_method = None
            self.getTimeElapsed()
            return self.intDataSet, self.window

    def getIntegratedDatabyML(self, dataset_train_y):
        int_dataset_X = pd.DataFrame()
        train_y = pd.DataFrame()
        for n in range(len(np.unique(dataset_train_y))):
            # 해당 class 데이터만 추출
            y_class_idx = np.where(dataset_train_y == n)
            print(">>>> class " , n, " , length : ", len(y_class_idx[0]))
            datay_class = dataset_train_y[y_class_idx]
            dataX_class = self.originalDataset[y_class_idx]
            
            # 데이터 형태 변환 및 통합 데이터 생성
            print("====== start trans3NP to DF ======")
            df_x = nptodf.trans3NPtoDFbyInputFreq(dataX_class, self.startTime, self.original_freq)
            df_y = pd.DataFrame({"value":datay_class})
            print("df_x length : ",len(df_x))
            
            print("====== start split & transform ======")
            dataSet = getSplitAndTransformDataByFrequency(df_x, self.splitNum, self.splitInterval, self.transformFreqList, self.freqTransformMode)
            
            print("====== start integration ======")
            int_data = p1.getIntDataFromDataset(self.integration_freq_sec, self.processParam, dataSet, self.integration_method, self.transformParam, self.integration_duration_criteria)
            # get integrated data length
            window = len(int_data)
            # get integrated data set
            int_dataset_X = pd.concat([int_dataset_X, int_data])
            train_y = pd.concat([train_y, df_y])
            
        X_timeIndex = pd.date_range(start=self.startTime, freq = str(self.integration_freq_sec)+"S", periods=len(int_dataset_X))
        y_timeIndex = pd.date_range(start=self.startTime, freq = self.original_freq, periods=len(train_y))
        
        int_dataset_X.set_index(X_timeIndex, inplace = True)
        train_y.set_index(y_timeIndex, inplace = True)
        
        return int_dataset_X, train_y, window
            
    def getIntegratedDataXbyMeta(self):
        count = 0
        int_dataset = pd.DataFrame()
        print("dataset shape : ",self.originalDataset.shape)

        for array_X in self.originalDataset:
            print("array num : ", count)
            print("......................")
            
            df_x = nptodf.trans2NPtoDF(array_X, self.startTime, self.original_freq)
            dataSet = getSplitAndTransformDataByFrequency(df_x, self.splitNum, self.splitInterval, self.transformFreqList, self.freqTransformMode)
            if self.splitNum ==1:
                int_data = dataSet.copy()
            else:    
                # get integrated data
                int_data = p1.getIntDataFromDataset(self.integration_freq_sec, self.processParam, dataSet, self.integration_method, self.transformParam, self.integration_duration_criteria)
                    
            # get integrated data length
            window = len(int_data)
            # get integrated data set
            int_dataset = pd.concat([int_dataset, int_data])
            
            count+=1

        # integrated data set index
        timeIndex = pd.date_range(start=self.startTime, freq = str(self.integration_freq_sec)+"S", periods=len(int_dataset))
        int_dataset.set_index(timeIndex, inplace = True)
        
        return int_dataset, window


    def getIntegratedDatay(self):
        #original_freq = self.Parameter["original_freq"]
        data_y = pd.DataFrame({"value":self.originalDataset})
        y_timeIndex = pd.date_range(start=self.startTime, freq = self.original_freq, periods=len(data_y))
        data_y["datetime"] = y_timeIndex
        data_y.set_index("datetime", inplace=True)
        window = 1
        
        return data_y, window
    
    def getTimeElapsed(self):
        # 전체 학습 시간 계산
        time_elapsed = time.time() - self.since
        print('\n>>>>>>Training complete in {:.0f}m {:.5f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.timeElapsed = str(datetime.timedelta(seconds=time_elapsed))