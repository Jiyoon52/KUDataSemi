import os
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../..")

DataMetaPath = "./integratedData.json"
csvDataFileRootDir ='./data/'
scalerRootDir ='./scaler/'
trainModelMetaFilePath ="./model.json"

IntDataFolderName = "data_integrated_result"

current = os.getcwd()
dataFolderPath = os.path.join(current, IntDataFolderName)