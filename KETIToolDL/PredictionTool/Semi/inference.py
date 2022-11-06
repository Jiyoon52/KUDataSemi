import sys
sys.path.append("..")
sys.path.append("../..")

from KETIToolDL.PredictionTool.inference import Inference
from KETIToolDL.TrainTool.Semi.SMATE import predict_ssl

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, roc_auc_score
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class SemiModelTestInference(Inference):
    def __init__(self, train_x, train_y, test_x, test_y, model_name, parameter, batch_size, device):
        super().__init__()

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.batch_size = batch_size
        self.device = device
        self.model_name = model_name
        self.parameter = parameter
    
    def inference(self, model, modelFilePath):
        print(f"Start testing model: {self.model_name}")

        # build test dataloader
        test_loader = self.get_labeled_dataloader(self.test_x, self.test_y, self.batch_size, shuffle=False)

        # get predicted classes
        if self.model_name == 'SemiTime_cf': # [backbone, sup_head, relation_head]
            backbone_model = model
            sup_head = torch.nn.Sequential(
                        torch.nn.Linear(self.parameter['feature_size'], self.parameter['num_classes']))
            backbone_model.load_state_dict(torch.load(modelFilePath[0])['backbone'])
            sup_head.load_state_dict(torch.load(modelFilePath[0])['sup_head'])
            pred_data, probs = self.test(backbone_model, sup_head, test_loader)

        elif self.model_name == 'SMATE_cf':
            backbone_encoder_model = model.encoder
            backbone_encoder_model.load_state_dict(torch.load(modelFilePath[0]))
            pred_data, probs, acc = predict_ssl(backbone_encoder_model, self.train_x, self.train_y, self.test_x, self.test_y)
            

        # class의 값이 0부터 시작하지 않으면 0부터 시작하도록 변환
        if self.model_name != 'SMATE_cf' and np.min(self.test_y) != 0:
            print('Set start class as zero')
            self.test_y = self.test_y - np.min(self.test_y)

        # calculate performance metrics
        acc = accuracy_score(self.test_y, pred_data)
        
        # merge true value and predicted value
        pred_df = pd.DataFrame()
        pred_df['actual_value'] = self.test_y
        pred_df['predicted_value'] = pred_data
        pred_df[['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5']] = probs

        test_y = np.array(self.test_y)
        pred_data = np.array(pred_data)

        df = pd.DataFrame(classification_report(test_y, pred_data, output_dict=True)).transpose()
        precision = df.loc['macro avg', 'precision']
        recall = df.loc['macro avg', 'recall']
        f1 = df.loc['macro avg', 'f1-score']
        auroc = roc_auc_score(np.array(pred_df['actual_value']), np.array(pred_df.loc[:, 'prob0':'prob5']), multi_class = 'ovo', average = 'macro')    
        result_metrics = classification_report(test_y, pred_data, output_dict=True)

        return pred_df, result_metrics, acc, auroc
    
    def get_labeled_dataloader(self, x_data, y_data, batch_size, shuffle):
        """
        Get DataLoader
        
        :param x_data: input data
        :type x_data: numpy array

        :param y_data: target data
        :type y_data: numpy array

        :param batch_size: batch size
        :type batch_size: int

        :param shuffle: shuffle for making batch
        :type shuffle: bool

        :return: dataloader
        :rtype: DataLoader
        """
        # class의 값이 0부터 시작하지 않으면 0부터 시작하도록 변환
        if self.model_name != 'SMATE' and np.min(y_data) != 0:
            print('Set start class as zero')
            y_data = y_data - np.min(y_data)
        
        # for labeled data
        dataset = torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)            

        return data_loader

    def test(self, backbone, sup_head, test_loader):
        """
        Predict classes for test dataset based on the trained model

        :param backbone: best trained backbone
        :type model: model

        :param sup_head: best trained sup_head
        :type model: model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: predicted classes
        :rtype: numpy array
        """

        # 각 모델을 gpu에 태우기
        backbone = backbone.to(self.device)
        sup_head = sup_head.to(self.device)
        
        # Inference 모드로 전환
        backbone.eval()
        sup_head.eval()   
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            preds = []
            probs = []
            label_list = []
            corrects = 0
            total = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)

                # input을 model에 넣어 output을 도출
                output = backbone(inputs)
                output = sup_head(output)
                prob = nn.Softmax(dim=1)(output)

                # 예측된 pred값을 append
                pred = output.argmax(-1)
                corrects += torch.sum(pred == labels.data)
                total += labels.size(0)

                preds.extend(pred.detach().cpu().numpy())
                probs.extend(prob.detach().cpu().numpy())
                label_list.extend(labels.detach().cpu().numpy())

        return  np.array(preds), np.array(probs)


