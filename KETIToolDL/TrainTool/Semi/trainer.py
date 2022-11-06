import os
import sys
from KETIToolDL.TrainTool.Semi.semitime import SimConv4
sys.path.append("..")
sys.path.append("../..")

from KETIToolDL.TrainTool.trainer import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import copy
import datetime
from KETIPreDataTransformation.dataFormatTransformation.DFToNPArray import transDFtoNP
from KETIToolDL.TrainTool.Classification.rnn import RNN_model
from KETIToolDL.TrainTool.Semi.SMATE import SMATE
from KETIToolDL.TrainTool.Semi.semitime import SimConv4


# Class Classification 에서 원래는 def train. test 등을 설정하고 모델 구조만 다른 .py에서 불러오는 형식이었음
# Semi-Supervised Learning 에서는 loader, 모델 구조, 학습 방식이 다르기 때문에 모두 여기서 수정하였음 (22/10/24)

class ClassificationML(Trainer):
    def __init__(self, model_name, parameter):
        # seed 고정
        random_seed = 42

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        super().__init__()
        
        self.device = parameter['device']
        # self.model_name = model_name
        self.parameter = parameter
        self.parameter['model_name'] = model_name
 

    def processInputData(self, data_dict, batch_size, windowNum = 0):
        """
        :param trainX: train x data
        :type trainX: Array
        
        :param trainy: train y data
        :type trainy: Array
        
        :param model_name: model training method
        :type model_name: string
        
        :param modelParameter: 선택한 Model의 Parameter
        :type modelParamter: dictionary
        
        :param trainParameter: 모델 학습 Parameter
        :type trainParameter: dictionary
        """
        if self.parameter['model_name'] == "SemiTime_cf":
            train_labeled_x = data_dict['train_labeled_x']
            train_labeled_y = data_dict['train_labeled_y']
            train_unlabeled_x = data_dict['train_unlabeled_x']
            val_x = data_dict['val_x']
            val_y = data_dict['val_y']
        elif self.parameter['model_name'] == "SMATE_cf":
            train_x = data_dict['train_x']
            train_y = data_dict['train_y']
            val_x = data_dict['val_x']
            val_y = data_dict['val_y']
        else:
            print('Choose the model correctly')

        dim = 3
        if self.parameter['model_name'] == "FC_cf":
           dim = 2
        # if type(train_x) !=  np.ndarray:
        #     train_x, train_y = transDFtoNP(train_x, train_y, windowNum, dim)
        #     val_x, val_y = transDFtoNP(val_x, val_y, windowNum, dim)
        
        # 22/10/24 수정 (input data-> transposed : # (Batch, timestep, variables))
        self.parameter['input_size'] = val_x.shape[2]
        if dim != 2:
            self.parameter['seq_len']  = val_x.shape[1] # seq_length
        

        # 아래 줄은 나중에 input data preparation 완성되면 다시 필요 (22/10/24)
        # train_labeled_x, train_labeled_y, train_unlabeled_x = self.labeled_unlabeled_split(train_x, train_y)
        
        # dataloaders_dict 수정 (22/10/24)
        if self.parameter['model_name'] == "SemiTime_cf":
            train_labeled_loader = self.get_labeled_dataloader(train_labeled_x, train_labeled_y, 
                                                               batch_size=batch_size, shuffle=True)
            train_unlabeled_loader = self.get_semitime_dataloader(train_labeled_x, train_unlabeled_x, 
                                                                  batch_size=batch_size, shuffle=True)
            valid_loader = self.get_labeled_dataloader(val_x, val_y, batch_size=batch_size, shuffle=False)
            self.dataloaders_dict = {'train_labeled': train_labeled_loader, 'train_unlabeled' : train_unlabeled_loader, 'val': valid_loader}
        
        elif self.parameter['model_name'] == "SMATE_cf":
            train_loader = self.get_labeled_dataloader(train_x, train_y, batch_size=batch_size, shuffle=True)
            valid_loader = self.get_labeled_dataloader(val_x, val_y, batch_size=batch_size, shuffle=False)
            self.dataloaders_dict = {'train': train_loader, 'val': valid_loader}
        else:
            print('Choose the model correctly')

    def getModel(self):
        """
        Build model and return initialized model for selected model_name
        """
       
        ## 2022/10/24 추가
        if self.parameter['model_name'] == 'SemiTime_cf':
            init_model = SimConv4(feature_size = self.parameter['feature_size'])
        elif self.parameter['model_name'] == 'SMATE_cf':
            init_model = SMATE(num_seq = self.parameter['num_seq'],
                               input_size = self.parameter['input_size'], 
                               num_classes = self.parameter['num_classes'],
                               sup_ratio = self.parameter['sup_ratio'],
                               p_ratio = self.parameter['p_ratio'],
                               d_prime_ratio = self.parameter['d_prime_ratio'],
                               kernels = self.parameter['kernels'],
                               num_layers = self.parameter['num_layers'],
                               hidden_dim = self.parameter['hidden_dim'])

        else:
            print('Choose the model correctly')
            
        return init_model
    
    def trainModel(self, init_model, modelFilePath, num_epochs):
        """
        Train model and return best model

        :return: best trained model
        :rtype: model
        """

        print("Start training model")
        
        # train model
        init_model = init_model.to(self.device)
        

        criterion = nn.CrossEntropyLoss() # 각 모델에서 다시 criterion을 정의하므로 CorssEntropyLoss로 고정은 아님
        optimizer = optim.Adam(init_model.parameters(), lr=self.parameter['lr']) # 각 모델에서 다시 optim.을 정의하고 있음
        
        # self.trainer, input args. 수정 (22/10/24) -> batch_size, optimizer, num_epochs 는 training.iypnb 에서 수정
        
        # self.trainer.train 안에 criterion, optimizer을 넣은 이유는 기존 KETI 문법과 동일하게 하려고
        if self.parameter['model_name'] == 'SemiTime_cf':
            trainer = SemiTime_Train_Test(self.parameter)
            self.best_model, self.timeElapsed = trainer.train(init_model, self.dataloaders_dict, criterion, num_epochs, optimizer, self.device)
        elif self.parameter['model_name'] == 'SMATE_cf':
            trainer = Train_SMATE(self.parameter)
            self.best_model, self.timeElapsed = trainer.train(init_model, self.dataloaders_dict, criterion, num_epochs, optimizer, self.device)
        else:
            print('Choose the model correctly')

        self._trainSaveModel(self.best_model, modelFilePath)
        
        return self.best_model, self.timeElapsed
        
    def _trainSaveModel(self, best_model, modelFilePath):
        """
        Save the best trained model

        :param best_model: best trained model
        :type best_model: model

        """

        # save_model 수정 (22/10/24)
        # 1개의 단일 모델만 저장할 경우
        if type(best_model) != list:
            torch.save(best_model.state_dict(), modelFilePath[0])

        # 여러개의 모델을 저장해야 할 경우
        else:
            torch.save({'backbone':best_model[0].state_dict(),
                        'sup_head':best_model[1].state_dict(), 
                        'relation_head':best_model[2].state_dict()} , modelFilePath[0])
                            

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
        if self.parameter['model_name'] != 'SMATE_cf' and np.min(y_data) != 0:
            print('Set start class as zero')
            y_data = y_data - np.min(y_data)
        
        # for labeled data
        dataset = torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)            

        return data_loader

    def get_semitime_dataloader(self, x_labeled, x_unlabeled, batch_size, shuffle):
        """
        Get DataLoader
        
        :param x_labeled: input labeled x
        :type x_labeled: numpy array

        :param x_unlabeled: input unlabeled x
        :type x_unlabeled: numpy array

        :param batch_size: batch size
        :type batch_size: int

        :param shuffle: shuffle for making batch
        :type shuffle: bool

        :return: dataloader
        :rtype: DataLoader
        """
        
        # Labeled와 Unlabeled Concat
        concat_x = torch.cat([torch.FloatTensor(x_labeled), torch.FloatTensor(x_unlabeled)], dim=0)

        # Past 길이 도출
        seq_len = concat_x.shape[1]
        past_length = round(seq_len * self.parameter['alpha'])
        
        # Past와 Future벡터 생성
        data_P = concat_x[:, :past_length, :]
        data_F = concat_x[:, past_length:, :]
        
        # 데이터 loader 구축
        dataset = torch.utils.data.TensorDataset(data_P, data_F)
        data_loader = torch.utils.data.DataLoader(dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=shuffle) 

        return data_loader

    def labeled_unlabeled_split(self, train_data_x, train_data_y):

        # unlabeled idx 구분
        nan_idx = np.argwhere(np.isnan(train_data_y)).reshape(-1)
        real_idx = np.argwhere(np.isfinite(train_data_y)).reshape(-1)

        # idx를 활용해서 labeled / unlabeled data 구분
        train_labeled_x = train_data_x[real_idx]
        train_labeled_y = train_data_y[real_idx]
        train_unlabeled_x = train_data_x[nan_idx]

        # shuffle train_unlabeled_x (labeled data는 train_test_split함수 쓰면서 shuffle 예정)
        unlabeled_shuffled_index = [x for x in range(train_unlabeled_x.shape[0])]
        random.shuffle(unlabeled_shuffled_index)
        train_unlabeled_x = train_unlabeled_x[unlabeled_shuffled_index]

        return train_labeled_x, train_labeled_y, train_unlabeled_x


############################# SMATE Trainer #############################
class Train_SMATE():
    def __init__(self, parameter):
        """
        Initialize Train_Test class

        :param config: configuration
        :type config: dictionary
        """

        self.parameter = parameter

    def train(self, model, dataloaders, criterion, num_epochs, optimizer, device):
        """
        Train the model

        :param model: initialized model
        :type model: model

        :param dataloaders: train & validation dataloaders
        :type dataloaders: dictionary

        :return: trained model
        :rtype: model
        """

        since = time.time()

        model = model.cuda()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.parameter['lr'])

        # best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 99999999

        for epoch in range(num_epochs):
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print()
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 training mode로 설정
                else:
                    model.eval()   # 모델을 validation mode로 설정

                running_loss = 0.0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()

                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        reg_loss, dec_output = model(inputs, labels)
                        loss1 = criterion(inputs, dec_output)
                        loss2 = reg_loss
                        loss = loss1 + loss2

                        # backward (optimize): training 단계에서만 수행
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # batch별 loss를 축적함
                    running_loss += loss.item() * inputs.size(0)
                    # running_corrects += torch.sum(preds == labels.data)
                    running_total += labels.size(0)

                    # epoch의 loss 및 accuracy 도출
                epoch_loss = running_loss / running_total
                # epoch_acc = running_corrects.double() / running_total

                if epoch == 0 or (epoch + 1) % 10 == 0:
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.encoder.state_dict())

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))

        # validation accuracy가 가장 높았을 때의 best model 가중치를 불러와 best model을 구축함
        model.encoder.load_state_dict(best_model_wts)

        return model.encoder, time_elapsed

############################# SemiTime Trainer #############################
class SemiTime_Train_Test():
    def __init__(self, parameter):
        """
        Initialize Train_Test class

        :param config: configuration
        :type config: dictionary
        """

        # init 수정 (22/10/24)
        self.parameter = parameter
        self.feature_size = self.parameter['feature_size']
        self.num_classes = self.parameter['num_classes']

        # batch_size, num_epochs, device 는 processInputData에서 args로 받아옴
        

    def train(self, backbone_model, dataloaders, criterion, num_epochs, optimizer, device):
        """
        Train the model

        :param backbone_model: backbone_model
        :type backbone_model: model

        :param dataloaders: train & validation dataloaders
        :type dataloaders: dictionary

        :return: trained model, trained model, trained model (backbone, sup_head, relation_head)
        :rtype: model, model, model
        """

        # 모델정의
        backbone = backbone_model.to(device)
        relation_head = torch.nn.Sequential(
                                torch.nn.Linear(self.feature_size*2, 256),
                                torch.nn.BatchNorm1d(256),
                                torch.nn.LeakyReLU(),
                                torch.nn.Linear(256, 1)).to(device)
        sup_head = torch.nn.Sequential(
                        torch.nn.Linear(self.feature_size, self.num_classes)).to(device)

        # 학습 시작시간 정의
        since = time.time()

        # Loss Function 정의
        crossentropy = nn.CrossEntropyLoss()
        binarycrossentropy = torch.nn.BCEWithLogitsLoss()
        
        # Optimizer 정의
        optimizer = optim.Adam([
                    {'params': backbone.parameters()},
                    {'params': relation_head.parameters()},
                    {'params': sup_head.parameters()}], lr=self.parameter['lr'])

        # Weight 및 값 초기화
        best_backbone_wts = copy.deepcopy(backbone.state_dict())
        best_relation_head_wts = copy.deepcopy(relation_head.state_dict())
        best_sup_head_wts = copy.deepcopy(sup_head.state_dict())
        valid_best_acc = 0.0

        for epoch in range(num_epochs):

            # Epoch 단위 Loss
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print()
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # 학습모드 전환
            backbone.train()
            relation_head.train()
            sup_head.train()

            # 값 초기화
            running_labeled_loss = 0.0
            running_labeled_corrects = 0
            running_labeled_total = 0

            running_unlabeled_loss = 0.0
            running_unlabeled_corrects = 0
            running_unlabeled_total = 0

            running_valid_loss = 0.0
            running_valid_corrects = 0
            running_valid_total = 0

            # Training Phase (1): Labeled data
            for inputs, labels in dataloaders['train_labeled']:
                
                # parameter gradients를 0으로 설정
                optimizer.zero_grad()

                # input과 output을 gpu에 태우기
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.long)

                # labeled loss 산출
                output = backbone(inputs)
                output = sup_head(output)
                loss_labeled = crossentropy(output, labels)

                # 가중치 update
                loss_labeled.backward()
                optimizer.step()

                # 각 iter마다 acc 구한 후 acc추가
                preds_labeled = output.argmax(-1)
                corrects_labeled = preds_labeled.eq(labels.view_as(preds_labeled)).sum()

                # batch별 loss를 축적함
                running_labeled_loss += loss_labeled.item() * inputs.size(0)
                running_labeled_corrects += corrects_labeled
                running_labeled_total += labels.size(0)

            # Training Phase (2): Unlabeled data
            for inputs_P, inputs_F in dataloaders['train_unlabeled']:
                
                # parameter gradients를 0으로 설정
                optimizer.zero_grad()

                # Past와 Future 데이터를 gpu에 태우기
                inputs_P = inputs_P.to(device)
                inputs_F = inputs_F.to(device)

                # past와 feature들 모두 backbone 통과
                features_P = backbone(inputs_P) ### Anchore
                features_F = backbone(inputs_F) ### Positive

                # negative pair 정의
                """
                torch.roll 활용 
                - 배치 내 1번째 데이터의 Past가 Anchor면 2번째 데이터의 Future가 Negative
                - 배치 내 마지막 데이터의 Past가 Anchor면 1번째 데이터의 Future가 Negative
                """
                features_F_rolled = torch.roll(features_F, shifts=1, dims=0) ### Negative

                # Pair 정의
                pos_pair = torch.cat([features_P, features_F], 1)
                neg_pair = torch.cat([features_P, features_F_rolled], 1)
                relation_pairs = torch.cat([pos_pair, neg_pair], 0).cuda() 
                targets = torch.cat([torch.ones(inputs_P.shape[0], dtype=torch.float32), torch.zeros(inputs_P.shape[0], dtype=torch.float32)], 0).cuda()

                # relation_head에 값들 통과
                output = relation_head(relation_pairs).squeeze()

                # Loss 산출
                loss_unlabeled = binarycrossentropy(output, targets)

                # 가중치 update
                loss_unlabeled.backward()
                optimizer.step()

                # 각 iter마다 acc 구한 후 acc추가
                preds_unlabeled = torch.round(torch.sigmoid(output))
                corrects_unlabeled = preds_unlabeled.eq(targets.view_as(preds_unlabeled)).sum()       

                # batch별 loss를 축적함
                running_unlabeled_loss += loss_unlabeled.item() * inputs_P.size(0) * 2 ### x2는 Pos Pair와 Neg Pair 2개가 존재하기 때문
                running_unlabeled_corrects += corrects_unlabeled
                running_unlabeled_total += inputs_P.size(0) * 2 ### x2는 Pos Pair와 Neg Pair 2개가 존재하기 때문

            # Validation Phase
            for inputs, labels in dataloaders['val']:

                # Inference모드 전환
                backbone.eval()
                relation_head.eval()
                sup_head.eval()

                # input과 output을 gpu에 태우기
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.long)

                # labeled loss 산출
                output = backbone(inputs)
                output = sup_head(output)
                loss_valid = crossentropy(output, labels)

                # 각 iter마다 acc 구한 후 acc추가
                preds_valid = output.argmax(-1)
                corrects_valid = preds_valid.eq(labels.view_as(preds_valid)).sum()

                # batch별 loss를 축적함
                running_valid_loss += loss_valid.item() * inputs.size(0)
                running_valid_corrects += corrects_valid
                running_valid_total += labels.size(0)

            # epoch 단위의 loss 및 accuracy 도출
            epoch_labeled_loss = running_labeled_loss / running_labeled_total
            epoch_labeled_acc = running_labeled_corrects.double() / running_labeled_total
            epoch_unlabeled_loss = running_unlabeled_loss / running_unlabeled_total
            epoch_unlabeled_acc = running_unlabeled_corrects.double() / running_unlabeled_total
            epoch_valid_loss = running_valid_loss / running_valid_total
            epoch_valid_acc = running_valid_corrects.double() / running_valid_total

            # Print log
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print('train(Labeled) Loss: {:.4f} Acc: {:.4f}'.format(epoch_labeled_loss, epoch_labeled_acc))
                print('train(Unlabeled) Loss: {:.4f} Acc: {:.4f}'.format(epoch_unlabeled_loss, epoch_unlabeled_acc))
                print('val Loss: {:.4f} Acc: {:.4f}'.format(epoch_valid_loss, epoch_valid_acc))
            
            # validation 단계에서 validation loss가 감소하면 Best model 새로 저장
            if epoch_valid_acc > valid_best_acc:
                valid_best_acc = epoch_valid_acc
                best_backbone_wts = copy.deepcopy(backbone.state_dict())
                best_relation_head_wts = copy.deepcopy(relation_head.state_dict())
                best_sup_head_wts = copy.deepcopy(sup_head.state_dict())

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(valid_best_acc))

        # validation accuracy가 가장 높았을 때의 best model 가중치를 불러와 best model을 구축함
        backbone.load_state_dict(best_backbone_wts)
        sup_head.load_state_dict(best_sup_head_wts)
        relation_head.load_state_dict(best_relation_head_wts)
        best_model = [backbone, sup_head, relation_head]

        return best_model, time_elapsed

    # Test 부분은 어디에 넣어야 할지... KETIToolDL.CLUSTTool 에 다시 만들어야 하는지?
    # def test(self, backbone, sup_head, test_loader):
    #     """
    #     Predict classes for test dataset based on the trained model

    #     :param backbone: best trained backbone
    #     :type model: model

    #     :param sup_head: best trained sup_head
    #     :type model: model

    #     :param test_loader: test dataloader
    #     :type test_loader: DataLoader

    #     :return: predicted classes
    #     :rtype: numpy array
    #     """

    #     # 각 모델을 gpu에 태우기
    #     backbone = backbone.to(device)
    #     sup_head = sup_head.to(device)
        
    #     # Inference 모드로 전환
    #     backbone.eval()
    #     sup_head.eval()   
        
    #     # test_loader에 대하여 검증 진행 (gradient update 방지)
    #     with torch.no_grad():
    #         preds = []
    #         probs = []
    #         label_list = []
    #         corrects = 0
    #         total = 0
    #         for inputs, labels in test_loader:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device, dtype=torch.long)

    #             # input을 model에 넣어 output을 도출
    #             output = backbone(inputs)
    #             output = sup_head(output)
    #             prob = nn.Softmax(dim=1)(output)

    #             # 예측된 pred값을 append
    #             pred = output.argmax(-1)
    #             corrects += torch.sum(pred == labels.data)
    #             total += labels.size(0)

    #             preds.extend(pred.detach().cpu().numpy())
    #             probs.extend(prob.detach().cpu().numpy())
    #             label_list.extend(labels.detach().cpu().numpy())

    #     return  np.array(preds), np.array(probs)


# ## Supervised Learning Trainer ## 
# class Train_Test():
#     def __init__(self, config):
#         """
#         Initialize Train_Test class

#         :param config: configuration
#         :type config: dictionary
#         """

#         self.model = config['model']
#         self.parameter = config['parameter']
#         self.num_epochs = self.parameter['num_epochs']
#         self.device = self.parameter['device']

#     def train(self, model, dataloaders):
#         """
#         Train the model

#         :param model: initialized model
#         :type model: model

#         :param dataloaders: train & validation dataloaders
#         :type dataloaders: dictionary

#         :return: trained model
#         :rtype: model
#         """

#         since = time.time()

#         model = model.to(self.device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=self.parameter['lr'])

#         val_acc_history = []

#         best_model_wts = copy.deepcopy(model.state_dict())
#         best_acc = 0.0

#         for epoch in range(self.num_epochs):
#             if epoch == 0 or (epoch + 1) % 10 == 0:
#                 print()
#                 print('Epoch {}/{}'.format(epoch + 1, self.num_epochs))

#             # 각 epoch마다 순서대로 training과 validation을 진행
#             for phase in ['train', 'val']:
#                 if phase == 'train':
#                     model.train()  # 모델을 training mode로 설정
#                 else:
#                     model.eval()   # 모델을 validation mode로 설정

#                 running_loss = 0.0
#                 running_corrects = 0
#                 running_total = 0

#                 # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
#                 for inputs, labels in dataloaders[phase]:
#                     inputs = inputs.to(self.device)
#                     labels = labels.to(self.device, dtype=torch.long)
                    
#                     # parameter gradients를 0으로 설정
#                     optimizer.zero_grad()

#                     # forward
#                     # training 단계에서만 gradient 업데이트 수행
#                     with torch.set_grad_enabled(phase == 'train'):
#                         # input을 model에 넣어 output을 도출한 후, loss를 계산함
#                         outputs = model(inputs)
#                         loss = criterion(outputs, labels)

#                         # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
#                         _, preds = torch.max(outputs, 1)

#                         # backward (optimize): training 단계에서만 수행
#                         if phase == 'train':
#                             loss.backward()
#                             optimizer.step()

#                     # batch별 loss를 축적함
#                     running_loss += loss.item() * inputs.size(0)
#                     running_corrects += torch.sum(preds == labels.data)
#                     running_total += labels.size(0)

#                 # epoch의 loss 및 accuracy 도출
#                 epoch_loss = running_loss / running_total
#                 epoch_acc = running_corrects.double() / running_total

#                 if epoch == 0 or (epoch + 1) % 10 == 0:
#                     print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

#                 # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
#                 if phase == 'val' and epoch_acc > best_acc:
#                     best_acc = epoch_acc
#                     best_model_wts = copy.deepcopy(model.state_dict())
#                 if phase == 'val':
#                     val_acc_history.append(epoch_acc)

#         # 전체 학습 시간 계산
#         time_elapsed = time.time() - since
#         print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#         print('Best val Acc: {:4f}'.format(best_acc))

#         # validation accuracy가 가장 높았을 때의 best model 가중치를 불러와 best model을 구축함
#         model.load_state_dict(best_model_wts)
#         return model

#     def test(self, model, test_loader):
#         """
#         Predict classes for test dataset based on the trained model

#         :param model: best trained model
#         :type model: model

#         :param test_loader: test dataloader
#         :type test_loader: DataLoader

#         :return: predicted classes
#         :rtype: numpy array
#         """

#         model = model.to(self.device)
#         model.eval()   # 모델을 validation mode로 설정
        
#         # test_loader에 대하여 검증 진행 (gradient update 방지)
#         with torch.no_grad():
#             preds = []
#             probs = []
            
#             for inputs, labels in test_loader:
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device, dtype=torch.long)

#                 # forward
#                 # input을 model에 넣어 output을 도출
#                 outputs = model(inputs)
#                 outputs = nn.Softmax(dim=1)(outputs)

#                 # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
#                 _, pred = torch.max(outputs, 1)

#                 preds.extend(pred.detach().cpu().numpy())
#                 probs.extend(outputs.detach().cpu().numpy())

#         return np.array(preds), np.array(probs)