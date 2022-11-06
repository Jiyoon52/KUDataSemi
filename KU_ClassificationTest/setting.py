##########################
# Default List Information
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import pandas as pd
import random

modelConfig={
    "LSTM_cf":{# Case 1. LSTM model (w/o data representation)
        'num_layers': 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
        'hidden_size': 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
        'dropout': 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        'bidirectional': True,  # 모델의 양방향성 여부, bool(default: True)   
        "lr":0.0001
    },
    "GRU_cf":{# Case 2. GRU model (w/o data representation)
        'num_layers': 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
        'hidden_size': 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
        'dropout': 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        'bidirectional': True,  # 모델의 양방향성 여부, bool(default: True)
        "lr":0.0001
        
    },
    "CNN_1D_cf":{# Case 3. CNN_1D model (w/o data representation)
        'output_channels': 64, # convolution layer의 output channel, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
        'kernel_size': 3, # convolutional layer의 filter 크기, int(default: 3, 범위: 3 이상, 홀수로 설정 권장)
        'stride': 1, # convolution layer의 stride 크기, int(default: 1, 범위: 1 이상)
        'padding': 0, # padding 크기, int(default: 0, 범위: 0 이상)
        'drop_out': 0.1, # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        "lr":0.0001
    },
    "LSTM_FCNs_cf":{#Case 4. LSTM_FCNs model (w/o data representation)
        'num_layers': 1,  # recurrent layers의 수, int(default: 1, 범위: 1 이상)
        'lstm_drop_out': 0.4, # LSTM dropout 확률, float(default: 0.4, 범위: 0 이상 1 이하)
        'fc_drop_out': 0.1, # FC dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        "lr":0.0001
    },
    "FC_cf":{# Case 5. fully-connected layers (w/ data representation)
        'drop_out': 0.1, # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        'bias': True,# bias 사용 여부, bool(default: True)
        "lr":0.0001
    }, 
     "SemiTime_cf": {  # Case 6. SemiTime model (w/o data representation)
        # 'model': 'SemiTime', 
        # 'best_model_path': ['./ckpt/semitime_backbone.pt', './ckpt/semitime_sup_head.pt', './ckpt/semitime_relation_head.pt',],  # 학습 완료 모델 저장 경로
        'input_size': 9,  # 데이터의 변수 개수, int -> input data.shape[0]으로 변경 해야함
        'num_classes': 6,  # 분류할 class 개수, int -> 다른 곳으로 보내야함 trainer.py
        'alpha' : 0.5, # 전체 length에서 past의 비율 -> OK
        'feature_size' : 64, # backbone의 Output feature size -> OK
        'lr': 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하) -> OK
        'device': 'cuda'  # 학습 환경, ["cuda", "cpu"] 중 선택 -> Training.ipynb 에 device 선언이 있음
    },
    "SMATE_cf": {  # Case 7. SMATE model (w/o data representation)
        # 'model': 'SMATE',
        # 'best_model_path': './ckpt/smate_encoder.pt',  # 학습 완료 모델 저장 경로

        'num_seq': 128,  # windowing(sequence) length, int
        'input_size': 9,  # 데이터의 변수 개수, int
        'num_classes' : 6, # 분류할 class 개수, int
        'sup_ratio': 0.3,  # semi-supervised의 경우 supervised(labeled) data의 비율
        'p_ratio': 0.1, # pooling size 결정 계수 (Pool_size = round(p_ratio * T))
        'd_prime_ratio': 1.0, # d_prime 결정 계수 (d_prime = round(d_prime_ratio * N))
        'kernels' : [7, 5, 3], # conv kernels
        'num_layers': 3,
        'hidden_dim':128,
        'lr': 0.0001,  # learning rate, float(default: 0.001, 범위: 0.1 이하)
        'device': 'cuda'  # 학습 환경, ["cuda", "cpu"] 중 선택
  
    }
}

###########################
# Get Data From Files
import pickle
import pandas as pd
def getTrainDataFromFilesForClassification(folderAddress, model_name):
    if model_name in ["LSTM_cf","GRU_cf", "CNN_1D_cf","LSTM_FCNs_cf", "SemiTime_cf", "SMATE_cf"]:
        # raw time series data
        train_x = pickle.load(open(folderAddress+'x_train.pkl', 'rb'))
        train_y = pickle.load(open(folderAddress+'y_train.pkl', 'rb'))
        test_x = pickle.load(open(folderAddress+'x_test.pkl', 'rb'))
        test_y = pickle.load(open(folderAddress+'y_test.pkl', 'rb'))

        print(train_x.shape)  
        print(train_y.shape) 
        print(test_x.shape)  
        print(test_y.shape)  

    if model_name in ["FC_cf"]:
        # representation data
        train_x = pd.read_csv(folderAddress+'ts2vec_repr_train.csv')
        train_y = pickle.load(open(folderAddress+'y_train.pkl', 'rb'))

        test_x = pd.read_csv(folderAddress+'ts2vec_repr_test.csv')
        test_y = pickle.load(open(folderAddress+'y_test.pkl', 'rb'))

        print(train_x.shape)  #shape : (num_of_instance x representation_dims) = (7352, 64)
        print(train_y.shape) #shape : (num_of_instance) = (7352, )
        print(test_x.shape)  #shape : (num_of_instance x representation_dims) = (2947, 64)
        print(test_y.shape)  #shape : (num_of_instance) = (2947, )
    
    return train_x, train_y,test_x, test_y

def load_data(folderAddress, model_name):
    if model_name in ["LSTM_cf","GRU_cf", "CNN_1D_cf","LSTM_FCNs_cf", "SemiTime_cf", "SMATE_cf"]:
        # raw time series data
        train_x = pickle.load(open(folderAddress + 'x_train.pkl', 'rb'))
        test_x = pickle.load(open(folderAddress + 'x_test.pkl', 'rb'))
        test_y = pickle.load(open(folderAddress + 'y_test.pkl', 'rb'))

        if model_name in ('SemiTime_cf', 'SMATE_cf'):
            train_y = pickle.load(open(folderAddress + 'y_train_semi.pkl', 'rb'))
            train_x = train_x.transpose((0, 2, 1))  # (B, timestep, variables)
            test_x = test_x.transpose((0, 2, 1))  # (B, timestep, variables)
        else:
            train_y = pickle.load(open(folderAddress + 'y_train.pkl', 'rb'))

        print(train_x.shape)
        print(train_y.shape)
        print(test_x.shape)
        print(test_y.shape)

    if model_name in ["FC"]:
        # representation data
        train_x = pd.read_csv(folderAddress + 'ts2vec_repr_train.csv')
        train_y = pickle.load(open(folderAddress + 'y_train.pkl', 'rb'))

        test_x = pd.read_csv(folderAddress + 'ts2vec_repr_test.csv')
        test_y = pickle.load(open(folderAddress + 'y_test.pkl', 'rb'))

    return train_x, train_y, test_x, test_y


def get_train_val_data(data_list, scaler_path):

    """
    data_list로 parameter 수정
     - train_data를 처음으로 넣을 것
     - Supervised Learning: train_data, valid_data
     - Semi-supervised Learning: train_labeled_data, train_unlabeled_data, valid_data
    """

    # normalization
    scaler = MinMaxScaler()

    # scaling을 할 객체 정의 (train_labeled)
    train_data = data_list[0]

    if len(train_data.shape) == 1:  # shape=(time_steps, )
        scaler = scaler.fit(np.expand_dims(train_data, axis=-1))
    elif len(train_data.shape) < 3:  # shape=(num_of_instance, input_dims)
        scaler = scaler.fit(train_data)
    else:  # shape=(num_of_instance, input_dims, time_steps)
        origin_shape = train_data.shape
        scaler = scaler.fit(np.transpose(train_data, (0, 2, 1)).reshape(-1, origin_shape[1]))

    scaled_data = []
    for data in data_list:
        if len(train_data.shape) == 1:  # shape=(time_steps, )
            data = scaler.transform(np.expand_dims(data, axis=-1))
            data = data.flatten()
        elif len(data.shape) < 3:  # shape=(num_of_instance, input_dims)
            data = scaler.transform(data)
        else:  # shape=(num_of_instance, input_dims, time_steps)
            data = scaler.transform(np.transpose(data, (0, 2, 1)).reshape(-1, origin_shape[1]))
            data = np.transpose(data.reshape(-1, origin_shape[2], origin_shape[1]), (0, 2, 1))
        scaled_data.append(data)

    # save scaler
    print(f"Save MinMaxScaler in path: {scaler_path}")
    pickle.dump(scaler, open(scaler_path, 'wb'))
    return scaled_data

def labeled_unlabeled_split(train_data_x, train_data_y):

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

def get_test_data(test_data, scaler_path):
    # load scaler
    scaler = pickle.load(open(scaler_path, 'rb'))

    # normalization
    if len(test_data.shape) == 1:  # shape=(time_steps, )
        scaled_test_data = scaler.transform(np.expand_dims(test_data, axis=-1))
        scaled_test_data = scaled_test_data.flatten()
    elif len(test_data.shape) < 3:  # shape=(num_of_instance, input_dims)
        scaled_test_data = scaler.transform(test_data)
    else:  # shape=(num_of_instance, input_dims, time_steps)
        origin_shape = test_data.shape
        scaled_test_data = scaler.transform(np.transpose(test_data, (0, 2, 1)).reshape(-1, origin_shape[1]))
        scaled_test_data = np.transpose(scaled_test_data.reshape(-1, origin_shape[2], origin_shape[1]), (0, 2, 1))
    return scaled_test_data, scaler