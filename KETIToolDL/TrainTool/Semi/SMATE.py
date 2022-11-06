from tkinter import Label
from turtle import xcor
import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F

from sklearn import svm
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def euclidean_dist(x, y):
    # x: n * d
    # y: m * d

    n = x.shape[0]
    l = x.shape[1]
    d = x.shape[2]
    m = y.shape[0]
    
    assert d == y.shape[2]

    x = x.reshape(x.shape[0], -1) # n * m * d
    y = y.reshape(y.shape[0], -1)

    x = torch.repeat_interleave(x.unsqueeze(1), m, dim=1)
    y = y.unsqueeze(0) # 1 * m * d

    return torch.sum(torch.pow(x-y, 2), axis = 2) # n * m

class spatial_dynamic_block1(nn.Module):
    def __init__(self, input_size, pool_size, d_prime, hidden_dim):
        super(spatial_dynamic_block1, self).__init__()
        
        padding_size = round((pool_size - 1) / 2)
        
        self.avg_pool = nn.AvgPool1d(kernel_size = pool_size, stride=1, padding=padding_size)
        self.linear = nn.LazyLinear(out_features = input_size)

    def forward(self, input):
        x = self.avg_pool(input.transpose(1,2))
        x = self.linear(x.transpose(1,2))
        x = nn.ReLU(inplace=True)(x)
        x = self.linear(x)
        x = nn.Sigmoid()(x)
        x = torch.multiply(input, x)

        return x

class spatial_dynamic_block2(nn.Module):
    def __init__(self, input_size, pool_size, d_prime, hidden_dim):
        super(spatial_dynamic_block2, self).__init__()
        
        padding_size = int((pool_size - 1) / 2)
        
        self.avg_pool = nn.AvgPool1d(kernel_size = pool_size, stride=1, padding=padding_size)
        self.linear = nn.LazyLinear(out_features = hidden_dim)

    def forward(self, input):
        x = self.avg_pool(input.transpose(1,2))
        x = self.linear(x.transpose(1,2))
        x = nn.ReLU(inplace=True)(x)
        x = self.linear(x)
        x = nn.Sigmoid()(x)
        x = torch.multiply(input, x)

        return x


class SMATE_encoder(nn.Module):
    def __init__(self, *, input_size, hidden_dim, num_layers, bidirectional = False, pool_step, d_prime, kernels):
        super(SMATE_encoder, self).__init__()
        self.num_features = input_size
        self.bidirectional = 2 if bidirectional == True else 1
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_layers
        self.d_prime = d_prime
        self.kernels = kernels
        self.pool_step = pool_step
        self.padding_size = int((pool_step - 1) / 2)
        self.gru1 = nn.GRU(input_size=self.num_features, 
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_gru_layers,
                            bidirectional = bidirectional,
                            batch_first=True)
        self.gru2 = nn.GRU(input_size=self.hidden_dim, 
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_gru_layers,
                            bidirectional = bidirectional,
                            batch_first=True)
        self.gru3 = nn.GRU(input_size=self.hidden_dim, 
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_gru_layers,
                            bidirectional = bidirectional,
                            batch_first=True)

        self.conv1 = nn.Conv1d(self.num_features, self.hidden_dim, kernel_size = 7, padding = 3)
        self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size = 3, padding = 1)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)

        self.se1 = spatial_dynamic_block1(pool_size = self.kernels[0], d_prime = self.d_prime, hidden_dim = self.hidden_dim, input_size = self.num_features)  # ex 128
        self.se2 = spatial_dynamic_block2(pool_size = self.kernels[1], d_prime = self.d_prime, hidden_dim = self.hidden_dim, input_size = self.num_features)  # ex 256
        self.se3 = spatial_dynamic_block2(pool_size = self.kernels[2], d_prime = self.d_prime, hidden_dim = self.hidden_dim, input_size = self.num_features)

        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()

        self.avgpool = nn.AvgPool1d(kernel_size = pool_step, stride=1, padding = 1)

        self.linear1 = nn.LazyLinear(out_features = 128)
        self.linear2 = nn.LazyLinear(out_features = 128)


    def forward(self, x):

        x1, _ = self.gru1(x)
        x1, _ = self.gru2(x1)
        x1, _ = self.gru3(x1)

        # x2 = x.transpose(2,1)
        x2 = self.se1(x)
        x2 = self.relu(self.bn1(self.conv1(x2.transpose(1, 2))))
        x2 = self.se2(x2)
        x2 = self.relu(self.bn2(self.conv2(x2)))
        x2 = self.se3(x2)
        x2 = self.relu(self.bn3(self.conv3(x2)))
        x2 = x2.transpose(1,2)

        x1 = nn.AvgPool1d(kernel_size = self.pool_step, stride = self.pool_step, padding=self.padding_size)(x1.transpose(1,2)).transpose(1,2)
        x2 = nn.AvgPool1d(kernel_size = self.pool_step, stride = self.pool_step, padding=self.padding_size)(x2.transpose(1,2)).transpose(1,2)

        x_all = torch.cat((x1,x2),dim=-1)
        x_out = self.linear1(x_all)
        x_out = self.bn1(x_out.transpose(1,2)).transpose(1,2)
        x_out = self.leaky(x_out)
        x_out = self.linear2(x_out)
        x_out = self.bn1(x_out.transpose(1,2)).transpose(1,2)

        # (7352, 10, 128) ==> (Sample 갯수, Window Length(128 -> 10), Variable(9 -> 128))

        return x_out

class SMATE_decoder(nn.Module):
    def __init__(self, *, hidden_dim, output_dim=9, num_layers, bidirectional = False, upsample_size):
        super(SMATE_decoder, self).__init__()
        self.upsample_size = upsample_size
        self.num_features = 128
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gru_layers = num_layers
        self.bidirectional = 2 if bidirectional == True else 1

        self.upsample = nn.Upsample(scale_factor = self.upsample_size)
        self.gru1 = nn.GRU(input_size=self.num_features, 
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_gru_layers,
                            bidirectional = bidirectional,
                            batch_first=True)
        self.gru2 = nn.GRU(input_size=self.hidden_dim, 
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_gru_layers,
                            bidirectional = bidirectional,
                            batch_first=True)
        self.gru3 = nn.GRU(input_size=self.hidden_dim, 
                            hidden_size=self.output_dim,
                            num_layers=self.num_gru_layers,
                            bidirectional = False,
                            batch_first=True)

    def forward(self, x): # x : B, window, variable
        x = self.upsample(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.gru1(x) # x : B, window, var ( 256, 128, 128 )
        x, _ = self.gru2(x) # x : B, window, var ( 256, 128, 128 )
        x, _ = self.gru3(x) # x : B, window, var ( 256, 128, 128 )

        return x



class SMATE(nn.Module):
    def __init__(self, num_seq, input_size, num_classes, sup_ratio, p_ratio, d_prime_ratio, kernels, num_layers, hidden_dim):
        super(SMATE, self).__init__()
        self.input_size = input_size
        self.window_len = num_seq
        self.num_class = num_classes
        self.sup_ratio = sup_ratio
        self.p_ratio = p_ratio
        self.d_prime_ratio = d_prime_ratio
        self.kernels = kernels # == pool_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pool_step = round(p_ratio * self.window_len)
        self.d_prime = round(d_prime_ratio * self.input_size)
        
        self.encoder = SMATE_encoder(input_size = self.input_size, num_layers = self.num_layers, hidden_dim = hidden_dim, pool_step = self.pool_step, d_prime = self.d_prime, kernels=self.kernels)
        self.decoder = SMATE_decoder(num_layers = num_layers, hidden_dim = hidden_dim, upsample_size = self.pool_step)

    def forward(self, x, label):
        enc_output = self.encoder(x)

        sup_idx = np.argwhere(np.isfinite(label.to('cpu'))).reshape(-1)
        unsup_idx = np.argwhere(np.isnan(label.to('cpu'))).reshape(-1)

        y_sup = label[sup_idx]
        h_sup = enc_output[sup_idx]

        y_sup_oneHot = np.eye(6, dtype='uint8')[y_sup.cpu().detach().numpy().astype(int)]
        y_sup_oneHot = torch.Tensor(y_sup_oneHot)
        # h_sup = enc_output[]
        proto_list = []
        
        # step 1
        for i in range(6):
            idx = np.where(y_sup.cpu() == i)[0]
            class_repr = torch.mean(h_sup[idx], dim=0, keepdim=True)
            proto_list.append(class_repr.to('cpu'))
        h_proto = torch.cat(proto_list)
        

        # step 2
        dists_sup = euclidean_dist(h_sup.to('cpu'), h_proto)
        dists_sum = torch.sum(dists_sup, dim=1, keepdims=True)
        dists_norm = dists_sup / dists_sum
        proba_sup = 1 - dists_norm #(8)
        proba_sup = torch.multiply(y_sup_oneHot, proba_sup)
        proba_sup, _ = torch.max(proba_sup, 1)
        proba_sup = proba_sup.unsqueeze(1)

        proto_list = []
        for i in range(6):
            idx = np.where(y_sup.cpu() == i)[0]
            
            for j in range(len(idx)):
                h_sup[idx][j] = torch.multiply(h_sup[idx][j].to('cpu'), proba_sup[idx][j])

            class_repr = torch.sum(h_sup[idx], axis=0, keepdim=True)
            proto_list.append(class_repr.to('cpu'))
        h_proto = torch.cat(proto_list)

        # Semi-supervised learning using unlabeled samples
        # step 3
        h_unsup = enc_output[unsup_idx]

        dists_unsup = euclidean_dist(h_unsup.to('cpu'), h_proto)
        dists_sum = torch.sum(dists_unsup, dim=1, keepdim=True)
        dists_norm = dists_unsup / dists_sum
        proba_unsup = 1 - dists_norm

        y_unsup_pseudo = torch.argmax(dists_unsup, axis=1)
        y_unsup_pseudo_oneHot = torch.nn.functional.one_hot(y_unsup_pseudo, num_classes= 6)

        proba_unsup = torch.multiply(y_unsup_pseudo_oneHot, proba_unsup)
        proba_unsup = torch.transpose(proba_unsup, 1, 0)

        proto_list = []
        for i in range(6):
            proba_i = proba_unsup[i].reshape(1, -1)
            proba_i = torch.transpose(proba_i, 1, 0)
            for j in range(len(proba_i)):
                h_unsup[j] = h_unsup[j].to('cpu') * proba_i[j]
            class_repr = torch.sum(h_unsup, dim=0, keepdim=True)
            proto_list.append(class_repr.to('cpu'))
        h_proto_unsup = torch.cat(proto_list)

        weight_sup = len(h_sup) / len(enc_output)
        weight_unsup = 1 - weight_sup
        h_proto = (weight_sup*h_proto) + (weight_unsup*h_proto_unsup)

        dec_output = self.decoder(enc_output)
        rec_size = min(x.shape[1], dec_output.shape[1])
        dec_output = dec_output[:, :rec_size, :]

        dists_sum = torch.sum(dists_sup, axis=1, keepdims=True)
        dists_norm = dists_sup / dists_sum
        y_pred = 1 - dists_norm

        reg_loss = nn.CrossEntropyLoss()(y_pred, y_sup.type(torch.LongTensor))/ len(y_sup)

        return reg_loss, dec_output



def predict_ssl(backbone_model, x_train, y_train, x_test, y_test):
        
    ls_model = LabelSpreading(kernel='knn', n_neighbors=7)

    labeled_indices = []
    unlabeled_indices = []

    for i in range(len(y_train)):
        if np.isnan(y_train[i]):
            unlabeled_indices.append(i)
        else:
            labeled_indices.append(i)

    x_sup = x_train[labeled_indices]
    x_unsup = x_train[unlabeled_indices]
    y_sup = y_train[labeled_indices]
    y_unsup = y_train[unlabeled_indices]

    indices = np.arange(len(x_train))
    unlabel_indices = indices[x_sup.shape[0]: ]
    y_sup_unsup = np.concatenate([y_sup, y_unsup])
    y_sup_unsup_train = np.copy(y_sup_unsup)
    y_sup_unsup_train[unlabel_indices] = -1

    x_fit = np.concatenate([x_sup, x_unsup], axis=0)
    h_fit = backbone_model(torch.Tensor(x_fit))
    h_fit = h_fit.reshape(h_fit.shape[0], -1)
    ls_model.fit(h_fit.detach(), y_sup_unsup_train)

    h_test = backbone_model(torch.Tensor(x_test))
    h_test = h_test.reshape(h_test.shape[0], -1)

    clf_svc = svm.SVC(kernel='linear', probability=True)
    y_fit_true = ls_model.transduction_
    clf_svc.fit(h_fit.detach(), y_fit_true)
    acc_svm = accuracy_score(y_test, clf_svc.predict(h_test.detach()))
    y_pred = clf_svc.predict(h_test.detach().to('cpu'))
    y_pred_proba = clf_svc.predict_proba(h_test.detach().to('cpu'))

    clf_svc = svm.LinearSVC()
    clf_svc.fit(h_fit.detach(), y_fit_true)
    acc_svm_linear = accuracy_score(y_test, clf_svc.predict(h_test.detach()))
    y_pred1 = clf_svc.predict(h_test.detach().to('cpu'))
    y_pred_proba_linear = clf_svc._predict_proba_lr(h_test.detach().to('cpu'))

    print('acc_svm : ', acc_svm, ' // acc_svm_linear : ', acc_svm_linear)
    print('Best performance : ', max(acc_svm, acc_svm_linear))

    if acc_svm > acc_svm_linear:
        acc = acc_svm
        pred_y = y_pred
        proba_y = y_pred_proba
    
    else:
        acc = acc_svm_linear
        pred_y = y_pred1
        proba_y = y_pred_proba_linear

    return pred_y, proba_y, acc
