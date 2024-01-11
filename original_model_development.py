# Deep Learning for Prediction of Intraoperative Hypotension by Analysing Biosignal Waveforms Acquired from Patient Monitoring

# Code for model developemtn and validation

# Author: Solam Lee, MD (solam@yonsei.ac.kr)
# Imports
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
# import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.collections as collections
import pickle
import os
import warnings
import copy

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from random import randint

# import cv2
# from torch.autograd import Function
# from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

i = 0
fig, ax = plt.subplots()
colors = ['red', 'orange', 'blue', 'cyan']
# torch.autograd.set_detect_anomaly(True)
# Prespecifications
lr = 0.0001
task = 'classification'  # either 'classification' or 'regression'
pred_lag = 300  # 300 for 5-min, 600 for 10-min, 900 for 15-min prediction or others
batch_size = 128
max_epoch = 2

cuda_number = 0  # -1 for multi GPU support
num_workers = 0

train_ratio = 0.6  # Size for training dataset
valid_ratio = 0.1  # Size for validation dataset
test_ratio = 0.3  # Size for test dataset

random_key = randint(0, 100000)  # Prespecify seed number if needed

dr_classification = 0.3  # Drop out ratio for classification model
dr_regression = 0.0  # Drop out ratio for regression model

model_dir = './model/'

if not (os.path.isdir(model_dir)):
    os.makedirs(os.path.join(model_dir))

# csv_dir = model_dir+str(random_key)+'/csv/'
pt_dir = model_dir+str(random_key)+'/pt/'

# if not (os.path.isdir(csv_dir)):
#     os.makedirs(os.path.join(csv_dir))

if not (os.path.isdir(pt_dir)):
    os.makedirs(os.path.join(pt_dir))


# Establish dataset

class dnn_dataset(torch.utils.data.Dataset):
    def __init__(self, abp, ecg, ple, co2, target, invasive, multi):
        self.invasive, self.multi = invasive, multi
        self.abp, self.ecg, self.ple, self.co2 = abp, ecg, ple, co2
        self.target = target

    def __getitem__(self, index):
        if self.invasive == True:
            if self.multi == True:  # Invasive multi-channel model
                return np.float32(np.vstack((np.array(self.abp[index]),
                                            np.array(self.ecg[index]),
                                            np.array(self.ple[index]),
                                            np.array(self.co2[index])))), np.float32(self.target[index])
            else:  # Invasive mono-channel model (arterial pressure-only model)
                return np.float32(np.array(self.abp[index])), np.float32(self.target[index])
        else:
            if self.multi == True:  # Non-invasive multi-channel model
                return np.float32(np.vstack((np.array(self.ecg[index]),
                                            np.array(self.ple[index]),
                                            np.array(self.co2[index])))), np.float32(self.target[index])
            # Non-invasive mono-channel model (photoplethysmography-only model)
            else:
                return np.float32(np.array(self.ple[index])), np.float32(self.target[index])

    def __len__(self):
        return len(self.target)


# Uni-dimensional convolutional neural network

class Net(nn.Module):
    def __init__(self, task, invasive, multi):

        self.task, self.invasive, self.multi = task, invasive, multi
        super(Net, self).__init__()

        if self.task == 'classification':
            self.dr = dr_classification
            self.final = 2
        else:
            self.dr = dr_regression
            self.final = 1

        if self.multi == True:
            self.inc = 4 if self.invasive == True else 3
        else:
            self.inc = 1

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.inc, out_channels=64,
                      kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )

        self.fc = nn.Sequential(
            nn.Linear(320, self.final),
            nn.Dropout(self.dr)
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = x.view(x.shape[0], self.inc, -1)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)

        out = out.view(x.shape[0], out.size(1) * out.size(2))

        out = self.fc(out)
        if self.task == 'classification':
            out = self.activation(out)
        return out


# Read dataset
# processed_dir = './processed/'
processed_dir = './test/'
# processed_dir = './bad/'

file_list = np.char.split(np.array(os.listdir(processed_dir)), '.')
case_list = []
for caseid in file_list:
    case_list.append(int(caseid[0]))
print('N of total cases: {}'.format(len(case_list)))

cases = {}
cases['train'], cases['valid+test'] = train_test_split(case_list,
                                                       test_size=(
                                                           valid_ratio+test_ratio),
                                                       random_state=random_key)
cases['valid'], cases['test'] = train_test_split(cases['valid+test'],
                                                 test_size=(
    test_ratio/(valid_ratio+test_ratio)),
    random_state=random_key)

for phase in ['train', 'valid', 'test']:
    print("- N of {} cases: {}".format(phase, len(cases[phase])))

for idx, caseid in enumerate(case_list):
    filename = processed_dir + str(caseid) + '.pkl'
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        data['caseid'] = [caseid] * len(data['abp'])
        raw_records = raw_records.append(pd.DataFrame(
            data)) if idx > 0 else pd.DataFrame(data)

raw_records = raw_records[(raw_records['map'] >= 20) & (
    raw_records['map'] <= 160)].reset_index(drop=True)  # Exclude abnormal range


# Define loader and model

if task == 'classification':
    task_target = 'hypo'
    criterion = nn.BCELoss()
else:
    task_target = 'map'
    criterion = nn.MSELoss()

print('\n===== Task: {}, Seed: {} =====\n'.format(task, random_key))
# print('Invasive: {}\nMulti: {}\nPred lag: {}\n'.format(invasive, multi, pred_lag))

records = raw_records.loc[(raw_records['input_length'] == 30) &
                          (raw_records['pred_lag'] == pred_lag)]

records = records[records.columns.tolist()[-1:] +
                  records.columns.tolist()[:-1]]
print('N of total records: {}'.format(len(records)))

split_records = {}
for phase in ['train', 'valid', 'test']:
    split_records[phase] = records[records['caseid'].isin(
        cases[phase])].reset_index(drop=True)
    print('- N of {} records: {}'.format(phase, len(split_records[phase])))


for invasive in [False]:
    for multi in [True]:
        print('\n\nInvasive: {}\nMulti: {}\nPred lag: {}\n'.format(
            invasive, multi, pred_lag))
        ext = {}
        for phase in ['train', 'valid', 'test']:
            ext[phase] = {}
            for x in ['abp', 'ecg', 'ple', 'co2', 'hypo', 'map']:
                ext[phase][x] = split_records[phase][x]

        dataset, loader = {}, {}
        epoch_loss, epoch_auc = {}, {}

        for phase in ['train', 'valid', 'test']:
            dataset[phase] = dnn_dataset(ext[phase]['abp'],
                                         ext[phase]['ecg'],
                                         ext[phase]['ple'],
                                         ext[phase]['co2'],
                                         ext[phase][task_target],
                                         invasive=invasive, multi=multi)
            loader[phase] = torch.utils.data.DataLoader(dataset[phase],
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        shuffle=True if phase == 'train' else False)
            epoch_loss[phase], epoch_auc[phase] = [], []

        # Model development and validation

        # torch.cuda.set_device(cuda_number)
        DNN = Net(task=task, invasive=invasive, multi=multi)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        DNN = DNN.to(device)

        optimizer = torch.optim.Adam(DNN.parameters(), lr=lr)
        n_epochs = max_epoch

        best_loss, best_auc = 99999.99999, 0.0

        for epoch in range(n_epochs):

            target_stack, output_stack = {}, {}
            current_loss, current_auc = {}, {}
            for phase in ['train', 'valid', 'test']:
                target_stack[phase], output_stack[phase] = [], []
                current_loss[phase], current_auc[phase] = 0.0, 0.0

            DNN.train()
            for dnn_inputs, dnn_target in loader['train']:
                dnn_inputs, dnn_target = dnn_inputs.to(
                    device), dnn_target.to(device)
                optimizer.zero_grad()
                dnn_output = DNN(dnn_inputs)
                loss = criterion(dnn_output[:, 0], dnn_target)
                current_loss['train'] += loss.item()*dnn_inputs.size(0)
                loss.backward()
                optimizer.step()

            current_loss['train'] = current_loss['train'] / \
                len(loader['train'].dataset)
            epoch_loss['train'].append(current_loss['train'])

            for phase in ['valid', 'test']:
                DNN.eval()
                with torch.no_grad():
                    for dnn_inputs, dnn_target in loader[phase]:
                        dnn_inputs, dnn_target = dnn_inputs.to(
                            device), dnn_target.to(device)
                        dnn_output = DNN(dnn_inputs)
                        target_stack[phase].extend(np.array(dnn_target.cpu()))
                        output_stack[phase].extend(
                            np.array(dnn_output.cpu().T[0]))

                        loss = criterion(dnn_output[:, 0], dnn_target)
                        current_loss[phase] += loss.item()*dnn_inputs.size(0)

                    current_loss[phase] = current_loss[phase] / \
                        len(loader[phase].dataset)
                    epoch_loss[phase].append(current_loss[phase])

            if task == 'classification':
                log_label = {}
                for phase in ['valid', 'test']:
                    current_auc[phase] = roc_auc_score(
                        target_stack[phase], output_stack[phase])
                    epoch_auc[phase].append(current_auc[phase])
            else:
                reg_output, reg_target, reg_label = {}, {}, {}
                for phase in ['valid', 'test']:
                    reg_output[phase] = np.array(
                        output_stack[phase]).reshape(-1, 1)
                    reg_target[phase] = np.array(
                        target_stack[phase]).reshape(-1, 1)
                    reg_label[phase] = np.where(reg_target[phase] < 65, 1, 0)
                    method = LogisticRegression(solver='liblinear')
                    # Model fitting
                    method.fit(reg_output[phase], reg_label[phase])
                    current_auc[phase] = roc_auc_score(
                        reg_label[phase], method.predict_proba(reg_output[phase]).T[1])
                    epoch_auc[phase].append(current_auc[phase])

            label_invasive = 'invasive' if invasive == True else 'noninvasive'
            label_multi = 'multi' if multi == True else 'mono'
            label_pred_lag = str(int(pred_lag / 60)) + 'min'

            filename = task+'_'+label_invasive+'_'+label_multi+'_'+label_pred_lag

            # pd.DataFrame({'train_loss': epoch_loss['train'],
            #             'valid_loss': epoch_loss['valid'],
            #             'test_loss': epoch_loss['test'],
            #             'valid_auc': epoch_auc['valid'],
            #             'test_auc': epoch_auc['test']}).to_csv(csv_dir+filename+'.csv')

            best = ''
            if task == 'regression' and abs(current_loss['valid']) < abs(best_loss):
                best = '< ! >'
                last_saved_epoch = epoch
                best_loss = abs(current_loss['valid'])
                torch.save(DNN.state_dict(), pt_dir+filename+'_epoch_best.pt')
            elif task == 'classification' and abs(current_auc['valid']) > abs(best_auc):
                best = '< ! >'
                last_saved_epoch = epoch
                best_auc = abs(current_auc['valid'])
                torch.save(DNN.state_dict(), pt_dir+filename+'_epoch_best.pt')

            torch.save(DNN.state_dict(), pt_dir+filename +
                       '_epoch_{0:03d}.pt'.format(epoch+1))

            print('Epoch [{:3d}] Train loss: {:.4f} / Valid loss: {:.4f} (AUC: {:.4f}) / Test loss: {:.4f} (AUC: {:.4f}) {}'.format
                  (epoch+1,
                   current_loss['train'],
                   current_loss['valid'], current_auc['valid'],
                   current_loss['test'], current_auc['test'], best))

        best_model_path = f'{filename}_epoch_best.pt'
        DNN.load_state_dict(torch.load(pt_dir + best_model_path))
        DNN.eval()

        # Evaluate the model on the test set
        if task == "classification":
            y_true = np.array(ext['test']['hypo'])
            y_scores = []

            with torch.no_grad():
                for dnn_inputs, dnn_target in loader['test']:
                    dnn_inputs = dnn_inputs.to(device)
                    dnn_output = DNN(dnn_inputs)
                    y_scores.extend(np.array(dnn_output.cpu().T[0]))
            # Calculate sensitivity and specificity
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i],
                     label='AUC: {:.3f}'.format(roc_auc))
        else:
            y_true = np.array(ext['test']['map'])
            y_pred = []

            with torch.no_grad():
                for dnn_inputs, dnn_target in loader['test']:
                    dnn_inputs = dnn_inputs.to(device)
                    dnn_output = DNN(dnn_inputs)
                    y_pred.extend(np.array(dnn_output.cpu().T[0]))
            errors = y_true - y_pred
            mae = np.mean(errors)
            ax.boxplot(errors, positions=[i], patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor='white', edgecolor='black'),
                       widths=0.5,
                       medianprops=dict(color=colors[i]))
        i += 1

if task == "classification":
    plt.plot([0, 1], [0, 1], color='gray', linestyle='dotted')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(f'{label_pred_lag} prediction')
    plt.legend(loc='lower right')
    plt.axis('scaled')
else:
    ax.set_xticks([])
    ax.axhline(y=0, color='gray', linestyle='dotted')
    plt.ylabel('Error in predicted value (mm Hg)')
    plt.title(f'{label_pred_lag} arterial pressure prediction')

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(f'curve/{task} {label_pred_lag} prediction.png')
plt.show()
