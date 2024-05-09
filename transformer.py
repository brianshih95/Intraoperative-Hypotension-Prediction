import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from random import randint
import math

warnings.filterwarnings("ignore")

colors = ['blue', 'cyan', 'red', 'orange']
lr = 5e-5
task = 'classification'
pred_lag = 300
batch_size = 128
max_epoch = 20

num_workers = 2

train_ratio = 0.6
valid_ratio = 0.1
test_ratio = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_key = randint(0, 100000)

model_dir = './model/'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

pt_dir = model_dir + str(random_key) + '/pt/'
if not os.path.isdir(pt_dir):
    os.makedirs(pt_dir)


class dnn_dataset(torch.utils.data.Dataset):
    def __init__(self, abp, ecg, ple, co2, target, invasive, multi):
        self.invasive, self.multi = invasive, multi
        self.abp, self.ecg, self.ple, self.co2 = abp, ecg, ple, co2
        self.target = target

    def __getitem__(self, index):
        if self.invasive:
            if self.multi:
                return np.float32(np.vstack((np.array(self.abp[index]),
                                             np.array(self.ecg[index]),
                                             np.array(self.ple[index]),
                                             np.array(self.co2[index])))), np.float32(self.target[index])
            else:
                return np.float32(np.array(self.abp[index])), np.float32(self.target[index])
        else:
            if self.multi:
                return np.float32(np.vstack((np.array(self.ecg[index]),
                                             np.array(self.ple[index]),
                                             np.array(self.co2[index])))), np.float32(self.target[index])
            else:
                return np.float32(np.array(self.ple[index])), np.float32(self.target[index])

    def __len__(self):
        return len(self.target)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, task, invasive, multi, hidden_dim, num_layers, num_heads, dim_feedforward, batch_first):
        super(TransformerModel, self).__init__()

        self.task = task
        self.invasive = invasive
        self.multi = multi

        if self.task == 'classification':
            self.final = 2
        else:
            self.final = 1

        if self.multi:
            self.inc = 4 if invasive else 3
        else:
            self.inc = 1

        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.d_model = self.inc
        self.linear1 = nn.Linear(self.d_model, 64)
        self.linear2 = nn.Linear(64, 256)
        self.linear3 = nn.Linear(256, 512)

        embedding_dim = hidden_dim
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=batch_first)
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_layers)

        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 64)
        self.linear6 = nn.Linear(64, 16)
        self.fc = nn.Linear(480, self.final)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.d_model)
        x = self.linear1(x)
        x = torch.relu(x)
        x = torch.permute(self.maxpool(torch.permute(x, (0, 2, 1))), (0, 2, 1))
        x = self.linear2(x)
        x = torch.relu(x)
        x = torch.permute(self.maxpool(torch.permute(x, (0, 2, 1))), (0, 2, 1))
        x = self.linear3(x)
        x = torch.relu(x)
        x = torch.permute(self.maxpool(torch.permute(x, (0, 2, 1))), (0, 2, 1))

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        target = torch.rand(batch_size, 30, 512).to(device)
        x = self.pos_encoder(x)
        x = self.transformer_decoder(target, x)

        x = self.linear4(x)
        x = torch.relu(x)
        x = self.linear5(x)
        x = torch.relu(x)
        x = self.linear6(x)
        x = torch.relu(x)

        x = x.view(x.shape[0], x.size(1) * x.size(2))
        x = self.fc(x)
        if self.task == 'classification':
            x = self.activation(x)
        return x


processed_dir = './processed/'
file_list = np.char.split(np.array(os.listdir(processed_dir)), '.')
case_list = []
for caseid in file_list:
    if (caseid[0].find('(') == -1):
        case_list.append(int(caseid[0]))

print('N of total cases: {}'.format(len(case_list)))

cases = {}
cases['train'], cases['valid+test'] = train_test_split(case_list, test_size=(valid_ratio+test_ratio), random_state=random_key)
cases['valid'], cases['test'] = train_test_split(cases['valid+test'], test_size=(
    test_ratio/(valid_ratio+test_ratio)),
    random_state=random_key)

for phase in ['train', 'valid', 'test']:
    print("- N of {} cases: {}".format(phase, len(cases[phase])))

for idx, caseid in enumerate(case_list):
    filename = processed_dir + str(caseid) + '.pkl'
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

        # bug
        for i in range(len(data['ple'])):
            data['ple'][i][np.isnan(data['ple'][i])] = 0

        data['caseid'] = [caseid] * len(data['abp'])
        raw_records = raw_records.append(pd.DataFrame(
            data)) if idx > 0 else pd.DataFrame(data)

raw_records = raw_records[(raw_records['map'] >= 20) & (
    raw_records['map'] <= 160)].reset_index(drop=True)

if task == 'classification':
    task_target = 'hypo'
    criterion = nn.BCELoss()
else:
    task_target = 'map'
    criterion = nn.MSELoss()

print('\n===== Task: {}, Seed: {} =====\n'.format(task, random_key))

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

c = 0
fig, ax = plt.subplots()

for invasive in [False, True]:
    for multi in [False, True]:
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
                                                        shuffle=True if phase == 'train' else False,
                                                        drop_last=True)
            epoch_loss[phase], epoch_auc[phase] = [], []

        model = TransformerModel(task, invasive, multi, hidden_dim=512,
                                 num_layers=4, num_heads=4, dim_feedforward=1024, batch_first=True)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        n_epochs = max_epoch

        best_loss, best_auc = 99999.99999, 0.0

        for epoch in range(n_epochs):

            target_stack, output_stack = {}, {}
            current_loss, current_auc = {}, {}
            for phase in ['train', 'valid', 'test']:
                target_stack[phase], output_stack[phase] = [], []
                current_loss[phase], current_auc[phase] = 0.0, 0.0

            model.train()
            for dnn_inputs, dnn_target in loader['train']:

                # bug
                for i in range(len(dnn_inputs)):
                    dnn_inputs[i] = torch.nan_to_num(dnn_inputs[i], nan=0.0)

                dnn_inputs, dnn_target = dnn_inputs.to(
                    device), dnn_target.to(device)
                optimizer.zero_grad()
                dnn_output = model(dnn_inputs)
                loss = criterion(dnn_output[:, 0], dnn_target)
                current_loss['train'] += loss.item()*dnn_inputs.size(0)
                loss.backward()
                optimizer.step()

            current_loss['train'] = current_loss['train'] / \
                len(loader['train'].dataset)
            epoch_loss['train'].append(current_loss['train'])

            for phase in ['valid', 'test']:
                model.eval()
                with torch.no_grad():
                    for dnn_inputs, dnn_target in loader[phase]:

                        # bug
                        for i in range(len(dnn_inputs)):
                          dnn_inputs[i] = torch.nan_to_num(
                              dnn_inputs[i], nan=0.0)

                        dnn_inputs, dnn_target = dnn_inputs.to(
                            device), dnn_target.to(device)
                        dnn_output = model(dnn_inputs)
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
                    method.fit(reg_output[phase], reg_label[phase])
                    current_auc[phase] = roc_auc_score(
                        reg_label[phase], method.predict_proba(reg_output[phase]).T[1])
                    epoch_auc[phase].append(current_auc[phase])

            label_invasive = 'invasive' if invasive == True else 'noninvasive'
            label_multi = 'multi' if multi == True else 'mono'
            label_pred_lag = str(int(pred_lag / 60)) + 'min'

            filename = task+'_'+label_invasive+'_'+label_multi+'_'+label_pred_lag

            best = ''
            if task == 'regression' and abs(current_loss['valid']) < abs(best_loss):
                best = '< ! >'
                last_saved_epoch = epoch
                best_loss = abs(current_loss['valid'])
                torch.save(model.state_dict(), pt_dir +
                           filename+'_epoch_best.pt')
            elif task == 'classification' and abs(current_auc['valid']) > abs(best_auc):
                best = '< ! >'
                last_saved_epoch = epoch
                best_auc = abs(current_auc['valid'])
                torch.save(model.state_dict(), pt_dir +
                           filename+'_epoch_best.pt')

            # torch.save(model.state_dict(), pt_dir+filename +
            #            '_epoch_{0:03d}.pt'.format(epoch+1))

            print('Epoch [{:3d}] Train loss: {:.4f} / Valid loss: {:.4f} (AUC: {:.4f}) / Test loss: {:.4f} (AUC: {:.4f}) {}'.format
                  (epoch+1,
                   current_loss['train'],
                   current_loss['valid'], current_auc['valid'],
                   current_loss['test'], current_auc['test'], best))

        best_model_path = f'{filename}_epoch_best.pt'
        model.load_state_dict(torch.load(pt_dir + best_model_path))
        model.eval()

        if task == "classification":
            y_true = np.array(ext['test']['hypo'])
            y_scores = []

            with torch.no_grad():
                for dnn_inputs, dnn_target in loader['test']:

                    # bug
                    for i in range(len(dnn_inputs)):
                      dnn_inputs[i] = torch.nan_to_num(dnn_inputs[i], nan=0.0)

                    dnn_inputs, dnn_target = dnn_inputs.to(
                        device), dnn_target.to(device)
                    dnn_output = model(dnn_inputs)

                    y_scores.extend(np.array(dnn_output.cpu().T[0]))
            y_true = y_true[:len(y_scores)]
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[c],
                     label='AUC: {:.3f}'.format(roc_auc))
        else:
            y_true = np.array(ext['test']['map'])
            y_pred = []

            with torch.no_grad():
                for dnn_inputs, dnn_target in loader['test']:

                    # bug
                    for i in range(len(dnn_inputs)):
                      dnn_inputs[i] = torch.nan_to_num(dnn_inputs[i], nan=0.0)

                    dnn_inputs, dnn_target = dnn_inputs.to(
                        device), dnn_target.to(device)
                    dnn_output = model(dnn_inputs)

                    y_pred.extend(np.array(dnn_output.cpu().T[0]))
            y_true = y_true[:len(y_pred)]
            errors = y_true - y_pred
            abs_errors = abs(y_true - y_pred)
            mae = np.mean(abs_errors)
            print('mae:', mae)
            ax.boxplot(errors, positions=[c], patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor='white', edgecolor='black'),
                       widths=0.5,
                       medianprops=dict(color=colors[c]))
        c += 1

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
