# Package import
from __future__ import print_function, division
from warnings import warn
from nilmtk.disaggregate import Disaggregator
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statistics import mean
import os
import time
import pickle
import random
import json
import torch
from torchsummary import summary
import torch.nn as nn
import torch.utils.data as tud
from torch.utils.data import Dataset
from torch.utils.data.dataset import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from nilmtk.disaggregate.dm.diffusion import ConditionalDiffusion
from nilmtk.disaggregate.dm.unet1d import UNet1D
from nilmtk.disaggregate.dm.uda_loss import coral, coral2

# Fix the random seed to ensure the reproducibility of the experiment
random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use cuda or not
USE_CUDA = torch.cuda.is_available()


def plot_transfer(main_src, app_src, main_dst, length=10000):
    start = 0
    end = start + main_src.shape[1]
    # end = start + min([length, main_src.shape[0], main_dst.shape[0]])
    xs = np.arange(0, end - start)

    plt.figure(figsize=(20, 8))
    plt.plot(xs, main_src[0].squeeze(-1), label='Main src')
    plt.plot(xs, main_dst[0].squeeze(-1), label='Main dst')
    # plt.plot(xs, app_src[start:end].squeeze(-1), label='App src')
    plt.legend()
    plt.show()


class GpuDataset(Dataset):
    def __init__(self, main: torch.Tensor, appliance: torch.Tensor = None, gpu=True):
        super().__init__()

        self.main = main
        self.appliance = appliance

        self.labeled = appliance is not None

        if gpu:
            self.main = self.main.cuda()
            if self.appliance is not None:
                self.appliance = self.appliance.cuda()

        # print(self.main.shape)
        # print(self.main[0][:20])
        # print(self.main[1][:20])

    def __len__(self):
        return self.main.size(0)

    def __getitem__(self, item):
        if self.labeled:
            return self.main[item], self.appliance[item]
        else:
            return self.main[item]


def initialize(layer):
    # Xavier_uniform will be applied to conv1d and dense layer, to be consistent with Keras and Tensorflow
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, val=0.0)


def train(appliance_name, model, mains, appliance, epochs, batch_size, pretrain,
          checkpoint_interval=None, train_patience=5, gpu_dataset=True):
    # Model configuration
    gpu_dataset = gpu_dataset and USE_CUDA
    if gpu_dataset:
        model = model.cuda()
    if not pretrain:
        model.apply(initialize)
    print("Use cuda", gpu_dataset)
    print("Main shape", mains.shape)
    print("App shape", appliance.shape)
    # summary(model, (1, mains.shape[1]))
    # Split the train and validation set
    train_mains, valid_mains, train_appliance, valid_appliance = train_test_split(mains, appliance,
                                                                                  test_size=.2,
                                                                                  random_state=random_seed)

    # Create optimizer, loss function, and dataloader
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    train_dataset = GpuDataset(torch.from_numpy(train_mains).float().permute(0, 2, 1),
                               torch.from_numpy(train_appliance).float().permute(0, 2, 1),
                               gpu=gpu_dataset)
    train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)

    valid_dataset = GpuDataset(torch.from_numpy(valid_mains).float().permute(0, 2, 1),
                               torch.from_numpy(valid_appliance).float().permute(0, 2, 1),
                               gpu=gpu_dataset)
    valid_loader = tud.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)

    # raise RuntimeError

    writer = SummaryWriter(comment='train_visual')
    patience, best_loss = 0, None

    for epoch in range(epochs):
        # Earlystopping
        if (patience == train_patience):
            print("val_loss did not improve after {} Epochs, thus Earlystopping is calling".format(
                train_patience))
            break
            # Train the model
        st = time.time()
        model.train()

        for i, (batch_mains, batch_appliance) in enumerate(train_loader):
            # if USE_CUDA:
            #     batch_mains = batch_mains.cuda()
            #     batch_appliance = batch_appliance.cuda()

            # DM special
            noise, noise_hat = model.train_step(batch_appliance, batch_mains)
            loss = loss_fn(noise, noise_hat)

            # original
            # batch_pred = model(batch_mains)
            # loss = loss_fn(batch_pred, batch_appliance)

            model.zero_grad()
            loss.backward()
            optimizer.step()
        ed = time.time()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            cnt, loss_sum = 0, 0
            for i, (batch_mains, batch_appliance) in enumerate(valid_loader):
                # if USE_CUDA:
                #     batch_mains = batch_mains.cuda()
                #     batch_appliance = batch_appliance.cuda()

                # DM special
                noise, noise_hat = model.train_step(batch_appliance, batch_mains)
                loss = loss_fn(noise, noise_hat)

                # batch_pred = model(batch_mains)
                # loss = loss_fn(batch_appliance, batch_pred)
                loss_sum += loss
                cnt += 1

        final_loss = loss_sum / cnt
        # Save best only
        if best_loss is None or final_loss < best_loss:
            best_loss = final_loss
            patience = 0
            net_state_dict = model.state_dict()
            path_state_dict = "./" + appliance_name + "_dm_coral_best_state_dict.pt"
            torch.save(net_state_dict, path_state_dict)
        else:
            patience = patience + 1

        print(
            "Epoch: {}, Valid_Loss: {}, Time consumption: {}s.".format(epoch, final_loss, ed - st))
        # For the visualization of training process
        for name, param in model.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, epoch)
            writer.add_histogram(name + '_data', param, epoch)
        writer.add_scalars("MSELoss", {"Valid": final_loss}, epoch)

        # Save checkpoint
        if (checkpoint_interval != None) and ((epoch + 1) % checkpoint_interval == 0):
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = "./" + appliance_name + "_dm_coral_checkpoint_{}_epoch.pt".format(epoch)
            torch.save(checkpoint, path_checkpoint)


def test(model, test_mains, batch_size=512, gpu_dataset=True):
    # Model test
    st = time.time()
    model.eval()
    # Create test dataset and dataloader
    batch_size = test_mains.shape[0] if batch_size > test_mains.shape[0] else batch_size
    # test_dataset = TensorDataset(torch.from_numpy(test_mains).float().permute(0, 2, 1))
    test_dataset = GpuDataset(torch.from_numpy(test_mains).float().permute(0, 2, 1),
                              gpu=gpu_dataset)
    test_loader = tud.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for i, batch_mains in enumerate(test_loader):
            batch_pred = model(batch_mains)
            # batch_pred = model(batch_mains[0])
            if i == 0:
                res = batch_pred
            else:
                res = torch.cat((res, batch_pred), dim=0)
    ed = time.time()
    print("Inference Time consumption: {}s.".format(ed - st))
    return res.detach().cpu().numpy()


class DMCoral(Disaggregator):
    def __init__(self, params):
        self.MODEL_NAME = "DMCoral"
        self.sequence_length = params.get('sequence_length', 480)
        self.overlapping_step = params.get('overlapping_step', 240)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size', 32)
        self.appliance_params = params.get('appliance_params', {})
        self.mains_mean = params.get('mains_mean', None)
        self.mains_std = params.get('mains_std', None)
        self.mains_dst_mean = params.get('mains_dst_mean', None)
        self.mains_dst_std = params.get('mains_dst_std', None)
        self.models = OrderedDict()
        self.test_only = params.get('test_only', False)
        # self.uda = params.get('uda', False)
        # self.lambda_coral = params.get('lambda_coral', 0.5)

    def partial_fit(self, train_main, train_appliances, pretrain=False, do_preprocessing=True,
                    pretrain_path="./dm_coral_pre_state_dict.pkl", **load_kwargs):
        # If no appliance wise parameters are specified, then they are computed from the data
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
        # print("Train", train_main)

        # Preprocess the data and bring it to a valid shape
        transfer_main = load_kwargs['dst_main']
        if do_preprocessing:
            print("Doing Preprocessing")
            train_main, train_appliances, transfer_main = \
                self.call_preprocessing(train_main, train_appliances, transfer_main,
                                        'train')

        train_main = pd.concat(train_main, axis=0).values
        train_main = train_main.reshape((-1, self.sequence_length, 1))
        transfer_main = pd.concat(transfer_main, axis=0).values
        transfer_main = transfer_main.reshape((-1, self.sequence_length, 1))

        print("Train main", train_main.shape)
        print("Transfer main", transfer_main.shape)

        train_main = self.coral_transform(train_main, transfer_main)
        print("Coral main", train_main.shape)

        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0).values
            app_df = app_df.reshape((-1, self.sequence_length, 1))
            new_train_appliances.append((app_name, app_df))
        train_appliances = new_train_appliances
        plot_transfer(train_main, train_appliances[0], transfer_main)

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for", appliance_name)
                backbone = UNet1D(dim=64,
                                  dim_mults=(1, 2, 4, 8),
                                  channels=2,
                                  out_dim=1,
                                  with_time_emb=True)
                self.models[appliance_name] = ConditionalDiffusion(backbone,
                                                                   1,
                                                                   1)
                # Load pretrain dict or not
                if pretrain is True:
                    self.models[appliance_name].load_state_dict(
                        torch.load("./" + appliance_name + "_dm_coral_pre_state_dict.pt"))

            model = self.models[appliance_name]
            if not self.test_only:
                train(appliance_name, model, train_main, power, self.n_epochs, self.batch_size,
                      pretrain, checkpoint_interval=3)
                # Model test will be based on the best model
            ckpt_name = "./" + appliance_name + "_dm_coral_best_state_dict.pt"
            # ckpt_name = "./fridge_dm_checkpoint_26_epoch.pt"
            print("Loaded from", ckpt_name)
            self.models[appliance_name].load_state_dict(
                torch.load(ckpt_name))

            # if self.uda:
            #     uda(appliance_name, model, train_main, power, transfer_main,
            #         self.n_epochs, self.batch_size, checkpoint_interval=3,
            #         lambda_coral=self.lambda_coral)
            #
            #     ckpt_name = "./" + appliance_name + "_dm_uda_best_state_dict.pt"
            #     # ckpt_name = "./fridge_dm_checkpoint_26_epoch.pt"
            #     print("Loaded from", ckpt_name)
            #     self.models[appliance_name].load_state_dict(
            #         torch.load(ckpt_name))

    def disaggregate_chunk(self, test_main_list, do_preprocessing=True):
        # Disaggregate (test process)
        print("Start dm disaggregating")
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None,
                                                     mains_lst_dst=None, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}

            for appliance in self.models:
                # Move the model to cpu, and then test it
                # model = self.models[appliance].to('cpu')
                print("Disaggregating", appliance)
                model = self.models[appliance]
                prediction = test(model, test_main)
                print(appliance, "prediction done, computing results")
                app_mean, app_std = self.appliance_params[appliance]['mean'], \
                    self.appliance_params[appliance]['std']
                prediction = self.denormalize_output(prediction, app_mean, app_std)
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                series = pd.Series(valid_predictions)
                disggregation_dict[appliance] = series
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def call_preprocessing(self, mains_lst_src, submeters_lst, mains_lst_dst, method):
        # Seq2Seq Version
        sequence_length = self.sequence_length
        overlap_step = self.overlapping_step
        if method == 'train':
            # Preprocess the main and appliance data, the parameter 'overlapping' will be set 'True'
            processed_mains = []
            # print(mains_lst)
            for mains in mains_lst_src:
                # print(mains)
                self.mains_mean, self.mains_std = mains.values.mean(), mains.values.std()
                mains = self.normalize_data(mains.values, sequence_length, mains.values.mean(),
                                            mains.values.std(), True, overlap_step)
                processed_mains.append(pd.DataFrame(mains))

            processed_mains_dst = []
            # print(mains_lst)
            for mains in mains_lst_dst:
                # print(mains)
                self.mains_dst_mean, self.mains_dst_std = mains.values.mean(), mains.values.std()
                mains = self.normalize_data(mains.values, sequence_length, mains.values.mean(),
                                            mains.values.std(), True, overlap_step)
                processed_mains_dst.append(pd.DataFrame(mains))

            tuples_of_appliances = []
            for (appliance_name, app_df_list) in submeters_lst:
                app_mean = self.appliance_params[appliance_name]['mean']
                app_std = self.appliance_params[appliance_name]['std']
                processed_app_dfs = []
                for app_df in app_df_list:
                    data = self.normalize_data(app_df.values, sequence_length, app_mean, app_std,
                                               True, overlap_step)
                    processed_app_dfs.append(pd.DataFrame(data))
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains, tuples_of_appliances, processed_mains_dst

        if method == 'test':
            # Preprocess the main data only, the parameter 'overlapping' will be set 'False'
            processed_mains = []
            for mains in mains_lst_src:
                mains = self.normalize_data(mains.values, sequence_length, mains.values.mean(),
                                            mains.values.std(), False)
                processed_mains.append(pd.DataFrame(mains))
            return processed_mains

    def coral_transform(self, src_main, dst_main):
        ds = src_main.squeeze(-1)
        dt = dst_main.squeeze(-1)
        cs = np.cov(ds.T) + np.eye(np.size(ds, 1))
        ct = np.cov(dt.T) + np.eye(np.size(dt, 1))

        ds = np.dot(ds, np.power(cs, -1/2))
        ds_star = np.dot(ds, np.power(ct, 1/2))
        return np.expand_dims(ds_star, -1)

    def normalize_data(self, data, sequence_length, mean, std,
                       overlapping=False, overlapping_step=1):
        # If you want to train the model,then overlapping = True will bring you a lot more training data; else overlapping = false to disaggregate the mains data
        n = sequence_length
        excess_entries = sequence_length - (data.size % sequence_length)
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst), axis=0)
        if overlapping:
            windowed_x = np.array(
                [arr[i:i + n] for i in range(0, len(arr) - n + 1, overlapping_step)])
        else:
            windowed_x = arr.reshape((-1, sequence_length))
        # z-score normalization: y = (x - mean)/std
        windowed_x = windowed_x - mean
        return (windowed_x / std).reshape((-1, sequence_length))

    def denormalize_output(self, data, mean, std):
        # x = y * std + mean
        return mean + data * std

    def set_appliance_params(self, train_appliances):
        # Set appliance mean and std to normalize the label(appliance data)
        for (app_name, df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std}})
