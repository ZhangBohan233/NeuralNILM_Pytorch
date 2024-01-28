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
from nilmtk.disaggregate.dm.ddim2 import DDIM_Sampler2
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


def find_by_name(list_of_tup, name):
    for tup in list_of_tup:
        if tup[0] == name:
            return tup
    return None


def initialize(layer):
    # Xavier_uniform will be applied to conv1d and dense layer, to be consistent with Keras and Tensorflow
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, val=0.0)


def sda(appliance_name, model, mains_src, appliance_src, mains_dst, appliance_dst, epochs,
        batch_size,
        checkpoint_interval=None, train_patience=5, gpu_dataset=True, lambda_coral=0.5):
    gpu_dataset = gpu_dataset and USE_CUDA
    if gpu_dataset:
        model = model.cuda()

    # (train_mains, valid_mains,
    #  train_appliance, valid_appliance,
    #  train_dst, valid_dst) = train_test_split(mains_src,
    #                                           appliance_src,
    #                                           mains_dst,
    #                                           test_size=.2,
    #                                           random_state=random_seed)

    # Create optimizer, loss function, and dataloader
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    reg_loss_fn = torch.nn.MSELoss(reduction='mean')
    domain_loss_fn = coral

    train_src_dataset = GpuDataset(torch.from_numpy(mains_src).float().permute(0, 2, 1),
                                   torch.from_numpy(appliance_src).float().permute(0, 2, 1),
                                   gpu=gpu_dataset)
    train_src_loader = tud.DataLoader(train_src_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=0,
                                      drop_last=True)

    # valid_src_dataset = GpuDataset(torch.from_numpy(valid_mains).float().permute(0, 2, 1),
    #                                torch.from_numpy(valid_appliance).float().permute(0, 2, 1),
    #                                gpu=gpu_dataset)
    # valid_src_loader = tud.DataLoader(valid_src_dataset, batch_size=batch_size, shuffle=True,
    #                                   num_workers=0,
    #                                   drop_last=True)

    train_dst_dataset = GpuDataset(torch.from_numpy(mains_dst).float().permute(0, 2, 1),
                                   torch.from_numpy(appliance_dst).float().permute(0, 2, 1),
                                   gpu=gpu_dataset)
    train_dst_loader = tud.DataLoader(train_dst_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=0,
                                      drop_last=True)

    writer = SummaryWriter(comment='train_visual')
    patience, best_loss = 0, None

    min_batches = min(len(train_src_loader), len(train_dst_loader))

    for epoch in range(epochs):
        # Earlystopping
        if (patience == train_patience):
            print("val_loss did not improve after {} Epochs, thus Earlystopping is calling".format(
                train_patience))
            break

        train_src_iter = iter(train_src_loader)
        train_dst_iter = iter(train_dst_loader)

        st = time.time()
        model.train()

        reg_loss_sum = 0
        domain_loss_sum = 0
        loss_sum = 0

        for _ in range(min_batches):
            # if USE_CUDA:
            #     batch_mains = batch_mains.cuda()
            #     batch_appliance = batch_appliance.cuda()

            batch_src_mains, batch_src_appliance = next(train_src_iter)
            batch_dst_mains, batch_dst_appliance = next(train_dst_iter)

            # DM special
            # noise, noise_hat = model.train_step(batch_src_appliance, batch_src_mains)
            # noise_dst, noise_hat_dst = model.train_forward(batch_src_appliance, batch_dst_mains)
            b, c, length = batch_src_mains.shape
            device = next(model.model.parameters()).device
            t = torch.randint(0, model.forward_process.num_timesteps, (b,), device=device).long()
            # x_t = torch.randn([b, model.generated_channels, length], device=device)

            noise, noise_hat_src = model.train_step_fixed(batch_src_appliance,
                                                          batch_src_mains,
                                                          t)
            _, noise_hat_dst = model.train_step_fixed(batch_dst_appliance,
                                                      batch_dst_mains,
                                                      t,
                                                      predefined_noise=noise)

            reg_loss = reg_loss_fn(noise, noise_hat_src)
            domain_loss = domain_loss_fn(noise_hat_src, noise_hat_dst)

            reg_loss_sum += reg_loss.item()
            domain_loss_sum += domain_loss.item()

            loss = reg_loss + lambda_coral * domain_loss

            loss_sum += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()
        ed = time.time()

        # Cannot perform validation since we do not have appliance data from target domain
        final_loss_reg = reg_loss_sum / min_batches
        final_loss_domain = domain_loss_sum / min_batches
        final_loss = loss_sum / min_batches

        # # Evaluate the model
        # model.eval()
        # with torch.no_grad():
        #     cnt, loss_sum = 0, 0
        #     for i, (batch_mains, batch_appliance) in enumerate(valid_loader):
        #         # if USE_CUDA:
        #         #     batch_mains = batch_mains.cuda()
        #         #     batch_appliance = batch_appliance.cuda()
        #
        #         # DM special
        #         noise, noise_hat = model.train_forward(batch_appliance, batch_mains)
        #         loss = loss_fn(noise, noise_hat)
        #
        #         # batch_pred = model(batch_mains)
        #         # loss = loss_fn(batch_appliance, batch_pred)
        #         loss_sum += loss
        #         cnt += 1
        #
        # final_loss = loss_sum / cnt
        # Save best only
        if best_loss is None or final_loss < best_loss:
            best_loss = final_loss
            patience = 0
            net_state_dict = model.state_dict()
            path_state_dict = "./" + appliance_name + "_dm_sda_best_state_dict.pt"
            torch.save(net_state_dict, path_state_dict)
        else:
            patience = patience + 1

        print(
            "Epoch: {}, Train_Loss: {}, Reg_Loss: {}, Domain_Loss: {}, Time consumption: {}s."
            .format(epoch, final_loss, final_loss_reg, final_loss_domain, ed - st))
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
            path_checkpoint = "./" + appliance_name + "_dm_sda_checkpoint_{}_epoch.pt".format(epoch)
            torch.save(checkpoint, path_checkpoint)


def train(appliance_name, model, mains, appliance, epochs, batch_size, pretrain,
          checkpoint_interval=None, train_patience=5, lr=3e-5, gpu_dataset=False):
    # Model configuration
    gpu_dataset = gpu_dataset and USE_CUDA
    if USE_CUDA:
        model = model.cuda()
    if not pretrain:
        model.apply(initialize)
    print("Cuda avail", USE_CUDA, "GPU dataset", gpu_dataset)
    print("Main shape", mains.shape)
    print("App shape", appliance.shape)
    # summary(model, (1, mains.shape[1]))
    # Split the train and validation set
    train_mains, valid_mains, train_appliance, valid_appliance = train_test_split(mains, appliance,
                                                                                  test_size=.2,
                                                                                  random_state=random_seed)

    # Create optimizer, loss function, and dataloader
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                             lr=lr)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    if gpu_dataset:
        train_dataset = GpuDataset(torch.from_numpy(train_mains).float().permute(0, 2, 1),
                                   torch.from_numpy(train_appliance).float().permute(0, 2, 1),
                                   gpu=True)
    else:
        train_dataset = GpuDataset(torch.from_numpy(train_mains).float().permute(0, 2, 1),
                                   torch.from_numpy(train_appliance).float().permute(0, 2, 1),
                                   gpu=False)
        # train_dataset = TensorDataset(torch.from_numpy(train_mains).float().permute(0, 2, 1),
        #                               torch.from_numpy(train_appliance).float().permute(0, 2, 1))

    train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)

    if gpu_dataset:
        valid_dataset = GpuDataset(torch.from_numpy(valid_mains).float().permute(0, 2, 1),
                                   torch.from_numpy(valid_appliance).float().permute(0, 2, 1),
                                   gpu=True)
    else:
        valid_dataset = GpuDataset(torch.from_numpy(valid_mains).float().permute(0, 2, 1),
                                   torch.from_numpy(valid_appliance).float().permute(0, 2, 1),
                                   gpu=False)
        # valid_dataset = TensorDataset(torch.from_numpy(valid_mains).float().permute(0, 2, 1),
        #                               torch.from_numpy(valid_appliance).float().permute(0, 2, 1))

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
        print("Started epoch", epoch)

        for i, (batch_mains, batch_appliance) in enumerate(train_loader):
            if USE_CUDA and not gpu_dataset:
                batch_mains = batch_mains.cuda()
                batch_appliance = batch_appliance.cuda()

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
                if USE_CUDA and not gpu_dataset:
                    batch_mains = batch_mains.cuda()
                    batch_appliance = batch_appliance.cuda()

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
            path_state_dict = "./" + appliance_name + "_dm_best_state_dict.pt"
            torch.save(net_state_dict, path_state_dict)

            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}

            path_checkpoint = "./" + appliance_name + "_dm_best_checkpoint.pt".format(epoch)
            torch.save(checkpoint, path_checkpoint)
        else:
            patience = patience + 1

        print(
            "Epoch: {}, Valid_Loss: {}, Time consumption: {}s.".format(epoch, final_loss, ed - st))
        # For the visualization of training process

        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(name + '_grad', param.grad, epoch)
                writer.add_histogram(name + '_data', param, epoch)
        writer.add_scalars("MSELoss", {"Valid": final_loss}, epoch)

        # Save checkpoint
        if (checkpoint_interval != None) and ((epoch + 1) % checkpoint_interval == 0):
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}

            path_checkpoint = "./" + appliance_name + "_dm_checkpoint_{}_epoch.pt".format(epoch)
            torch.save(checkpoint, path_checkpoint)


def fine_tune(appliance_name, model,
              mains, appliance,
              mains_dst, appliance_dst,
              epochs, batch_size, pretrain,
              src_rate=1.0,
              checkpoint_interval=None, train_patience=5, lr=5e-6, gpu_dataset=False):
    # Model configuration
    gpu_dataset = gpu_dataset and USE_CUDA
    if USE_CUDA:
        model = model.cuda()
    if not pretrain:
        model.apply(initialize)
    print("Cuda avail", USE_CUDA, "GPU dataset", gpu_dataset)
    print("Main shape", mains.shape)
    print("App shape", appliance.shape)
    # summary(model, (1, mains.shape[1]))
    # Split the train and validation set
    train_mains, valid_mains, train_appliance, valid_appliance = train_test_split(mains_dst,
                                                                                  appliance_dst,
                                                                                  test_size=.2,
                                                                                  random_state=random_seed)

    if src_rate != 0:
        (train_mains_src, valid_mains_src,
         train_appliance_src, valid_appliance_src) = train_test_split(mains,
                                                                      appliance,
                                                                      test_size=.2,
                                                                      random_state=random_seed)
        len_src = round(train_mains.shape[0] * src_rate)
        len_src_val = round(valid_mains.shape[0] * src_rate)
        train_mains = np.concatenate([train_mains, train_mains_src[:len_src]], axis=0)
        valid_mains = np.concatenate([valid_mains, valid_mains_src[:len_src_val]], axis=0)
        train_appliance = np.concatenate([train_appliance, train_appliance_src[:len_src]], axis=0)
        valid_appliance = np.concatenate([valid_appliance, valid_appliance_src[:len_src_val]],
                                         axis=0)

    # Create optimizer, loss function, and dataloader
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                             lr=lr)

    checkpoint = torch.load("./" + appliance_name + "_dm_best_checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.freeze(True)

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                              lr=lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                              lr=lr*10)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    if gpu_dataset:
        train_dataset = GpuDataset(torch.from_numpy(train_mains).float().permute(0, 2, 1),
                                   torch.from_numpy(train_appliance).float().permute(0, 2, 1),
                                   gpu=True)
    else:
        train_dataset = GpuDataset(torch.from_numpy(train_mains).float().permute(0, 2, 1),
                                   torch.from_numpy(train_appliance).float().permute(0, 2, 1),
                                   gpu=False)
        # train_dataset = TensorDataset(torch.from_numpy(train_mains).float().permute(0, 2, 1),
        #                               torch.from_numpy(train_appliance).float().permute(0, 2, 1))

    train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)

    if gpu_dataset:
        valid_dataset = GpuDataset(torch.from_numpy(valid_mains).float().permute(0, 2, 1),
                                   torch.from_numpy(valid_appliance).float().permute(0, 2, 1),
                                   gpu=True)
    else:
        valid_dataset = GpuDataset(torch.from_numpy(valid_mains).float().permute(0, 2, 1),
                                   torch.from_numpy(valid_appliance).float().permute(0, 2, 1),
                                   gpu=False)
        # valid_dataset = TensorDataset(torch.from_numpy(valid_mains).float().permute(0, 2, 1),
        #                               torch.from_numpy(valid_appliance).float().permute(0, 2, 1))

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
        print("Started epoch", epoch)

        for i, (batch_mains, batch_appliance) in enumerate(train_loader):
            if USE_CUDA and not gpu_dataset:
                batch_mains = batch_mains.cuda()
                batch_appliance = batch_appliance.cuda()

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
                if USE_CUDA and not gpu_dataset:
                    batch_mains = batch_mains.cuda()
                    batch_appliance = batch_appliance.cuda()

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
            path_state_dict = "./" + appliance_name + "_dm_ft_best_state_dict.pt"
            torch.save(net_state_dict, path_state_dict)
        else:
            patience = patience + 1

        print(
            "Epoch: {}, Valid_Loss: {}, Time consumption: {}s.".format(epoch, final_loss, ed - st))
        # For the visualization of training process

        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(name + '_grad', param.grad, epoch)
                writer.add_histogram(name + '_data', param, epoch)
        writer.add_scalars("MSELoss", {"Valid": final_loss}, epoch)

        # Save checkpoint
        if (checkpoint_interval != None) and ((epoch + 1) % checkpoint_interval == 0):
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = "./" + appliance_name + "_dm_ft_checkpoint_{}_epoch.pt".format(
                epoch)
            torch.save(checkpoint, path_checkpoint)


def test(model, test_mains, batch_size=512, gpu_dataset=False):
    if USE_CUDA:
        model = model.cuda()
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
            if USE_CUDA and not gpu_dataset:
                batch_mains = batch_mains.cuda()
            batch_pred = model(batch_mains)
            # batch_pred = model(batch_mains[0])
            if i == 0:
                res = batch_pred.detach().cpu()
            else:
                res = torch.cat((res, batch_pred.detach().cpu()), dim=0)
    ed = time.time()
    print("Inference Time consumption: {}s.".format(ed - st))
    return res.numpy()


class DM_SDA(Disaggregator):
    def __init__(self, params):
        self.MODEL_NAME = "DM_SDA"
        self.sequence_length = params.get('sequence_length', 480)
        self.overlapping_step = params.get('overlapping_step', 240)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size', 32)
        self.appliance_params = params.get('appliance_params', {})
        self.appliance_params_transfer = params.get('appliance_params_transfer', {})
        self.mains_mean = params.get('mains_mean', None)
        self.mains_std = params.get('mains_std', None)
        self.mains_dst_mean = params.get('mains_dst_mean', None)
        self.mains_dst_std = params.get('mains_dst_std', None)
        self.models = OrderedDict()
        self.test_only = params.get('test_only', False)
        # self.uda = params.get('uda', False)
        self.sda = params.get('sda', False)
        self.lambda_coral = params.get('lambda_coral', 0.5)
        self.fine_tune = params.get('fine_tune', False)
        self.lr = params.get('lr', 5e-6 if self.fine_tune else 3e-5)
        self.sampler_class = params.get("sampler", "ddpm")
        self.src_rate = params.get("src_rate", 1.0)
        self.gpu_dataset = params.get("gpu_dataset", False)

    def partial_fit(self, train_main, train_appliances, pretrain=False, do_preprocessing=True,
                    pretrain_path="./dm_sda_pre_state_dict.pkl", **load_kwargs):
        # If no appliance wise parameters are specified, then they are computed from the data
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
        # print("Train", train_main)

        # Preprocess the data and bring it to a valid shape
        if do_preprocessing:
            print("Doing Preprocessing")
            train_main, train_appliances = self.call_preprocessing(train_main, train_appliances,
                                                                   'train')

        print("Train df", type(train_main))
        train_main = pd.concat(train_main, axis=0).values
        train_main = train_main.reshape((-1, self.sequence_length, 1))
        print("Train main", train_main.shape)

        if self.sda or self.fine_tune:
            if 'dst_main' in load_kwargs and 'dst_appliances' in load_kwargs:
                transfer_main = load_kwargs['dst_main']
                transfer_appliances = load_kwargs['dst_appliances']
                print("Transfer", type(transfer_main), type(transfer_appliances))
                if len(self.appliance_params_transfer) == 0:
                    self.set_transfer_appliance_params(transfer_appliances)
                if do_preprocessing:
                    transfer_main, transfer_app = self.call_preprocessing(transfer_main,
                                                                          transfer_appliances,
                                                                          'transfer')
                    # print("Train df", train_main)
                    transfer_main = pd.concat(transfer_main, axis=0).values
                    transfer_main = transfer_main.reshape((-1, self.sequence_length, 1))

                    new_transfer_appliances = []
                    for app_name, app_df in transfer_app:
                        app_df = pd.concat(app_df, axis=0).values
                        app_df = app_df.reshape((-1, self.sequence_length, 1))
                        new_transfer_appliances.append((app_name, app_df))
                    transfer_appliances = new_transfer_appliances

                    print("Transfer main", transfer_main.shape,
                          "transfer app", transfer_appliances[0][1].shape)

                    plt.figure(figsize=(8, 4))
                    plt.plot(transfer_main[0].reshape(-1), label='Transfer main')
                    for i in range(len(transfer_appliances)):
                        plt.plot(transfer_appliances[i][1][0].reshape(-1), label='Transfer truth')
                    # plt.plot(pred_overall[clf][i], label='')
                    plt.title("Transfer")
                    plt.legend()
                    plt.show()
            else:
                raise RuntimeError("If sda is set True, 'dst_main' must be provided")
        else:
            transfer_main = None
            transfer_appliances = None

        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0).values
            app_df = app_df.reshape((-1, self.sequence_length, 1))
            new_train_appliances.append((app_name, app_df))
        train_appliances = new_train_appliances

        plt.figure(figsize=(8, 4))
        plt.plot(train_main[0].reshape(-1), label='Train main')
        for i in range(len(train_appliances)):
            plt.plot(train_appliances[i][1][0].reshape(-1), label='Train truth')
        # plt.plot(pred_overall[clf][i], label='')
        plt.title("Train")
        plt.legend()
        plt.show()

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for", appliance_name)
                backbone = UNet1D(dim=64,
                                  sequence_length=self.sequence_length,
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
                        torch.load("./" + appliance_name + "_dm_pre_state_dict.pt"))

            model = self.models[appliance_name]
            if not self.test_only:
                train(appliance_name, model, train_main, power, self.n_epochs, self.batch_size,
                      pretrain, checkpoint_interval=3, lr=self.lr, gpu_dataset=self.gpu_dataset)
                # Model test will be based on the best model
            ckpt_name = "./" + appliance_name + "_dm_best_state_dict.pt"
            # ckpt_name = "./fridge_dm_checkpoint_26_epoch.pt"
            print("Loaded from", ckpt_name)
            self.models[appliance_name].load_state_dict(
                torch.load(ckpt_name))

            if self.sda:
                app_dst_power = find_by_name(transfer_appliances, appliance_name)[1]
                # print(type(power), type(app_dst_power))
                sda(appliance_name, model, train_main, power, transfer_main, app_dst_power,
                    self.n_epochs, self.batch_size, checkpoint_interval=3,
                    lambda_coral=self.lambda_coral)

                ckpt_name = "./" + appliance_name + "_dm_sda_best_state_dict.pt"
                # ckpt_name = "./fridge_dm_checkpoint_26_epoch.pt"
                print("Loaded from", ckpt_name)
                self.models[appliance_name].load_state_dict(
                    torch.load(ckpt_name))
            if self.fine_tune:
                print("Fine tuning")
                app_dst_power = find_by_name(transfer_appliances, appliance_name)[1]
                print(type(power), type(app_dst_power))
                # model.freeze(True)
                fine_tune(appliance_name, model,
                          train_main, power,
                          transfer_main, app_dst_power,
                          self.n_epochs, self.batch_size,
                          pretrain,
                          src_rate=self.src_rate,
                          checkpoint_interval=3,
                          train_patience=5,
                          lr=self.lr,
                          gpu_dataset=self.gpu_dataset)

                ckpt_name = "./" + appliance_name + "_dm_ft_best_state_dict.pt"
                # ckpt_name = "./fridge_dm_checkpoint_26_epoch.pt"
                print("Loaded from", ckpt_name)
                self.models[appliance_name].load_state_dict(
                    torch.load(ckpt_name))

    def disaggregate_chunk(self, test_main_list, do_preprocessing=True):
        # Disaggregate (test process)
        print("Start dm disaggregating")
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None,
                                                     method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}

            for appliance in self.models:
                # Move the model to cpu, and then test it
                # model = self.models[appliance].to('cpu')
                print("Disaggregating", appliance)
                model: ConditionalDiffusion = self.models[appliance]

                if self.sampler_class == "ddim":
                    sampler = DDIM_Sampler2(model.model)
                    model.sampler = sampler

                prediction = test(model, test_main, batch_size=self.batch_size)
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

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        # Seq2Seq Version
        sequence_length = self.sequence_length
        overlap_step = self.overlapping_step
        if method == 'train':
            # Preprocess the main and appliance data, the parameter 'overlapping' will be set 'True'
            processed_mains = []
            # print(mains_lst)
            for mains in mains_lst:
                # print(mains)
                self.mains_mean, self.mains_std = mains.values.mean(), mains.values.std()
                mains = self.normalize_data(mains.values, sequence_length, mains.values.mean(),
                                            mains.values.std(), True, overlap_step)
                processed_mains.append(pd.DataFrame(mains))

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

            return processed_mains, tuples_of_appliances

        if method == 'transfer':
            # Preprocess the main and appliance data, the parameter 'overlapping' will be set 'True'
            processed_mains = []
            # print(mains_lst)
            for mains in mains_lst:
                # print(mains)
                self.mains_dst_mean, self.mains_dst_std = mains.values.mean(), mains.values.std()
                mains = self.normalize_data(mains.values, sequence_length, mains.values.mean(),
                                            mains.values.std(), True, overlap_step)
                processed_mains.append(pd.DataFrame(mains))

            tuples_of_appliances = []
            for (appliance_name, app_df_list) in submeters_lst:
                # app_mean = self.appliance_params_transfer[appliance_name]['mean']
                # app_std = self.appliance_params_transfer[appliance_name]['std']
                app_mean = self.appliance_params[appliance_name]['mean']
                app_std = self.appliance_params[appliance_name]['std']
                processed_app_dfs = []
                for app_df in app_df_list:
                    data = self.normalize_data(app_df.values, sequence_length, app_mean, app_std,
                                               True, overlap_step)
                    processed_app_dfs.append(pd.DataFrame(data))
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains, tuples_of_appliances

        if method == 'test':
            # Preprocess the main data only, the parameter 'overlapping' will be set 'False'
            processed_mains = []
            for mains in mains_lst:
                mains = self.normalize_data(mains.values, sequence_length, mains.values.mean(),
                                            mains.values.std(), False)
                processed_mains.append(pd.DataFrame(mains))
            return processed_mains

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

    def set_transfer_appliance_params(self, transfer_appliances):
        # Set appliance mean and std to normalize the label(appliance data)
        for (app_name, df_list) in transfer_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            self.appliance_params_transfer.update({app_name: {'mean': app_mean, 'std': app_std}})
