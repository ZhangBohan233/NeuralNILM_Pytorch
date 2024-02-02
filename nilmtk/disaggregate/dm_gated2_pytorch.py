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
import nilmtk.utils as utils

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
    def __init__(self,
                 sequence_length,
                 main: torch.Tensor,
                 appliance: torch.Tensor = None,
                 gpu=True,
                 stride=1,
                 random_shift=False):
        super().__init__()

        self.sequence_length = sequence_length
        self.main = main
        self.appliance = appliance
        self.stride = stride
        self.random_shift = random_shift

        self.labeled = appliance is not None

        if gpu:
            self.main = self.main.cuda()
            if self.appliance is not None:
                self.appliance = self.appliance.cuda()

        self.last_index = self.main.shape[1] - self.sequence_length
        print(self.main.shape)

    def __len__(self):
        return self.last_index // self.stride

    def __getitem__(self, item):
        """
        :param item:
        """
        index = item * self.stride

        if self.stride > 1 and self.random_shift:
            half_stride = self.stride // 2
            rnd_index = random.randint(index - half_stride, index + half_stride - 1)
            index = max(0, min(self.last_index, rnd_index))

        end = index + self.sequence_length

        if self.labeled:
            return self.main[:, index:end], self.appliance[:, index:end]
        else:
            return self.main[:, index:end]

    def disable_rnd_shift(self):
        self.random_shift = False

    def enable_rnd_shift(self):
        self.random_shift = True


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


def shuffle_train_val(mains, appliances, val_rate=0.2, segments=16):
    train_mains = np.array(0)
    train_appliance = np.array(0)
    valid_mains = np.array(0)
    valid_appliance = np.array(0)

    each_seg_len = mains.shape[1] // segments
    for i in range(segments):
        index = i * each_seg_len
        end = mains.shape[1] if i == segments - 1 else index + each_seg_len

        # sometimes let train at front, sometimes not
        minor_sep = int(1 / val_rate)
        minor_len = each_seg_len // minor_sep
        minor_idx = random.randint(0, minor_sep - 1)
        val_b = index + minor_idx * minor_len
        val_e = val_b + minor_len
        if minor_idx == minor_sep - 1:
            val_e = end

        tm = np.concatenate([mains[:, index:val_b], mains[:, val_e:end]], axis=1)
        ta = np.concatenate([appliances[:, index:val_b], appliances[:, val_e:end]], axis=1)

        vm = mains[:, val_b:val_e]
        va = appliances[:, val_b:val_e]

        if i == 0:
            train_mains = tm.reshape(1, -1)
            train_appliance = ta.reshape(1, -1)
            valid_mains = vm.reshape(1, -1)
            valid_appliance = va.reshape(1, -1)
        else:
            train_mains = np.concatenate([train_mains, tm], axis=1)
            train_appliance = np.concatenate([train_appliance, ta], axis=1)
            valid_mains = np.concatenate([valid_mains, vm], axis=1)
            valid_appliance = np.concatenate([valid_appliance, va], axis=1)

    return train_mains, train_appliance, valid_mains, valid_appliance


def fine_tune(appliance_name, model,
              mains, appliance,
              mains_dst, appliance_dst,
              sequence_length,
              epochs, batch_size, pretrain,
              src_rate=1.0,
              checkpoint_interval=None, train_patience=5, lr=5e-6, gpu_dataset=False,
              stride=1, src_dataset="", freeze=True):
    # Model configuration
    gpu_dataset = gpu_dataset and USE_CUDA
    if USE_CUDA:
        model = model.cuda()
    if not pretrain:
        model.apply(initialize)
    print("Cuda avail", USE_CUDA, "GPU dataset", gpu_dataset)
    print("Main shape", mains.shape)
    print("App shape", appliance.shape)
    print("Main dst shape", mains_dst.shape)
    print("App dst shape", appliance_dst.shape)
    # summary(model, (1, mains.shape[1]))
    # Split the train and validation set

    base_name = "./" + appliance_name + "_dm_ft"
    if freeze:
        base_name += "_freeze"

    out_dir = base_name + "_training"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    mains = mains.reshape(1, -1)
    appliance = appliance.reshape(1, -1)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(mains[0], label='mains', linewidth=1)
    plt.plot(appliance[0], label='truth', linewidth=1)

    plt.show()

    train_mains, train_appliance, valid_mains, valid_appliance = \
        shuffle_train_val(mains_dst,
                          appliance_dst,
                          segments=1)

    if src_rate != 0:
        train_mains_src, train_appliance_src, valid_mains_src, valid_appliance_src = \
            shuffle_train_val(mains,
                              appliance,
                              segments=1)

        len_src = round(train_mains.shape[0] * src_rate)
        len_src_val = round(valid_mains.shape[0] * src_rate)
        train_mains = np.concatenate([train_mains, train_mains_src[:len_src]], axis=1)
        valid_mains = np.concatenate([valid_mains, valid_mains_src[:len_src_val]], axis=1)
        train_appliance = np.concatenate([train_appliance, train_appliance_src[:len_src]], axis=1)
        valid_appliance = np.concatenate([valid_appliance, valid_appliance_src[:len_src_val]],
                                         axis=1)

    # Create optimizer, loss function, and dataloader
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                             lr=lr)

    ckpt_name = "./" + appliance_name + "_" + src_dataset + "_dm_best_checkpoint.pt"
    checkpoint = torch.load(ckpt_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded state dict again from: " + ckpt_name)

    if freeze:
        model.freeze(True)

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                              lr=lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                              lr=lr*10)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    train_dataset = GpuDataset(sequence_length,
                               torch.from_numpy(train_mains).float(),
                               torch.from_numpy(train_appliance).float(),
                               stride=stride,
                               gpu=gpu_dataset)

    train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

    valid_dataset = GpuDataset(sequence_length,
                               torch.from_numpy(valid_mains).float(),
                               torch.from_numpy(valid_appliance).float(),
                               stride=stride,
                               gpu=gpu_dataset)

    valid_loader = tud.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

    # raise RuntimeError

    losses = []

    writer = SummaryWriter(comment='train_visual')
    patience, best_loss = 0, None

    n_batches = len(train_loader)

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

        train_loss_sum = 0
        train_cnt = 0

        for i, (batch_mains, batch_appliance) in enumerate(train_loader):
            print(f"Epoch {epoch} train batch {i}/{n_batches}", end="\r")
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

            train_loss_sum += loss.item()
            train_cnt += 1

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
                loss_sum += loss.item()
                cnt += 1

        # generate some samples
        with torch.no_grad():
            valid_dataset.disable_rnd_shift()
            sampler = DDIM_Sampler2(model.model)
            fwd = sequence_length // stride
            batch_n = min(len(valid_loader.dataset) // fwd, batch_size)
            batch_input = torch.zeros((batch_n, 1, sequence_length))
            batch_truth = torch.zeros((batch_n, 1, sequence_length))
            if USE_CUDA:
                batch_input = batch_input.cuda()
                batch_truth = batch_truth.cuda()
            for i in range(0, batch_n, 1):
                idx = i * fwd
                (batch_mains, batch_appliance) = valid_loader.dataset[idx]
                if USE_CUDA and not gpu_dataset:
                    batch_mains = batch_mains.cuda()
                    batch_appliance = batch_appliance.cuda()

                batch_input[i] = batch_mains
                batch_truth[i] = batch_appliance

            batch_output = model(batch_input, sampler=sampler)
            flat_input = batch_input.detach().cpu().reshape(-1).numpy()
            flat_truth = batch_truth.detach().cpu().reshape(-1).numpy()
            flat_output = batch_output.detach().cpu().reshape(-1).numpy()

            df = pd.DataFrame(
                {'mains': flat_input, 'truth': flat_truth, 'pred': flat_output})
            df.to_csv(f'{out_dir}/epoch{epoch}.csv', index=False)

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(flat_input, label='mains', linewidth=1)
            plt.plot(flat_truth, label='truth', linewidth=1)
            plt.plot(flat_output, label='pred', linewidth=1)
            # plt.xlabel('Index')
            # plt.ylabel('Value')
            plt.title(f'Sample of epoch {epoch}')
            plt.legend()
            plt.savefig(f'{out_dir}/epoch{epoch}.png')
            plt.show()

            valid_dataset.enable_rnd_shift()

        final_loss = loss_sum / cnt
        # Save best only
        if best_loss is None or final_loss < best_loss:
            best_loss = final_loss
            patience = 0
            net_state_dict = model.state_dict()
            path_state_dict = base_name + "_best_state_dict.pt"
            torch.save(net_state_dict, path_state_dict)

            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}

            path_checkpoint = base_name + "_best_checkpoint.pt"
            torch.save(checkpoint, path_checkpoint)
        else:
            patience = patience + 1

        losses.append((epoch, train_loss_sum / train_cnt, loss_sum / cnt, (ed - st)))
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

            path_checkpoint = base_name + "_checkpoint_{}_epoch.pt".format(epoch)
            torch.save(checkpoint, path_checkpoint)

        # write once at the end of each epoch
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        df = pd.DataFrame({'epoch': [lo[0] for lo in losses],
                           'train_loss': [lo[1] for lo in losses],
                           'val_loss': [lo[2] for lo in losses],
                           'time': [lo[3] for lo in losses]})
        df.to_csv(out_dir + '/loss.csv', index=False)


def train(appliance_name, model: ConditionalDiffusion,
          mains, appliance, sequence_length, epochs, batch_size, threshold, pretrain,
          checkpoint_interval=None, train_patience=5, lr=3e-5, gpu_dataset=True,
          stride=1, note="", filter_train=True):
    # Model configuration
    gpu_dataset = gpu_dataset and USE_CUDA
    if USE_CUDA:
        model = model.cuda()
    if not pretrain:
        model.apply(initialize)

    if filter_train:
        base_name = "./" + appliance_name + "_" + note + "_dm_g2"
    else:
        base_name = "./" + appliance_name + "_" + note + "_dm"

    out_dir = base_name + "_trainings"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    assert mains.shape[0] == appliance.shape[0]
    print("Cuda avail", USE_CUDA, "GPU dataset", gpu_dataset)
    print("Main shape", mains.shape)
    print("App shape", appliance.shape)
    # summary(model, (1, mains.shape[1]))

    if filter_train:
        real_mains = []
        real_appliances = []

        for i in range(0, mains.shape[0], sequence_length):
            main = mains[i:i + sequence_length]
            app = appliance[i:i + sequence_length]
            if (app > threshold).any():
                real_mains.append(main)
                real_appliances.append(app)

        mains = np.concatenate(real_mains, 0)
        appliance = np.concatenate(real_appliances, 0)
        print("Main shape valid", mains.shape)
        print("App shape valid", appliance.shape)

    mains = mains.reshape(1, -1)
    appliance = appliance.reshape(1, -1)

    train_mains, train_appliance, valid_mains, valid_appliance = \
        shuffle_train_val(mains,
                          appliance,
                          segments=16 if mains.shape[1] > 1000000 else 8)

    train_dataset = GpuDataset(sequence_length,
                               torch.from_numpy(train_mains).float(),
                               torch.from_numpy(train_appliance).float(),
                               stride=stride,
                               gpu=gpu_dataset)

    train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

    valid_dataset = GpuDataset(sequence_length,
                               torch.from_numpy(valid_mains).float(),
                               torch.from_numpy(valid_appliance).float(),
                               stride=stride,
                               gpu=gpu_dataset)

    valid_loader = tud.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

    n_batches = len(train_loader)
    n_val_batches = len(valid_loader)

    print("Seq len", sequence_length, "stride", stride, "Train len", len(train_dataset),
          "Train batches", len(train_loader), "Val batches", len(valid_loader))

    # Create optimizer, loss function, and dataloader
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                             lr=lr)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    # loss_fn = utils.lognorm

    # raise RuntimeError

    writer = SummaryWriter(comment='train_visual')
    patience, best_loss = 0, None

    # plt.subplots(4, 4)
    plt_x = np.arange(sequence_length)

    losses = []

    for epoch in range(epochs):
        # Earlystopping
        if (patience == train_patience):
            print("val_loss did not improve after {} Epochs, thus Earlystopping is calling".format(
                train_patience))
            break
            # Train the model

        st = time.time()
        model.train()

        train_loss_sum = 0
        train_cnt = 0

        for i, (batch_mains, batch_appliance) in enumerate(train_loader):
            print(f"Epoch {epoch} train batch {i}/{n_batches}", end="\r")
            if USE_CUDA and not gpu_dataset:
                batch_mains = batch_mains.cuda()
                batch_appliance = batch_appliance.cuda()

            if epoch == 0 and i == 0:
                for j in range(16):
                    plt.subplot(4, 4, j + 1)
                    plt.plot(plt_x, batch_mains[j].reshape(-1).detach().cpu().numpy())
                    plt.plot(plt_x, batch_appliance[j].reshape(-1).detach().cpu().numpy())
                plt.show()

            # DM special
            noise, noise_hat = model.train_step(batch_appliance, batch_mains)
            loss = loss_fn(noise, noise_hat)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_cnt += 1

        ed = time.time()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            cnt, loss_sum = 0, 0
            for i, (batch_mains, batch_appliance) in enumerate(valid_loader):
                print(f"Epoch {epoch} val batch {i}/{n_val_batches}", end="\r")
                if USE_CUDA and not gpu_dataset:
                    batch_mains = batch_mains.cuda()
                    batch_appliance = batch_appliance.cuda()

                # cal_app_state = (batch_appliance > threshold).float().cpu().numpy()
                # true_app_state = torch.from_numpy(cal_app_state).cuda().detach()

                # if (true_app_state > 0).any():
                # DM special
                noise, noise_hat = model.train_step(batch_appliance, batch_mains)
                loss = loss_fn(noise, noise_hat)

                # batch_pred = model(batch_mains)
                # loss = loss_fn(batch_appliance, batch_pred)
                loss_sum += loss.item()
                cnt += 1

        # generate some samples
        with torch.no_grad():
            valid_dataset.disable_rnd_shift()
            sampler = DDIM_Sampler2(model.model)
            fwd = sequence_length // stride
            batch_n = min(len(valid_loader.dataset) // fwd, batch_size)
            batch_input = torch.zeros((batch_n, 1, sequence_length))
            batch_truth = torch.zeros((batch_n, 1, sequence_length))
            if USE_CUDA:
                batch_input = batch_input.cuda()
                batch_truth = batch_truth.cuda()
            for i in range(0, batch_n, 1):
                idx = i * fwd
                (batch_mains, batch_appliance) = valid_loader.dataset[idx]
                if USE_CUDA and not gpu_dataset:
                    batch_mains = batch_mains.cuda()
                    batch_appliance = batch_appliance.cuda()

                batch_input[i] = batch_mains
                batch_truth[i] = batch_appliance

            batch_output = model(batch_input, sampler=sampler)
            flat_input = batch_input.detach().cpu().reshape(-1).numpy()
            flat_truth = batch_truth.detach().cpu().reshape(-1).numpy()
            flat_output = batch_output.detach().cpu().reshape(-1).numpy()

            df = pd.DataFrame({'mains': flat_input, 'truth': flat_truth, 'pred': flat_output})
            df.to_csv(f'{out_dir}/epoch{epoch}.csv', index=False)

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(flat_input, label='mains', linewidth=1)
            plt.plot(flat_truth, label='truth', linewidth=1)
            plt.plot(flat_output, label='pred', linewidth=1)
            # plt.xlabel('Index')
            # plt.ylabel('Value')
            plt.title(f'Sample of epoch {epoch}')
            plt.legend()
            plt.savefig(f'{out_dir}/epoch{epoch}.png')
            # plt.show()

            valid_dataset.enable_rnd_shift()

        final_loss = loss_sum / cnt
        # Save best only
        if best_loss is None or final_loss < best_loss:
            best_loss = final_loss
            patience = 0
            net_state_dict = model.state_dict()
            path_state_dict = base_name + "_best_state_dict.pt"
            torch.save(net_state_dict, path_state_dict)

            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}

            path_checkpoint = base_name + "_best_checkpoint.pt"
            torch.save(checkpoint, path_checkpoint)
        else:
            patience = patience + 1

        losses.append((epoch, train_loss_sum / train_cnt, loss_sum / cnt, (ed - st)))

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

            path_checkpoint = base_name + "_checkpoint_{}_epoch.pt".format(epoch)
            torch.save(checkpoint, path_checkpoint)

        # write once at the end of each epoch
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        df = pd.DataFrame({'epoch': [lo[0] for lo in losses],
                           'train_loss': [lo[1] for lo in losses],
                           'val_loss': [lo[2] for lo in losses],
                           'time': [lo[3] for lo in losses]})
        df.to_csv(out_dir + '/loss.csv', index=False)


def test(model: ConditionalDiffusion, sequence_length, test_mains, batch_size=512, gpu_dataset=False,
         pred_gate=None, plot_net=False):
    if USE_CUDA:
        model = model.cuda()
    # Model test
    st = time.time()
    model.eval()
    # Create test dataset and dataloader
    print("test main", test_mains.shape)
    batch_size = test_mains.shape[0] if batch_size > test_mains.shape[0] else batch_size
    # test_dataset = TensorDataset(torch.from_numpy(test_mains).float().permute(0, 2, 1))
    test_mains = test_mains.reshape(1, -1)
    test_dataset = GpuDataset(sequence_length,
                              torch.from_numpy(test_mains).float(),
                              gpu=gpu_dataset,
                              stride=sequence_length,
                              random_shift=False)
    test_loader = tud.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for i, batch_mains in enumerate(test_loader):
            if USE_CUDA and not gpu_dataset:
                batch_mains = batch_mains.cuda()

            batch_pred = model(batch_mains, plot=plot_net)
            plot_net = False

            if i == 0:
                res = batch_pred.detach().cpu()
            else:
                res = torch.cat((res, batch_pred.detach().cpu()), dim=0)

            if i == 0:
                for j in range(min(16, batch_mains.shape[0])):
                    plt.subplot(4, 4, j + 1)
                    plt.plot(batch_mains[j].reshape(-1).detach().cpu().numpy())
                    plt.plot(batch_pred[j].reshape(-1).detach().cpu().numpy())
                plt.show()

            # batch_index += batch_pred.shape[0]
    ed = time.time()
    print("Inference Time consumption: {}s.".format(ed - st))
    return res.numpy()


class DM_GATE2(Disaggregator):
    def __init__(self, params):
        self.MODEL_NAME = "DM_GATE2"
        self.sequence_length = params.get('sequence_length', 720)
        self.overlapping_step = params.get('overlapping_step', 10)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size', 32)
        self.appliance_params = params.get('appliance_params', {})
        self.appliance_params_transfer = params.get('appliance_params_transfer', {})
        self.mains_mean = params.get('mains_mean', None)
        self.mains_std = params.get('mains_std', None)
        self.mains_min, self.mains_max = 0, 0
        self.mains_dst_mean = params.get('mains_dst_mean', None)
        self.mains_dst_std = params.get('mains_dst_std', None)
        self.mains_dst_min, self.mains_dst_max = 0, 0
        self.models = OrderedDict()
        self.test_only = params.get('test_only', False)
        self.fine_tune = params.get('fine_tune', False)
        self.lr = params.get('lr', 5e-6 if self.fine_tune else 3e-5)
        self.sampler_class = params.get("sampler", "ddpm")
        self.src_rate = params.get("src_rate", 1.0)
        self.app_meta = params.get("app_meta", utils.APP_META['ukdale'])
        self.filter_train = params.get("filter_train", True)
        self.note = params.get("note", "")
        self.load_from = params.get("load_from", self.note)
        self.patience = params.get('patience', 5)
        self.scaler = "minmax"
        self.freeze = params.get('freeze', False)
        self.plot = params.get('plot', False)

    def partial_fit(self, train_main, train_appliances, pretrain=False, do_preprocessing=True,
                    pretrain_path="./dm_g2_pre_state_dict.pkl", **load_kwargs):
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
        train_main = train_main.reshape(-1, 1)
        # train_main = train_main.reshape((-1, self.sequence_length, 1))
        print("Train main", train_main.shape)

        if self.fine_tune:
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
            # app_df = app_df.reshape((-1, self.sequence_length + self.train_window_shift, 1))
            app_df = app_df.reshape(-1, 1)
            new_train_appliances.append((app_name, app_df))
        train_appliances = new_train_appliances

        plt.figure(figsize=(8, 4))
        plt.plot(train_main[0:self.sequence_length, 0].reshape(-1), label='Train main')
        for i in range(len(train_appliances)):
            plt.plot(train_appliances[i][1][0:self.sequence_length, 0].reshape(-1),
                     label='Train truth')
        # plt.plot(pred_overall[clf][i], label='')
        plt.title("Train")
        plt.legend()
        plt.show()

        for appliance_name, power in train_appliances:
            # threshold = (10.0 - self.appliance_params[appliance_name]['mean']) / \
            #             self.appliance_params[appliance_name]['std']
            on_thresh = self.app_meta[appliance_name]["on"]
            if self.scaler == "std":
                threshold = ((on_thresh - self.appliance_params[appliance_name]['mean']) /
                             self.appliance_params[appliance_name]['std'])
            else:
                app_max = self.appliance_params[appliance_name]['max']
                app_min = self.appliance_params[appliance_name]['min']
                threshold = (on_thresh - app_min) / (app_max - app_min)
            #             self.appliance_params[appliance_name]['std']
            # threshold = (self.app_meta[appliance_name]['on'] -
            #              self.appliance_params[appliance_name]['mean']) / \
            #             self.appliance_params[appliance_name]['std']

            if appliance_name not in self.models:
                print("First model training for", appliance_name)
                backbone = UNet1D(dim=64,
                                  sequence_length=self.sequence_length,
                                  dim_mults=(1, 2, 4, 8),
                                  channels=2,
                                  out_dim=1,
                                  with_time_emb=True)
                self.models[appliance_name] = \
                    ConditionalDiffusion(backbone,
                                         1,
                                         1,
                                         # beta_start=1e-4,
                                         # beta_end=0.02,
                                         # apply_input_t=self.scaler != "std"
                                         apply_input_t=False
                                         )
                # Load pretrain dict or not
                if pretrain is True:
                    self.models[appliance_name].load_state_dict(
                        torch.load(
                            "./" + appliance_name + "_" + self.note + "_dm_g2_pre_state_dict.pt"))

            model = self.models[appliance_name]
            if not self.test_only:
                train(appliance_name, model, train_main, power,
                      self.sequence_length,
                      self.n_epochs, self.batch_size,
                      threshold,
                      pretrain,
                      checkpoint_interval=1,
                      lr=self.lr,
                      stride=self.overlapping_step,
                      filter_train=self.filter_train,
                      note=self.note,
                      train_patience=self.patience)
                # Model test will be based on the best model
            # ckpt_name = "./" + appliance_name + "_dm_g2_best_state_dict.pt"
            if self.filter_train:
                ckpt_name = "./" + appliance_name + "_" + self.load_from + "_dm_g2_best_state_dict.pt"
            else:
                ckpt_name = "./" + appliance_name + "_" + self.load_from + "_dm_best_state_dict.pt"
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
                          self.sequence_length,
                          self.n_epochs, self.batch_size,
                          pretrain,
                          src_rate=self.src_rate,
                          checkpoint_interval=1,
                          train_patience=3,
                          lr=self.lr,
                          src_dataset=self.load_from,
                          freeze=self.freeze)

                if self.freeze:
                    ckpt_name = "./" + appliance_name + "_dm_ft_freeze_best_state_dict.pt"
                else:
                    ckpt_name = "./" + appliance_name + "_dm_ft_best_state_dict.pt"
                # ckpt_name = "./fridge_dm_checkpoint_26_epoch.pt"
                print("Loaded from", ckpt_name)
                self.models[appliance_name].load_state_dict(
                    torch.load(ckpt_name))

    def disaggregate_chunk(self, test_main_list, do_preprocessing=True, pred_gate=None):
        # Disaggregate (test process)
        print("Start dm disaggregating")
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None,
                                                     method='test')

        test_predictions = []
        for test_main in test_main_list:
            # test_main = test_main.values.reshape((-1, self.sequence_length, 1))
            test_main = test_main.values.reshape(-1, 1)
            disggregation_dict = {}

            for appliance in self.models:
                # Move the model to cpu, and then test it
                # model = self.models[appliance].to('cpu')
                print("Disaggregating", appliance)
                model: ConditionalDiffusion = self.models[appliance]

                if self.sampler_class == "ddim":
                    sampler = DDIM_Sampler2(model.model)
                    model.sampler = sampler

                prediction = test(model, self.sequence_length, test_main,
                                  batch_size=self.batch_size, plot_net=self.plot)
                print(appliance, "prediction done, computing results")
                if self.fine_tune:
                    if self.scaler == "std":
                        app_mean, app_std = self.appliance_params_transfer[appliance]['mean'], \
                            self.appliance_params_transfer[appliance]['std']
                        prediction = self.denormalize_output(prediction, app_mean, app_std)
                    else:
                        app_min, app_max = self.appliance_params_transfer[appliance]['min'], \
                            self.appliance_params_transfer[appliance]['max']
                        prediction = self.deminmax_output(prediction, app_min, app_max)
                else:
                    if self.scaler == "std":
                        app_mean, app_std = self.appliance_params[appliance]['mean'], \
                            self.appliance_params[appliance]['std']
                        prediction = self.denormalize_output(prediction, app_mean, app_std)
                    else:
                        app_min, app_max = self.appliance_params[appliance]['min'], \
                            self.appliance_params[appliance]['max']
                        prediction = self.deminmax_output(prediction, app_min, app_max)

                valid_predictions = prediction.flatten()
                # if pred_gate is not None:
                #     pred_gate = pred_gate.to_numpy().flatten()
                #     if valid_predictions.shape[0] > pred_gate.shape[0]:
                #         gate_valid = np.ones((valid_predictions.shape[0],))
                #         gate_valid[:pred_gate.shape[0]] = pred_gate
                #         pred_gate = gate_valid
                #
                #     plt.plot(valid_predictions, label=appliance)
                #     plt.plot(pred_gate * 1000, label="gate")
                #     plt.show()
                #
                #     valid_predictions *= pred_gate

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
                mains = mains.clip(upper=self.app_meta["mains"]["max"])
                self.mains_mean, self.mains_std = mains.values.mean(), mains.values.std()
                self.mains_min, self.mains_max = 0, mains.values.max()
                if self.scaler == "std":
                    # mains = self.normalize_data(mains.values,
                    #                             sequence_length + self.train_window_shift,
                    #                             mains.values.mean(),
                    #                             mains.values.std(), True, overlap_step)
                    mains = self.normalize_data_linear(mains.values,
                                                       sequence_length,
                                                       mains.values.mean(),
                                                       mains.values.std())
                else:
                    mains = self.min_max_data_linear(mains.values,
                                                     sequence_length,
                                                     0,
                                                     mains.values.max())
                processed_mains.append(pd.DataFrame(mains))

            tuples_of_appliances = []
            for (appliance_name, app_df_list) in submeters_lst:
                app_mean = self.appliance_params[appliance_name]['mean']
                app_std = self.appliance_params[appliance_name]['std']
                app_min = self.appliance_params[appliance_name]['min']
                app_max = self.appliance_params[appliance_name]['max']

                processed_app_dfs = []
                for app_df in app_df_list:
                    if self.scaler == "std":
                        # data = self.normalize_data(app_df.values,
                        #                            sequence_length,
                        #                            app_mean, app_std,
                        #                            True, overlap_step)
                        data = self.normalize_data_linear(app_df.values,
                                                          sequence_length,
                                                          app_mean, app_std)
                    else:
                        data = self.min_max_data_linear(app_df.values,
                                                        sequence_length,
                                                        app_min, app_max)
                    processed_app_dfs.append(pd.DataFrame(data))
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains, tuples_of_appliances

        if method == 'transfer':
            processed_mains = []
            # print(mains_lst)
            for mains in mains_lst:
                # print(mains)
                mains = mains.clip(upper=self.app_meta["mains"]["max"])
                self.mains_dst_mean, self.mains_dst_std = mains.values.mean(), mains.values.std()
                self.mains_dst_min, self.mains_dst_max = 0, mains.values.max()
                if self.scaler == "std":
                    mains = self.normalize_data_linear(mains.values,
                                                       sequence_length,
                                                       mains.values.mean(),
                                                       mains.values.std())
                else:
                    mains = self.min_max_data_linear(mains.values,
                                                     sequence_length,
                                                     0,
                                                     mains.values.max())
                processed_mains.append(pd.DataFrame(mains))

            tuples_of_appliances = []
            for (appliance_name, app_df_list) in submeters_lst:
                app_mean = self.appliance_params_transfer[appliance_name]['mean']
                app_std = self.appliance_params_transfer[appliance_name]['std']
                app_min = self.appliance_params_transfer[appliance_name]['min']
                app_max = self.appliance_params_transfer[appliance_name]['max']

                processed_app_dfs = []
                for app_df in app_df_list:
                    if self.scaler == "std":
                        data = self.normalize_data_linear(app_df.values,
                                                          sequence_length,
                                                          app_mean, app_std)
                    else:
                        data = self.min_max_data_linear(app_df.values,
                                                        sequence_length,
                                                        app_min, app_max)
                    processed_app_dfs.append(pd.DataFrame(data))
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains, tuples_of_appliances

        if method == 'test':
            # Preprocess the main data only, the parameter 'overlapping' will be set 'False'
            processed_mains = []
            for mains in mains_lst:
                mains = mains.clip(upper=self.app_meta["mains"]["max"])
                if self.scaler == "std":
                    # mains = self.normalize_data(mains.values, sequence_length, mains.values.mean(),
                    #                             mains.values.std(), False)
                    mains = self.normalize_data_linear(mains.values, sequence_length,
                                                       mains.values.mean(),
                                                       mains.values.std())
                else:
                    mains = self.min_max_data_linear(mains.values, sequence_length,
                                                     0,
                                                     mains.values.max())
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

    def normalize_data_linear(self, data, sequence_length, mean, std):
        # If you want to train the model,then overlapping = True will bring you a lot more training data; else overlapping = false to disaggregate the mains data
        n = sequence_length
        excess_entries = sequence_length - (data.size % sequence_length)
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst), axis=0)
        windowed_x = arr
        # if overlapping:
        #     windowed_x = np.array(
        #         [arr[i:i + n] for i in range(0, len(arr) - n + 1, overlapping_step)])
        # else:
        #     windowed_x = arr.reshape((-1, sequence_length))
        # z-score normalization: y = (x - mean)/std
        windowed_x = windowed_x - mean
        return windowed_x / std

    def min_max_data_linear(self, data, sequence_length, min_, max_):
        # If you want to train the model,then overlapping = True will bring you a lot more training data; else overlapping = false to disaggregate the mains data
        # n = sequence_length
        excess_entries = sequence_length - (data.size % sequence_length)
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst), axis=0)
        windowed_x = arr.clip(max=max_)
        rng = max_ - min_
        return (windowed_x - min_) / rng

    def denormalize_output(self, data, mean, std):
        # x = y * std + mean
        return mean + data * std

    def deminmax_output(self, data, min_, max_):
        rng = max_ - min_
        return data * rng + min_

    def set_appliance_params(self, train_appliances):
        # Set appliance mean and std to normalize the label(appliance data)
        for (app_name, df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            l = l.clip(max=self.app_meta[app_name]['max'])
            app_mean = np.mean(l)
            app_std = np.std(l)
            self.appliance_params.update({app_name: {'mean': app_mean,
                                                     'std': app_std,
                                                     'min': 0,
                                                     'max': np.max(l)}})

    def set_transfer_appliance_params(self, transfer_appliances):
        # Set appliance mean and std to normalize the label(appliance data)
        for (app_name, df_list) in transfer_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            self.appliance_params_transfer.update({app_name: {'mean': app_mean,
                                                              'std': app_std,
                                                              'min': 0,
                                                              'max': np.max(l)}})
