import torch
from torch.utils.data import Dataset


class GpuDataset(Dataset):
    def __init__(self, main: torch.Tensor, appliance: torch.Tensor = None, gpu=True):
        super().__init__()

        self.main = main
        self.appliance = appliance

        self.labeled = appliance is not None
        self.in_gpu = gpu

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
            x, y = self.main[item], self.appliance[item]
            if not self.in_gpu:
                x = x.cuda()
                y = y.cuda()
            return x, y
        else:
            x = self.main[item]
            if not self.in_gpu:
                x = x.cuda()
            return x
