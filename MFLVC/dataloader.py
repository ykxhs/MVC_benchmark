from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path+'BDGP.mat')
        data1 = data['X1'].astype(np.float32)
        data2 = data['X2'].astype(np.float32)
        labels = data['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        return [torch.from_numpy(self.x1), torch.from_numpy(self.x2)], torch.from_numpy(self.y)


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        x1 = self.data1
        x2 = self.data2
        x3 = self.data3

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], torch.from_numpy(self.labels)


class MNIST_USPS(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'MNIST_USPS.mat')
        self.Y = data['Y'].astype(np.int32).reshape(5000,)
        self.V1 = data['X1'].astype(np.float32)
        self.V2 = data['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

    def full_data(self):
        x1 = self.V1.reshape(-1,784)
        x2 = self.V2.reshape(-1,784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y



class Fashion(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Fashion.mat')
        self.Y = data['Y'].astype(np.int32).reshape(10000,)
        self.V1 = data['X1'].astype(np.float32)
        self.V2 = data['X2'].astype(np.float32)
        self.V3 = data['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

    def full_data(self):
        x1 = self.V1.reshape(-1,784)
        x2 = self.V2.reshape(-1,784)
        x3 = self.V3.reshape(-1,784)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        if self.view == 2:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2)], torch.from_numpy(self.labels)
        if self.view == 3:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2), torch.from_numpy(self.view5)], torch.from_numpy(self.labels)
        if self.view == 4:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2), torch.from_numpy(self.view5), torch.from_numpy(self.view3)], torch.from_numpy(self.labels)
        if self.view == 5:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2), torch.from_numpy(self.view5), torch.from_numpy(self.view4), torch.from_numpy(self.view3)], torch.from_numpy(self.labels)

class Hdigit():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][0][1].T.astype(np.float32)
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y

class Prokaryotic():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32).reshape(551,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 551
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        x3 = self.V3
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y

class YoutubeFace():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/YoutubeFace_sel_fea.mat')
        self.Y = data['Y'].astype(np.int32).reshape(101499,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
    def __len__(self):
        return 101499
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4),torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        x3 = self.V3
        x4 = self.V4
        x5 = self.V5
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4),torch.from_numpy(x5)], self.Y

class Synthetic3d():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/synthetic3d.mat')
        self.Y = data['Y'].astype(np.int32).reshape(600,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        x3 = self.V3
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y

class Cifar10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        x3 = self.V3
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y
class Cifar100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        x3 = self.V3
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y

def load_data(dataset,datapath = "./data/"):
    if dataset == "BDGP":
        dataset = BDGP(datapath)
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS(datapath)
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV(datapath)
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion(datapath)
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech(f'{datapath}/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech(f'{datapath}/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech(f'{datapath}/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 1984]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech(f'{datapath}/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "Hdigit":
        dataset = Hdigit(datapath)
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "Prokaryotic":
        dataset = Prokaryotic(datapath)
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4
    elif dataset == "YoutubeFace":
        dataset = YoutubeFace(datapath)
        dims = [64, 512, 64, 647, 838]
        view = 5
        data_size = 101499
        class_num = 31
    elif dataset == "Synthetic3d":
        dataset = Synthetic3d(datapath)
        dims = [3, 3, 3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == "Cifar10":
        dataset = Cifar10(datapath)
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
    elif dataset == "Cifar100":
        dataset = Cifar100(datapath)
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
