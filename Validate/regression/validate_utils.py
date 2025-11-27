import numpy as np
from torch.utils.data import DataLoader


def get_data(batch_size, dataset): 
    dataset = dataset
    
    dataset = DataLoader(dataset,
                               batch_size = batch_size,
                               shuffle=False,
                               num_workers=2,
                               pin_memory=True)

    return dataset


# Statistics 
# m: model output, o: observation
def IOA(m, o):
    ioa = 1 -(np.sum((o-m)**2))/(np.sum((np.abs(m-np.mean(o))+np.abs(o-np.mean(o)))**2))
    return float(ioa)

def R(m, o):
    r = np.sum((m - np.mean(m)) * (o - np.mean(o))) / (
        np.sqrt(np.sum((m - np.mean(m)) ** 2)) * np.sqrt(np.sum((o - np.mean(o)) ** 2))
    )
    return float(r)

def RMSE(m, o):
    rmse = (np.mean((o-m)**2))**0.5
    return float(rmse)

def NMB(m, o):
    nmb = np.sum(m-o) / np.sum(o) * 100
    return float(nmb)

def BIAS(m, o):
    bias = np.mean(m)-  np.mean(o)
    return float(bias)

def MAE(m, o):
    mae = np.mean(np.abs(m-o))
    return float(mae)