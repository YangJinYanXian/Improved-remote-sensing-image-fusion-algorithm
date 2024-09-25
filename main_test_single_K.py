import torch.nn.modules as nn
import torch
import cv2
import numpy as np
from model_8band_K import tdnet
import h5py
import scipy.io as sio
import os
import torch.nn.functional as F

def get_edge(data):  # get high-frequency
    rs = np.zeros_like(data)
    if len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
    else:
        rs = data - cv2.boxFilter(data, -1, (5, 5))
    return rs

def load_gt_compared(file_path_gt,file_path_compared):
    data1 = sio.loadmat(file_path_gt)  # HxWxC
    data2 = sio.loadmat(file_path_compared)
    gt = torch.from_numpy(data1['gt']/2047)
    compared_data = torch.from_numpy(data2['output_dmdnet_newdata6']*2047)
    return gt, compared_data

def load_gt_compared_band4(file_path_gt,file_path_compared):
    data1 = sio.loadmat(file_path_gt)  # HxWxC
    data2 = sio.loadmat(file_path_compared)
    gt = torch.from_numpy(data1['gt']/2047)
    #compared_data = torch.from_numpy(data2['output_dmdnet_GF_data2']*2047)output_dmdnet_QB_data2
    compared_data = torch.from_numpy(data2['output_dmdet'] * 2047)
    return gt, compared_data

def load_set_h5(file_path):

    with h5py.File(file_path, 'r') as f: # HxWxC
        lms_dataset = f['lms']
        ms_dataset = f['ms']
        pan_dataset = f['pan']
        lms_array = lms_dataset[:]
        ms_array = ms_dataset[:]
        pan_array = pan_dataset[:]

        # tensor type:
    lms = torch.from_numpy(lms_array / 2047).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy(ms_array / 2047).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy(pan_array / 2047)   # HxW = 256x256

    return lms, ms, pan




def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy(data['ms'] / 2047).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy(data['pan'] / 2047)   # HxW = 256x256
    return lms, ms, pan
#ckpt = "dicussion_lr_wr/lr=0.4600.pth"#bdpn_mra11

#ckpt = "pretrained.pth"   # chose model

ckpt = "Weight_1104_new_K/300.pth"


def mytest(file_path):



    ########################################################################################################

    lms, ms, pan = load_set(file_path)
    #lms, ms, pan = load_set_h5(file_path)

    ########################################################################################################
    model = tdnet().cuda().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)
    #model = torch.load(ckpt)["model"]
    with torch.no_grad():

        x1, x2, x3 = lms, ms, pan   # read data: CxHxW (numpy type)
        print(x1.shape)
        x1 = x1.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        x2 = x2.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        x3 = x3.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW
        #pan_down = F.interpolate(x3, scale_factor=0.5).cuda()



        #################################  h5换顺序！！！！！！！！！
        #x2 = x2.permute(0, 2, 1, 3)



        ms_up_1_out, ms_up_2_out = model(x2, x3)  # tensor type: CxHxW
       # tensor type: CxHxW

        # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
        sr = torch.squeeze(ms_up_2_out).permute(1, 2, 0).cpu().detach().numpy()

        print(sr.shape)
        save_name = os.path.join("results", "result.mat")
        sio.savemat(save_name, {'result': sr})


if __name__ == '__main__':
    file_path = "test_data/new_data11.mat"
    #file_path = "test_data/reduced_examples/Test(HxWxC)_wv3_data4.mat"
    mytest(file_path)
