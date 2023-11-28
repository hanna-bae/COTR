import argparse
import os
import time 

import cv2
import numpy as np
import torch
import imageio
from scipy.spatial import distance_matrix 

from COTR.utils import utils, debug_utils
from COTR.models import build_model
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.inference.sparse_engine import SparseEngine, FasterSparseEngine
import torchvision.transforms as transforms 
import time
import torch 



utils.fix_randomness(0)
torch.set_grad_enabled(False)
def cuda_time() -> float:
     '''Returns the current time after CUDA synchronization'''
     torch.cuda.synchronize() 
     return time.perf_counter()

def main(opt):
    model = build_model(opt)
    model = model.cuda()
    weights = torch.load(opt.load_weights_path)['model_state_dict']
    utils.safe_load_weights(model, weights)
    model = model.eval()

    img_a = imageio.imread('./sample_data/imgs/21526113_4379776807.jpg')
    img_b = imageio.imread('./sample_data/imgs/21126421_4537535153.jpg')
    kp_a = np.load('./sample_data/21526113_4379776807.jpg.disk.kpts.npy')
    kp_b = np.load('./sample_data/21126421_4537535153.jpg.disk.kpts.npy')
    
    latencies = []
    if opt.faster_infer:
        engine = FasterSparseEngine(model, 32, mode='tile')
    else:
        engine = SparseEngine(model, 32, mode='tile')
    for i in range(1000):
            start = cuda_time() 
            corrs_a_b = engine.cotr_corr_multiscale(img_a, img_b, np.linspace(0.5, 0.0625, 4), 1, max_corrs=kp_a.shape[0], queries_a=kp_a, force=True)
            corrs_b_a = engine.cotr_corr_multiscale(img_b, img_a, np.linspace(0.5, 0.0625, 4), 1, max_corrs=kp_b.shape[0], queries_a=kp_b, force=True)
            if i >=500:
                 latencies.append((cuda_time() - start)*1000)
    latencies = sorted(latencies)
    drop = int(len(latencies) * 0.25)
    print (f"Latency: {np.mean(latencies[drop:-drop])}")
    


    
