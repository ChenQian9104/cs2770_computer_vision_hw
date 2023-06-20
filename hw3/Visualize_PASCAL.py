import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from helper import PASCALDataset 
import utils 

import copy
import torch.optim as optim 
from torch.optim import lr_scheduler 
import os
from PIL import Image

def plot_bbox( idx ):
    
    img_id = data[-idx][0]
    mAP = data[-idx][1]
    bounding_box = data[-idx][2]
    labels = data[-idx][3]
    num_bbox = bounding_box.shape[0]
    
    img = Image.open( image_id_to_path[ img_id ])
    
    fig, ax = plt.subplots(1, dpi = 600)
    ax.imshow(img)
    plt.title("figure %d_mAP: %.3f"%(idx,mAP) )
    
    for i in range(num_bbox):
        x1,y1,x2,y2 = bounding_box[i]
        rec = patches.Rectangle( (x1,y1), x2-x1, y2-y1, 
                               linewidth = 1, edgecolor = color[i%8], facecolor = 'none')
        ax.add_patch(rec)
        label = category[ labels[i] ]
        plt.text(( x1 + x2)/2, y1 -5,label,fontsize = 12, color =color[i%8] )
    
    plt.show()


if __name__ == '__main__':
	dataset_test = PASCALDataset( os.path.join('PASCAL', 'test') )
	data = np.load("data.npy")

	image_id_to_path = {}

	for image, target, filename in dataset_test:
    	image_id_to_path[ target["image_id"].item() ] = filename

    category = ['background','person', 'bicycle', 'car', 'motorcycle', 'airplane']
    color = ['b', 'g','r', 'c','m','y','k','w']

    plot_bbox(1)
