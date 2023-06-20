import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from helper_Penn import PennFudanDataset
import utils 


import copy
import torch.optim as optim 
from torch.optim import lr_scheduler 
import os
from PIL import Image


def plot_bbox( idx ):
    
    img_id = data[idx][0]
    mAP = data[idx][1]
    bounding_box = data[idx][2]
    labels = data[idx][3]
    num_bbox = bounding_box.shape[0]
    
    
    path = 'PennFudanPed_hw3/test/Images/FudanPed000' + str( img_id + 30 ) +'.png' 
    
    img = Image.open( path )
    
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

if __name__ == "__main__":
	dataset_test = PennFudanDataset( os.path.join('PennFudanPed_hw3', 'test') )
	data = np.load("data.npy")

	category = ['background','person']
	color = ['b', 'g','r', 'c','m','y','k','w']

	plot_bbox(1)