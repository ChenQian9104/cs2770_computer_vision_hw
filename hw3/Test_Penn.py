import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 
from pascal_dataset import PASCALDataset 
import utils 
from coco_utils import get_coco_api_from_dataset 
from coco_eval import CocoEvaluator 
import copy
import torch.optim as optim 
from torch.optim import lr_scheduler 
from PennFudanDataset import PennFudanDataset
import os
import numpy as np

dataset_test = PennFudanDataset( os.path.join('PennFudanPed_hw3', 'test') )
data_loader_test = torch.utils.data.DataLoader( dataset_test, batch_size = 1, 
	shuffle = True, num_workers = 4, collate_fn = utils.collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn( pretrained = True)
in_features = model.roi_heads.box_predictor.bbox_pred.in_features 
num_classes_PASCAL = 6 
num_classes_PennFudan = 2 

model.roi_heads.box_predictor = FastRCNNPredictor( in_features, num_classes_PennFudan)
model.load_state_dict( torch.load('best_model_weights_PennFudan_mid.pth', map_location = 'cpu') )


model.eval()

coco = get_coco_api_from_dataset( data_loader_test.dataset )
iou_types = ["bbox"]

result = []

for image, target in data_loader_test:

	
	target = [ {k: v for k,v in t.items() } for t in target ]




	output = model(image )


	res = {x["image_id"].item(): y for x, y in 
			zip(target, output) }


	coco_evaluator = CocoEvaluator(coco, iou_types )
	coco_evaluator.update(res)

	coco_evaluator.synchronize_between_processes()
	coco_evaluator.accumulate()
	coco_evaluator.summarize()
	mAP = coco_evaluator.coco_eval['bbox'].stats[0]	

	result.append( [ target[0]["image_id"].item(), mAP, 
		output[0]["boxes"].detach().numpy(), output[0]["labels"].detach().numpy() ] )


result.sort( key = lambda x: -x[1])

result = np.array( result )
np.save("data", result )