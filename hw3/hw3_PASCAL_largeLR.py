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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = {x:PASCALDataset( os.path.join('PASCAL', x) ) for x in ['train', 'val']  }
data_loader = { x: torch.utils.data.DataLoader(dataset[x], batch_size = 4 if x is 'train' else 1, 
	shuffle = True, num_workers = 4, collate_fn = utils.collate_fn)  for x in ['train','val']}


model = torchvision.models.detection.fasterrcnn_resnet50_fpn( pretrained = True)
in_features = model.roi_heads.box_predictor.bbox_pred.in_features 
num_classes_PASCAL = 6 
num_classes_PennFudan = 2 

model.roi_heads.box_predictor = FastRCNNPredictor( in_features, num_classes_PASCAL)
model.to(device)

num_epochs = 10
optimizer = optim.SGD( model.parameters(), lr = 0.01, momentum = 0.9 )
scheduler = lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.9)

coco = get_coco_api_from_dataset( data_loader['val'].dataset )
iou_types = ["bbox"]

best_mAP = 0.0

for epoch in range(num_epochs):

	epoch_loss = 0.0 

	for phase in ['train', 'val']:
		if phase == 'train':
			model.train()
		else:
			model.eval()

			coco_evaluator = CocoEvaluator(coco, iou_types )



		for images, targets in data_loader[phase]:

			images = list( image.to(device) for image in images )
			if phase == 'train':
				targets = [ {k: v.to(device) for k,v in t.items() } for t in targets ]
			else:
				targets = [ {k: v for k,v in t.items() } for t in targets ]

			optimizer.zero_grad()

			

			with torch.set_grad_enabled( phase == 'train'):

				if phase == 'train':

					loss_dict = model(images, targets)
					

					loss = sum( value for value in loss_dict.values() )					

					epoch_loss += loss.item()

					loss.backward()
					optimizer.step()

				else:

					outputs = model(images)
					
					res = {target["image_id"].item(): output for target, output in 
					zip(targets, outputs) }
					coco_evaluator.update(res)

		if phase == 'train':

			scheduler.step()
			print( epoch, epoch_loss )



		if phase == 'val':
			coco_evaluator.synchronize_between_processes()
			coco_evaluator.accumulate()
			coco_evaluator.summarize()
			mAP = coco_evaluator.coco_eval['bbox'].stats[0]
			print("mAP is %f"%mAP)

			if mAP > best_mAP:
				best_mAP = mAP 
				best_model_wts = copy.deepcopy(model.state_dict() )
				torch.save( best_model_wts, 'best_model_weights_LargerLR.pth')













		


			









