from __future__ import print_function
import argparse
import torch
import os
import json
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from utils.data_process import PointCloudDataset
from model import PointNetDenseCls
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input datapoint size')
parser.add_argument('--batchSize', type=int, default=32, help='input datapoint size')


opt = parser.parse_args()
print(opt)

test_base_path = 'data/test'
# Load and prepend base paths to test paths
with open('data/test.json', 'r') as json_file:
    test_relative_paths = json.load(json_file)
    test_roots = [os.path.join(test_base_path, rel_path) for rel_path in test_relative_paths]
test_dataset = PointCloudDataset(roots=test_roots, npoints=opt.num_points, split='test', data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.batchSize, shuffle=True)

# Load model
classifier = PointNetDenseCls(k=14)  # Replace 14 with the number of classes
classifier.load_state_dict(torch.load(opt.model))
classifier.cuda()
classifier.eval()

# Setup test data loader
# Assuming test_dataset is already created similar to your training script
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True)

# Change the loss function to CrossEntropyLoss
criterion = torch.nn.CrossEntropyLoss()


# Initialization for mIoU calculation
intersection_counts = np.zeros(14)  # Replace 14 with the number of classes
union_counts = np.zeros(14)

total_correct = 0
total_tested = 0
total_test_loss = 0
for i, data in enumerate(test_dataloader, 0):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()

    # Forward pass
    with torch.no_grad():
        pred, _, _ = classifier(points)

    # Reshape model output and target for segmentation
    pred = pred.view(-1, 14)  # Replace 14 with the number of classes
    target = target.view(-1)

    # Calculate loss
    loss = criterion(pred, target)

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    
    total_correct += correct.item()
    total_tested += float(opt.batchSize * 2500)
    total_test_loss += loss.item()

    # Convert to numpy arrays for intersection and union calculation
    pred_np = pred_choice.cpu().numpy()
    target_np = target.cpu().numpy()

    # Calculate intersection and union for each class
    for class_id in range(14):  # Replace 14 with the number of classes
        intersection = np.sum((pred_np == class_id) & (target_np == class_id))
        union = np.sum((pred_np == class_id) | (target_np == class_id))
        
        intersection_counts[class_id] += intersection
        union_counts[class_id] += union
        
# Calculate, log test loss and accuracy
avg_test_loss = total_test_loss / len(test_dataloader)
test_accuracy = total_correct / total_tested
print('[Epoch %d] Test Loss: %f, Accuracy: %f' % (epoch, avg_test_loss, test_accuracy))
        
# Calculate IoU for each class and mean IoU
ious = []
for class_id in range(14):  # Replace 14 with the number of classes
    iou = intersection_counts[class_id] / union_counts[class_id] if union_counts[class_id] > 0 else 1
    ious.append(iou)

mean_iou = np.mean(ious)
print('Mean IoU: %f' % mean_iou)