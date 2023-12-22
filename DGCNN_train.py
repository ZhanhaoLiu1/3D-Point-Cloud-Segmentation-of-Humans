from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from DGCNN_model import DGCNN
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
from utils.data_process import PointCloudDataset
import logging
from datetime import datetime

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

# Setup logging
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_filename = f'res/training_log_{current_time}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

def log_info(message):
    print(message)
    logging.info(message)

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--out', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--train', type=str, required=True, help="dataset path")
parser.add_argument('--test', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
log_info(str(opt))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Seed setup
opt.manualSeed = random.randint(1, 10000)  # fix seed
log_info("Random Seed: " + str(opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

train_base_path = 'data/train'
test_base_path = 'data/test'

# Load and prepend base paths to train paths
with open(opt.train, 'r') as json_file:
    train_relative_paths = json.load(json_file)
    train_roots = [os.path.join(train_base_path, rel_path) for rel_path in train_relative_paths]

# Load and prepend base paths to test paths
with open(opt.test, 'r') as json_file:
    test_relative_paths = json.load(json_file)
    test_roots = [os.path.join(test_base_path, rel_path) for rel_path in test_relative_paths]

# Create datasets
train_dataset = PointCloudDataset(roots=train_roots, npoints=2500, split='train', data_augmentation=True)
test_dataset = PointCloudDataset(roots=test_roots, npoints=2500, split='test', data_augmentation=False)

# Create dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))


print(len(train_dataset), len(test_dataset))
log_info(f"train_size: {len(train_dataset)}, test_size: {len(test_dataset)}")
# Assuming all classes have the same number of segmentation classes, you can set a fixed number
# You need to replace 'num_classes' with the actual number of segmentation classes in your dataset
num_classes = 14
print('classes', num_classes)

try:
    os.makedirs(opt.out)
except OSError:
    pass

# Model setup
classifier = DGCNN(output_channels=14)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

# Initialize best test accuracy for model saving
best_test_accuracy = 0.0

num_batch = len(train_dataset) // opt.batchSize
log_info(str(opt.nepoch))

for epoch in range(opt.nepoch):
    scheduler.step()
    total_train_loss = 0
    total_correct = 0
    total_train_data = 0
    for i, data in enumerate(train_dataloader, 0):
        print(i)
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        print("Target max:", target.max())
        print("Target min:", target.min())
        if target.max() >= num_classes or target.min() < 0:
            print("Target values out of expected range:", target)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_train_loss += loss.item()
        total_correct += correct.item()
        total_train_data += float(opt.batchSize * 2500)
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))


        if i % 10 == 0:
            log_info('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))
            # Evaluation on a portion of the test dataset
            test_loss = 0
            correct = 0
            total = 0
            for j, test_data in enumerate(test_dataloader, 0):
                if j * opt.batchSize >= opt.batchSize * 10:  # Evaluate on the first 10 batches
                    break
                points, target = test_data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                with torch.no_grad():
                    pred, _, _ = classifier(points)
                    pred = pred.view(-1, num_classes)
                    target = target.view(-1, 1)[:, 0]
                    test_loss += F.nll_loss(pred, target).item()
                    pred_choice = pred.data.max(1)[1]
                    correct += pred_choice.eq(target.data).cpu().sum()
                    total += float(opt.batchSize * 2500)

            avg_test_loss = test_loss / 10
            test_accuracy = correct.item() / total
            log_info('Interim [%d: %d/%d] test loss: %f accuracy: %f' % (epoch, i, num_batch, avg_test_loss, test_accuracy))

    # Logging average training loss and accuracy
    avg_train_loss = total_train_loss / num_batch
    avg_train_accuracy = total_correct / total_train_data
    log_info('[Epoch %d] Average Training Loss: %f, Accuracy: %f' % (epoch, avg_train_loss, avg_train_accuracy))

    # Evaluation on test dataset and model saving
    total_correct = 0
    total_tested = 0
    total_test_loss = 0
    classifier = classifier.eval()
    for j, data in enumerate(test_dataloader, 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, _, _ = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        loss = F.nll_loss(pred, target)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_tested += float(opt.batchSize * 2500)
        total_test_loss += loss.item()

    # Calculate, log test loss and accuracy
    avg_test_loss = total_test_loss / len(test_dataloader)
    test_accuracy = total_correct / total_tested
    log_info('[Epoch %d] Test Loss: %f, Accuracy: %f' % (epoch, avg_test_loss, test_accuracy))

    # Save the model if it has the best test accuracy so far
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_model_path = '%s/best_seg_model_%s_%d.pth' % (opt.out, 'body', epoch)
        torch.save(classifier.state_dict(), best_model_path)
        log_info('Saved best model at epoch %d with accuracy %f' % (epoch, test_accuracy))


## benchmark mIOU
shape_ious = [] 
for i,data in tqdm(enumerate(test_dataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)#np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
mean_iou = np.mean(shape_ious)
log_info('Mean IoU: %f' % mean_iou)

