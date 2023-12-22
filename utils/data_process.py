import os
import torch
import numpy as np
import torch.utils.data as data

class PointCloudDataset(data.Dataset):
    def __init__(self, roots, npoints=2500, split='train', data_augmentation=True):
        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.datapath = []

        for root in roots:
            points_dir = os.path.join(root, 'points')
            labels_dir = os.path.join(root, 'points_label')

            # Aggregate paths for .pts and .seg files from each root
            for file in os.listdir(points_dir):
                if file.endswith('.pts'):
                    pts_file = os.path.join(points_dir, file)
                    seg_file = os.path.join(labels_dir, file.replace('.pts', '.seg'))
                    self.datapath.append((pts_file, seg_file))

    def __getitem__(self, index):
        pts_file, seg_file = self.datapath[index]
        point_set = np.loadtxt(pts_file).astype(np.float32)
        seg = np.loadtxt(seg_file).astype(np.int64)

        # Randomly sample npoints
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        # Normalize
        point_set -= np.mean(point_set, axis=0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set /= dist

        # Data augmentation
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        return torch.from_numpy(point_set), torch.from_numpy(seg)

    def __len__(self):
        return len(self.datapath)
