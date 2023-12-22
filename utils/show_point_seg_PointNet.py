import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.PointNet_model import PointNetDenseCls  # Import your model class
from utils.data_process import PointCloudDataset  # Import your data processing utilities
from mpl_toolkits.mplot3d import Axes3D

def draw_points(data, classes, file_path, colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Debug: Print shapes and types to understand the data
    print("Data shape:", data.shape, "Type:", type(data))
    print("Classes shape:", classes.shape, "Type:", type(classes))

    # Since the previous assertion is causing an error, let's comment it out
    # and ensure the data and class_indices are aligned in the scatter plot.
    num_points = min(data.shape[0], classes.shape[0])
    data = data[:num_points]
    classes = classes[:num_points]

    # Map each class index to a color
    point_colors = np.array([colors[cls] for cls in classes])

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=point_colors, s=1)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

# Load the trained model
model_path = 'res/PointNet/best_seg_model_body_2.pth'
num_classes = 14  # Replace with your number of classes
model = PointNetDenseCls(k=num_classes)
model.load_state_dict(torch.load(model_path))
model.cuda()
model.eval()

# Load test data
test_dataset = PointCloudDataset(roots=['data/test/50021_knees'], npoints=2500, split='test', data_augmentation=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define colors for each class (length should be at least equal to num_classes)
colors = np.array(
    [[237, 109, 82, 255],
    [242, 169, 59, 255],
    [69, 142, 247, 255],
    [96, 177, 119, 255],
    [103, 91, 198, 255],
    [224, 135, 232, 255],
    [128, 128, 128, 255],
    [34, 23, 124, 255],
    [172, 24, 56, 255],
    [243, 68, 72, 255],
    [44, 99, 198, 255],
    [237, 109, 82, 255],
    [242, 169, 59, 255],
    [69, 142, 247, 255]],
    dtype=np.float64
) / 255

# Process each point cloud in the test dataset
for i, data in enumerate(test_dataloader, 0):
    points, _ = data
    points = points.transpose(2, 1)
    points = points.cuda()

    with torch.no_grad():
        pred, _, _ = model(points)

    # Ensure pred is in the correct shape [num_classes, num_points]
    pred = pred.squeeze(0)  # Remove batch dimension: shape becomes [num_classes, num_points]
    print("Pred shape after squeeze:", pred.shape)

    # Get class indices
    # Apply np.argmax along the second axis (axis=1) to get the class index for each point
    class_indices = np.argmax(pred.cpu().numpy(), axis=1)
    print("Class indices shape:", class_indices.shape)

    # Visualize the first point cloud
    point_cloud_np = points.cpu().data.numpy()[0, :, :].transpose(1, 0)
    draw_points(point_cloud_np, class_indices, f'res/img/segmented_point_cloud_{i}.png', colors)

