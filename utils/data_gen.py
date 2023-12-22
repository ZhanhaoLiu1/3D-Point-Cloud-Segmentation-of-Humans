
import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


def read_raw_data(file_path):
    """Reads the raw data file and returns coordinates and class indices."""
    with open(file_path, 'r') as file:
        data = file.readlines()

    points = []
    classes = []
    for line in data:
        values = line.strip().split(',')
        points.append(values[:3])  # X, Y, Z coordinates
        classes.append(int(values[-1]))  # Class index
    return np.array(points, dtype=np.float64), np.array(classes)

def save_pts_and_seg(points, classes, pts_path, seg_path):
    """Saves points to .pts file and class indices to .seg file."""
    np.savetxt(pts_path, points, fmt='%f')
    np.savetxt(seg_path, classes, fmt='%i')

def save_rgb(file_path, data, colors):
    class_indices = data[:, -1].astype(int)  # Assuming last column is the class index
    class_indices = class_indices % len(colors)  # Cycle through colors if index exceeds length
    rgb_values = colors[class_indices]
    np.savetxt(file_path, rgb_values, fmt='%f')

def draw_points(data, classes, file_path, colors):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assuming the last column of 'data' contains the segmentation labels

    # Map each label to a color in the 'colors' array
    # The modulo operation ensures cycling through colors if there are more labels than colors
    point_colors = colors[classes % len(colors)]

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=point_colors, s=1)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig(file_path, dpi=300)
    plt.close(fig)


def process_file(input_folder, output_folder, file_name, colors):
    """Processes a single file."""
    file_path = os.path.join(input_folder, file_name)
    base_name = file_name.replace('.txt', '')

    # Paths for different file types
    pts_path = os.path.join(output_folder, 'points', base_name + '.pts')
    seg_path = os.path.join(output_folder, 'points_label', base_name + '.seg')
    png_path = os.path.join(output_folder, 'points_rgb', base_name + '_rgb.txt')
    img_path = os.path.join(output_folder, 'seg_img', base_name + '.png')

    points, classes = read_raw_data(file_path)
    save_pts_and_seg(points, classes, pts_path, seg_path)
    save_rgb(png_path, points, colors)
    draw_points(points, classes, img_path, colors)

def process_folders_parallel(input_folder, output_folder, colors, num_workers=4):
    """Processes files in the input folder in parallel and outputs results in the output folder."""

    # Check if the input directory exists
    if not os.path.exists(input_folder):
        print(f"Input directory {input_folder} does not exist.")
        return
    subdirs = ['points', 'points_label', 'points_rgb', 'seg_img']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_folder, subdir), exist_ok=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Prepare a list of file processing tasks
        tasks = []
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.txt'):
                tasks.append(executor.submit(process_file, input_folder, output_folder, file_name, colors))

        # Wait for all tasks to complete
        for task in tasks:
            task.result()

def data_gen(file_path,label):
    # Load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Regular expression to extract the label before "_00000_normal_label"
    regex_pattern = r'(.+?)_00000_normal_label'

    # Extracting and storing the labels in a list
    extracted_labels = []
    for item in data:
        match = re.search(regex_pattern, item)
        if match:
            extracted_labels.append(match.group(1))

    for folder in extracted_labels:
        data_directory = 'data/train_data/'
        input_dir = os.path.join(data_directory, f'{folder}')
        output_dir = f'data/{label}/{folder}'
        # Define your colors array here
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

        process_folders_parallel(input_dir, output_dir, colors)


if __name__ == '__main__':
    train_path = 'data/train_data/train_list.json'
    test_path = 'dataset/test_list.json'

    data_gen(train_path,'train')
    data_gen(test_path,'test')

