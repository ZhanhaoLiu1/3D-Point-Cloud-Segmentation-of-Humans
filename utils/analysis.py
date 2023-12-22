import re
import argparse
import matplotlib.pyplot as plt
# Read the log file

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, default='res', help='output folder')
parser.add_argument('--log', type=str, required=True, help="log path")
opt = parser.parse_args()

with open(opt.log, 'r') as file:
    log_data_segment = file.read()

def group_num(e,g,d):
    return int(int(g)+(int(e)*int(d)))

# Adjusting the regular expression pattern for accurate data extraction
pattern = r"\[(\d+): (\d+)/(\d+)\] train loss: ([\d.]+) accuracy: ([\d.]+)\n2023-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - Interim \[\d+: \d+/\d+\] test loss: ([\d.]+) accuracy: ([\d.]+)"

# Extracting data using the updated regex pattern
extracted_data = re.findall(pattern, log_data_segment, re.MULTILINE)
#print(extracted_data)

# Converting to a more readable format
data = [{"Group":group_num(epoch, group, data_points), "Train Loss": float(train_loss), "Test Loss": float(test_loss), 
                   "Train Accuracy": float(train_acc), "Test Accuracy": float(test_acc)} 
                  for epoch, group, data_points, train_loss, train_acc, test_loss, test_acc in extracted_data]

# Extracting data for plotting
groups = [d['Group'] for d in data]
train_losses = [d['Train Loss'] for d in data]
test_losses = [d['Test Loss'] for d in data]
train_accuracies = [d['Train Accuracy'] for d in data]
test_accuracies = [d['Test Accuracy'] for d in data]

# Plotting Loss with thicker lines
plt.figure()
plt.plot(groups, train_losses, label='Train Loss', color='red', linewidth=2)  # Thicker line for train loss
plt.plot(groups, test_losses, label='Test Loss', color='blue', linestyle='dashed', linewidth=2)  # Thicker line for test loss
plt.xlabel('Group')
plt.ylabel('Loss')
plt.title('Train and Test Loss per Group')
plt.legend()
plt.savefig(f'{opt.out}loss.png')
plt.show()

# Plotting Accuracy with thicker lines
plt.figure()
plt.plot(groups, train_accuracies, label='Train Accuracy', color='green', linewidth=2)  # Thicker line for train accuracy
plt.plot(groups, test_accuracies, label='Test Accuracy', color='orange', linestyle='dashed', linewidth=2)  # Thicker line for test accuracy
plt.xlabel('Group')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy per Group')
plt.legend()
plt.savefig(f'{opt.out}accuracy.png')
plt.show()