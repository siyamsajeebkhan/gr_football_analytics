import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import pickle
from PIL import Image
# Test loading the .npy files


dataset = list(pd.read_csv('dataset/frames.csv', header=None)[0])
samples, _ = train_test_split(dataset, test_size = 0.3, random_state = 42)
train = samples[0:1000]
# print(len(train), type(train))
# print(train[0:5])


data_path = 'dataset/npy_obs'
start = time.perf_counter()
for pkl_file_name in tqdm(train):
    pkl_folder = pkl_file_name.split('_')[0]
    pkl_file_path = os.path.join(data_path, pkl_folder, f"{pkl_file_name}.npy")
    try:
        obs = np.load(pkl_file_path)
    except ValueError:
        print(f"Could not open the file")
    # obs, action = np.array(obs_with_action[0:115]), obs_with_action[115]
    
        # total_files += 1
end = time.perf_counter()

print(f"Total time to read {len(train)} .npy files from the disk: {(end-start):0.4f} seconds")
print(f"Average time to read each .npy file from the disk: {(end-start)/len(train):0.4f} seconds")

# ------------------------------------------------------------- #

# Test loading the pickle files

data_path = 'dataset/single_obs_pkl_files'
start = time.perf_counter()
for pkl_file_name in tqdm(train):
    pkl_folder = pkl_file_name.split('_')[0]
    pkl_file_path = os.path.join(data_path, pkl_folder, f"{pkl_file_name}.p")
    try:
        with open(pkl_file_path, 'rb') as handle:
            obs_with_action = pickle.load(handle, encoding='bytes')
    except ValueError:
        print(f"Could not open the file")
end = time.perf_counter()

print(f"\nTotal time to read {len(train)} picklefiles from the disk: {(end-start):0.4f} seconds")
print(f"Average time to read each picklefile from the disk: {(end-start)/len(train):0.4f} seconds")

# --------------------------------------------------------------- #

# Compare it to image loading time for 1000 images from CIFAR 10 dataset
img_data_path = 'dataset/cifar-10-batches-py/data_batch_1'
# train = sorted(os.listdir(img_data_path))[0]
start = time.perf_counter()
try:
    with open(img_data_path, 'rb') as fo:
        img_data = pickle.load(fo, encoding='bytes')
except ValueError:
    print(f"Could not open the file")
end = time.perf_counter()

print(f"\nTotal time to read 1000 images from the disk: {(end-start):0.4f} seconds")
# print(f"Average time to read each image file from the disk: {(end-start)/len(train):0.4f} seconds")


# -------------------------------------------------------------- #
# Test for imagenet instances
data_path = 'dataset/imagenet/train'
samples, _ = train_test_split(sorted(os.listdir(data_path)), test_size=0.3, random_state=42)
train = samples[0:1000]

start = time.perf_counter()
for img_name in tqdm(train):
    img_file_path = os.path.join(data_path, img_name)
    try:
        img = Image.open(img_file_path)
        # img.rotate(45).show()
    except ValueError:
        print(f"Could not open the file")
end = time.perf_counter()

print(f"\nTotal time to read {len(train)} images from the disk: {(end-start):0.4f} seconds")
print(f"Average time to read each image file from the disk: {(end-start)/len(train):0.4f} seconds")