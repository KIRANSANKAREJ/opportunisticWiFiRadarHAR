import scipy.io
import numpy as np
import pandas as pd
import os, shutil
from tqdm import tqdm
from plot_pwr_spectrogram import PWR
import matplotlib.pyplot as plt
import pickle
from PIL import Image

cwd = os.getcwd()
pwr_path = os.path.join(cwd, "pwr")


def preprocess(pwr_path, visualize=False):
    selectedActivities = [
        'bodyrotate',
        'liedown',
        'noactivity',
        'sit',
        'stand',
        'standfromlie',
        'walk'
    ]
    dataDict = {
        'ch1': [],
        'ch2': [],
        'ch3': [],
        'activity': []
    }
    for file in tqdm(os.listdir(pwr_path), desc="Processing Files..."):
        file_path = os.path.join(pwr_path, file)
        pwr = scipy.io.loadmat(file_path)['PWR']

        columns = [item[0] for item in pwr[0]]
        df = pd.DataFrame(data=pwr[1:], columns=columns)

        df['timestamp'] = df['timestamp'].apply(lambda x: x[0][0][0])
        start_time = df.iloc[0]['timestamp']
        end_time = df.iloc[-1]['timestamp']

        pwr = PWR(file_path, start_time, end_time)
        ch1, ch2, ch3, activity = pwr.pwr_extract()

        # Convert to NumPy for indexing
        activity = np.array(activity)
        mask = np.isin(activity, selectedActivities)

        dataDict['ch1'].append(ch1[:, mask])
        dataDict['ch2'].append(ch2[:, mask])
        dataDict['ch3'].append(ch3[:, mask])
        dataDict['activity'].extend(activity[mask])
        
        if visualize:
            pwr.vis_specs(ch1, ch2, ch3, f"images/{file[:-4]}.png")
    
    for key in ['ch1', 'ch2', 'ch3']:
        dataDict[key] = np.concatenate([x.T for x in dataDict[key]], axis=0)
    
    dataDict['activity'] = np.array(dataDict['activity'])
    return dataDict


def segment_activity_blocks(dataDict):
    ch1 = dataDict['ch1']
    ch2 = dataDict['ch2']
    ch3 = dataDict['ch3']
    activity = dataDict['activity']
    blocks = []
    current_activity = activity[0]
    start_idx = 0

    for i in range(1, len(activity)):
        if activity[i] != current_activity:
            # End of current block
            block = {
                'activity': current_activity,
                'ch1': ch1[start_idx:i],
                'ch2': ch2[start_idx:i],
                'ch3': ch3[start_idx:i]
            }
            blocks.append(block)
            # Start a new block
            current_activity = activity[i]
            start_idx = i

    # Add the final block
    blocks.append({
        'activity': current_activity,
        'ch1': ch1[start_idx:],
        'ch2': ch2[start_idx:],
        'ch3': ch3[start_idx:]
    })

    return blocks


def blocks_to_images(blocks):
    image_blocks = []

    for block in tqdm(blocks, desc="Preprocessing for images..."):
        ch1_img = block['ch1'].T  # shape: (100, segment_length)
        ch2_img = block['ch2'].T
        ch3_img = block['ch3'].T

        # Stack to (3, 100, segment_length)
        image = np.stack([ch1_img, ch2_img, ch3_img], axis=0)

        image_blocks.append({
            'activity': block['activity'],
            'image': image  # shape: (3, 100, segment_length)
        })

    return image_blocks


def save_blocks_as_png(image_blocks, save_dir="pwr_images"):
    os.makedirs(save_dir, exist_ok=True)

    for i, block in tqdm(enumerate(image_blocks), desc="Creating Images..."):
        img = block['image']  # shape: (3, 100, W)

        # Normalize each channel separately to 0–255
        img_norm = []
        for channel in img:
            ch_min, ch_max = channel.min(), channel.max()
            ch_scaled = (channel - ch_min) / (ch_max - ch_min + 1e-8) * 255
            img_norm.append(ch_scaled.astype(np.uint8))

        # Stack normalized channels
        img_uint8 = np.stack(img_norm, axis=0)  # shape: (3, 100, W)

        # Convert to (H, W, 3) for saving
        img_uint8 = np.transpose(img_uint8, (1, 2, 0))  # (100, W, 3)

        # Save using PIL
        activity = block['activity'].replace(" ", "_")
        filename = f"{activity}_{i}.png"
        Image.fromarray(img_uint8).save(os.path.join(save_dir, filename))

    print(f"Saved {len(image_blocks)} images to {save_dir}")


def reorganize_images_by_label(src_dir="pwr_images", dst_dir="pwr_dataset"):
    os.makedirs(dst_dir, exist_ok=True)
    for img_file in tqdm(os.listdir(src_dir), desc="Reordering directories..."):
        if img_file.endswith(".png"):
            label = img_file.split("_")[0]
            label_dir = os.path.join(dst_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            shutil.move(os.path.join(src_dir, img_file), os.path.join(label_dir, img_file))
    os.rmdir(os.path.join(cwd, src_dir))


# Data loading
try:
    with open("activityDict.pkl", "rb") as file:
        dataDict = pickle.load(file)
    print("Loaded Pickle data...")
except:
    dataDict = preprocess(pwr_path)
    with open("activityDict.pkl", "wb") as file:
        pickle.dump(dataDict, file)
    print("Preprocessing completed, pickle file created...")

# Function calls
blocks = segment_activity_blocks(dataDict)
image_blocks = blocks_to_images(blocks)
save_blocks_as_png(image_blocks)
reorganize_images_by_label()


