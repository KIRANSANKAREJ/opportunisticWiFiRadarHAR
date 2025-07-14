import scipy.io
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from plot_pwr_spectrogram import PWR

cwd = os.getcwd()
pwr_path = os.path.join(cwd, "pwr")

def getData(path=pwr_path, force_reload=False):
    frames = []
    selectedActivities = [
        'bodyrotate',
        'liedown',
        'noactivity',
        'sit',
        'stand',
        'standfromlie',
        'walk'
    ]
    try:
        print("Checking for raw data...")
        raw_data = pd.read_pickle("pkl files/raw_data.pkl")
        if force_reload:
            raise Exception("Force Reload is True.")
        else:
            print("Raw data file found, loading data...")
    except: 
        print("Raw data not found...")
        for file in tqdm(os.listdir(path), desc="Processing files..."):
            filepath = os.path.join(pwr_path, file)
            pwr = scipy.io.loadmat(filepath)['PWR']

            # Flattening to extract column list
            columns = [item[0] for item in pwr[0]]

            # Making a DataFrame for readability
            df = pd.DataFrame(data=pwr[1:], columns=columns)

            # Column Access
            # print(f"Columns : \n{df.columns.values}")

            # Activity access
            """print(df.iloc[0]['activity'][0][0])"""

            df['exp_no'] = df['exp_no'].apply(lambda x: x[0])
            df['timestamp'] = df['timestamp'].apply(lambda x: x[0][0][0])
            df['activity'] = df['activity'].apply(lambda x: x[0][0][0])

            # Select only labels that are necessary
            df = df[df["activity"].isin(selectedActivities)].reset_index(drop=True)
            frames.append(df[[
                'exp_no',
                # 'timestamp',
                'activity',
                # 'person_id',
                # 'room_no',
                'PWR_ch1',
                'PWR_ch2',
                'PWR_ch3'
                ]]
                )
        print(f"\n\n\nColumns : {columns}\n\n\n")
        raw_data = pd.concat(frames)
        raw_data.to_pickle("pkl files/raw_data.pkl")
    finally:
        return raw_data

# data = getData()

# label_names = sorted(data["activity"].unique())
# print(f"\n\n\nActivities : {label_names}\n\n\n")

# first_row = data.iloc[0]
# print(f"\n\n\nFirst Row : \n{first_row}\n\n\n")

# PWR_ch1 = data.iloc[0]['PWR_ch1']
# print(f"\n\n\nFirst Channel PWR_ch1 : \n{PWR_ch1}\n\n\n")

# person_id = data.iloc[0]['person_id']
# print(f"\n\n\nPerson ID : {person_id}\n\n\n")

# room_no = data.iloc[0]['room_no']
# print(f"\n\n\nRoom ID : {room_no}\n\n\n")


# filepath = os.path.join(pwr_path, "PWR_exp_002.mat")
# pwr = scipy.io.loadmat(filepath)['PWR']

# # Flattening to extract column list
# columns = [item[0] for item in pwr[0]]

# # Making a DataFrame for readability
# df = pd.DataFrame(data=pwr[1:], columns=columns)
# print(df.iloc[0])