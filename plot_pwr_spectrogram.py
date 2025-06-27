# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:31:34 2021

@author:  Chong Tang

Script to plot PWR data from 3 surveillances channels "rx2", "rx3", "rx4".
These correspond to PWR Channel 1, PWR Channel 2 and PWR CHannel 3, respectively, in this example.
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datetime import datetime


class PWR:
    def __init__(self, dirc, start_time, end_time):
        # Set date time format
        FMT = '%H:%M:%S.%f'
        
        self.start_time = start_time
        self.end_time = end_time
        
        self.data = loadmat(dirc)['PWR']
        
        # timestamp and the start and end time of the whole data
        self.timestamp = self.data[1:, 1]
        
        data_start = datetime.strptime(self.timestamp[0].item(0)[0], FMT)
        data_end = datetime.strptime(self.timestamp[-1].item(0)[0], FMT)
        
        # Check start time and end time
        try:
            s = datetime.strptime(start_time, FMT)
            e = datetime.strptime(end_time, FMT)
            if e <= s:
               raise Exception("Incorrect start and end time, the end time should be greater than the start time.")
               
            if s < data_start or s > data_end or e > data_end:
               raise Exception("Incorrect start and end time, the start/end time out of the data range.")
        except ValueError:
            raise ValueError("Incorrect data format, start/end time format should be H-M-S.f")
        
        self.exp_id = self.data[1:, 0]

        
        self.activity = self.data[1:, 2]
        self.person_id = self.data[1:, 3]
        self.room_id = self.data[1:, 4]
        
        self.pwr_ch1 = self.data[1:, 5]
        self.pwr_ch2 = self.data[1:, 6]
        self.pwr_ch3 = self.data[1:, 7]
        
        # extract pwr data and visualize them
        # ch1, ch2, ch3 = self.pwr_extract()
        # self.vis_specs(ch1, ch2, ch3)
        
    def __str__(self):
        return "The PWR data is recorded with 3 channels.\nThe start time is: {}\nThe end time is: {}\nThe total number of frame is: {}".format(
            self.timestamp[0].item(0)[0], self.timestamp[-1].item(0)[0], self.timestamp.shape[0])
    
    def __repr__(self):
        return "PWR(data_directory, start_time, end_time)"
        
    def start_end_index(self):
        FMT = '%H:%M:%S.%f'
        START = True
        
        # initialize the time difference with a big value
        minum_dt = 60000
        start_index = 0
        end_index = 0
        
        for i in range(0, self.timestamp.shape[0]):
            tsp = self.timestamp[i].item(0)[0] # timestamp of frame i
            
            if START:# if start time index has not decided yet, do this first
                tdelta = abs((datetime.strptime(self.start_time, FMT) - datetime.strptime(tsp, FMT)).total_seconds())
            elif not START: # than find end time index
                tdelta = abs((datetime.strptime(self.end_time, FMT) - datetime.strptime(tsp, FMT)).total_seconds())
            
            # if the current tdelta is bigger than the previous minum value, 
            # that means the previous timestamp is what we want.
            if tdelta > minum_dt and START:
                START = False
                # re-initialize everything for finding end time index
                minum_dt = 60000
                tdelta = abs((datetime.strptime(self.end_time, FMT) - datetime.strptime(tsp, FMT)).total_seconds())
            elif tdelta > minum_dt and not START: # if end time index is found, stop the loop
                break
            
            # update the mimum time difference
            if tdelta < minum_dt:
                minum_dt = tdelta
                if START:
                    start_index = i
                elif not START:
                    end_index = i
            
        return start_index, end_index
    
    def pwr_extract(self, db=True):
        start_index, end_index = self.start_end_index()
        
        # concatenate pwr frames of the required period
        ch1 = np.zeros((100, end_index-start_index+1))
        ch2 = np.zeros((100, end_index-start_index+1))
        ch3 = np.zeros((100, end_index-start_index+1))
        
        for i in range(start_index, end_index+1):
            ch1[:, i-start_index] = self.pwr_ch1[i].reshape(-1)[50:150]
            ch2[:, i-start_index] = self.pwr_ch2[i].reshape(-1)[50:150]
            ch3[:, i-start_index] = self.pwr_ch3[i].reshape(-1)[50:150]
            
        # replace all 0 value with the minimum value for avoid the issue in log() calculation
        min_gt0 = np.ma.array(ch1, mask=ch1==0).min(0)
        ch1 = np.where(ch1 == 0, min_gt0, ch1)
        
        min_gt1 = np.ma.array(ch2, mask=ch2==0).min(0)
        ch2 = np.where(ch2 == 0, min_gt1, ch2)
        
        min_gt2 = np.ma.array(ch3, mask=ch3==0).min(0)
        ch3 = np.where(ch3 == 0, min_gt2, ch3)
        
        # want data in dB or not, suggest db=True
        if db:
            ch1 = 20*np.log10(ch1)
            ch2 = 20*np.log10(ch2)
            ch3 = 20*np.log10(ch3)

        activity = [
            self.activity[i].item(0)[0] 
            for i in range(start_index, end_index+1)
            ]
            
        return ch1, ch2, ch3, activity
    
    def vis_specs(self, ch1, ch2, ch3, filename):
        fig, axs = plt.subplots(3, 1)
        
        spec1 = axs[0].imshow(ch1, cmap='jet')
        axs[0].set_title('PWR Channel 1')
        spec1.set_clim(-55, -20)
        plt.colorbar(spec1, ax=axs[0])
        
        
        spec2 = axs[1].imshow(ch2, cmap='jet')
        axs[1].set_title('PWR Channel 2')
        spec2.set_clim(-55, -20)
        plt.colorbar(spec1, ax=axs[1])
        
        spec3 = axs[2].imshow(ch3, cmap='jet')
        axs[2].set_title('PWR Channel 2')
        spec3.set_clim(-40, -20)
        plt.colorbar(spec1, ax=axs[2])
        
        for ax in axs.flat:
            ax.set(xlabel='time', ylabel='velocity')
        plt.savefig(filename)
        plt.close()
            
if __name__ == '__main__':
    dirc = '../pwr/PWR_exp_018.mat'
    
    start_time = '18:04:44.553'
    end_time   = '18:08:11.553'
    
    pwr = PWR(dirc, start_time, end_time)

    