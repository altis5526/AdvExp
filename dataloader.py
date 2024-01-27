import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import csv
import random
import os
import pandas as pd
import cv2
from torchvision import transforms


class Chexpert(Dataset):
    def __init__(self, dataset_csv, noise=False):
        with open(dataset_csv , newline='') as csvfile:
            data = list(csv.reader(csvfile))
            self.title = data[0]
            self.rows = data[1::]
        
        self.noise = noise

    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        img_dir = self.rows[idx][0]
        idx_num = idx
        full_img = Image.open(img_dir).convert('RGB')
        full_img = full_img.resize((256,256))
        full_img = np.transpose(full_img, (2, 0, 1))/255.0
        full_img = torch.from_numpy(full_img.copy()).float()
        h, w, _ = full_img.size()
        if self.noise==True:
            noise_matrix = torch.randn_like(full_img)
            full_img = full_img*0.75 + noise_matrix*0.25
        man = 0
        woman = 0
        if self.rows[idx][1] == 'Female':
            woman = float(1.)
        elif self.rows[idx][1] == 'Male':
            man = float(1.)
        gender = torch.tensor([man, woman])

        age = self.rows[idx][2]

        if self.rows[idx][4] == 'AP':
            AP_PA = 0
        elif self.rows[idx][4] == 'PA':
            AP_PA = 1
        elif self.rows[idx][4] == 'LL':
            AP_PA = 2
        else:
            AP_PA = 3
        
        race_vec = self.rows[idx][5:11]
        race = [float(i) for i in race_vec]
        race =  torch.tensor([race]).float()
        race = torch.squeeze(race, 0)

        diagnosis_vec = self.rows[idx][11:]
        diagnosis = [float(i) for i in diagnosis_vec]
        diagnosis = torch.tensor([diagnosis]).float()
        diagnosis = torch.squeeze(diagnosis, 0)
       
       
        patient_num = img_dir.split('/')[2]
        view = img_dir.split('/')[4]
        gender_pattern = torch.zeros(4)
        if self.rows[idx][1] == 'Male' and self.rows[idx][21] == '1':
            gender_pattern[0] = 1.
        elif self.rows[idx][1] == 'Male' and self.rows[idx][21] == '0':
            gender_pattern[1] = 1.
        elif self.rows[idx][1] == 'Female' and self.rows[idx][21] == '1':
            gender_pattern[2] = 1.
        elif self.rows[idx][1] == 'Female' and self.rows[idx][21] == '0':
            gender_pattern[3] = 1.
        
        output = {'full_img': full_img, 'gender': gender, 'age': age, 
                'AP/PA': AP_PA, 'race': race, 'diagnosis': diagnosis,
                'patient_num': patient_num, 'gender_pattern': gender_pattern, 'idx_num': idx_num, 'img_dir': img_dir,
                'view': view}
        return output





