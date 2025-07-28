import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch.nn.functional as F
class AIReadiDataset(Dataset):
    def __init__(self, df, img_type, img_path_cfp, img_path_ir,split, cfp_transform=None, ir_transform=None, label="mhoccur_crt, Cataracts (in one or both eyes)", clinical_data=[]):
        #clinical data is a list of clinical cols
        super(AIReadiDataset, self).__init__()

        #df_split = df.reset_index(drop=True)
        self.split = split
        df_split = df[df["recommended_split"] == split].reset_index(drop=True)
        df_pos = df_split[df_split[label] == 1]
        df_neg = df_split[df_split[label] == 0]

        # 3) find minority‐class size
        n_min = min(len(df_pos), len(df_neg))

        # 4) sample each to that size
        df_balanced = pd.concat([
            df_pos.sample(n=n_min, random_state=42),
            df_neg.sample(n=n_min, random_state=42)
        ])

        # 5) shuffle and reset index
        self.df = df_split #df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        self.pat_list = list(self.df["patientID"])
        self.img_path_cfp = img_path_cfp
        self.img_path_ir = img_path_ir
        self.img_type = img_type
        self.cfp_transform = cfp_transform
        self.ir_transform = ir_transform
        self.label = label 
        self.clinical_data = clinical_data
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        dct = {}
        pathlist = []
        imglist = []

        if self.img_type == "ir" or self.img_type == "cfp and ir":
            path1 = f"{self.img_path_ir}//{self.df['rc_hr_ir_l'].iloc[i]}"
            path2 = f"{self.img_path_ir}//{self.df['rc_hr_ir_r'].iloc[i]}"
            path3 = f"{self.img_path_ir}//{self.df['mac_hr_ir_r'].iloc[i]}"
            path4 = f"{self.img_path_ir}//{self.df['mac_hr_ir_l'].iloc[i]}" 
            pathlist.extend([path1, path2, path3, path4]) 
            for path in pathlist: 
                img = Image.open(path)
                if self.ir_transform:
                    img = self.ir_transform(img)
                else:
                    img = transforms.ToTensor()(img)
                imglist.append(img)
            dct["ir"] = imglist
            
        pathlist = []
        imglist = []
        if self.img_type == "cfp" or self.img_type == "cfp and ir":
            path1 = f"{self.img_path_cfp}//{self.df['6x6_cfp_l'].iloc[i]}"
            path2 = f"{self.img_path_cfp}//{self.df['6x6_cfp_r'].iloc[i]}"
            path3 = f"{self.img_path_cfp}//{self.df['12x12_cfp_r'].iloc[i]}"
            path4 = f"{self.img_path_cfp}//{self.df['12x12_cfp_l'].iloc[i]}" 
            pathlist.extend([path1, path2, path3, path4]) #, path2, path3, path4]) 
            for path in pathlist: 
                img = Image.open(path)
                if self.cfp_transform:
                    img = self.cfp_transform(img)
                else:
                    img = transforms.ToTensor()(img)
                imglist.append(img)
            dct["cfp"] = imglist
            
    

        clinical_cols = ['bmi_vsorres, BMI', 'Urine Creatinine (mg/dL)','Glucose (mg/dL)', 'HbA1c (%)', 'INSULIN (ng/mL)', 'C-Peptide (ng/mL)',
        'waist_vsorres, Waist Circumference (cm)', 'whr_vsorres, Waist to Hip Ratio (WHR)', 'Urine Albumin (mg/dL)','age', 'mhoccur_hbp, High blood pressure',
        'mhoccur_lbp, Low blood pressure']

        if len(self.clinical_data) != 0: #no clinical data
            clinical_series = self.df.loc[i, self.clinical_data]  
            clinical_arr = clinical_series.to_numpy(dtype=np.int64)
            clinical_tensor = torch.tensor(clinical_arr, dtype=torch.long)
            dct["clinical_meta"] = clinical_tensor
        
        row = self.df.iloc[i]
        label_val = int(row[self.label])  
        dct["labels"] = torch.tensor(label_val, dtype=torch.long)
        
        return dct

    