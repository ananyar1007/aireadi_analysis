import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class AIReadiDataset(Dataset):
    def __init__(self, df, img_type, img_path, split="train", transform=None, label="mhoccur_crt, Cataracts (in one or both eyes)"):
        super(AIReadiDataset, self).__init__()

        self.split = split
        self.df = df[df["recommended_split"] == split].reset_index(drop=True)
        self.pat_list = list(self.df["patientID"])
        self.img_path = img_path
        self.img_type = img_type
        self.transform = transform
        self.label = label 
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        dct = {}
        patient = self.pat_list[i]
        if self.img_type == "ir":
            path1 = f"{self.img_path}//{self.df['central_ir_l'].iloc[i]}"
            
        
        if self.img_type == "cfp":
            path1 = f"{self.img_path}//{self.df['central_cfp_l'].iloc[i]}"
        
        img = Image.open(path1)

        if self.transform:
            img1_t = self.transform(img)
        else:
            img1_t = transforms.to_tensor(img)
        #img2_t = self.transform(img2)
        dct["img"] = img1_t #, img2_t]  

        clinical_cols = ['bmi_vsorres, BMI', 'Urine Creatinine (mg/dL)','Glucose (mg/dL)', 'HbA1c (%)', 'INSULIN (ng/mL)', 'C-Peptide (ng/mL)',
       'waist_vsorres, Waist Circumference (cm)', 'whr_vsorres, Waist to Hip Ratio (WHR)', 'Urine Albumin (mg/dL)','age', 'mhoccur_hbp, High blood pressure',
       'mhoccur_lbp, Low blood pressure']
        dct["clinical_meta"] = self.df.loc[i, clinical_cols].tolist()


        condition_cols = [self.label]

        row = self.df.iloc[i]
        dct["labels"] = torch.tensor([row[condition_cols].values], dtype=torch.long)
        dct["labels"] = torch.reshape(dct["labels"], (1,))
        return dct

        