# aireadi_dataloader
A PyTorch Dataset for loading and processing [AI-READI](https://docs.aireadi.org/docs/2/about) dataset.


# Table of Contents

- [PatientDataset Class Overview](#patientdataset-class-overview)
- [Installation](#-installation)
- [Usage](#usage)

---

# PatientDataset Class Overview
The `PatientDataset` class is a PyTorch `Dataset` designed to handle patient data from the AI-READI dataset. The dataset includes 3D OCT, 3D OCTA, 2D fundus, and en face images, with 1,067 cases (and growing) and over 1,500 diverse tasks spanning segmentation, classification, and regression. This dataloader supports flexible configuration to filter, preprocess, and transform patient data for machine learning tasks in both image analysis and clinical prediction. This module provides MONAI-compatible dataset classes for loading OCT/OCTA/Photography imaging data, supporting both cache-based and on-demand access. It is tested with `monai.data.DataLoader` and follows MONAI-style dictionary-based outputs.

Each dataset returns a dictionary per sample with the following keys:

- `data_dict["frames"]`: the imaging data (2D or 3D tensor depending on settings)
- `data_dict["label"]`: the ground truth label associated with the sample

### Key Features:  
- **Concept ID-based Data Loading**: Loads data with respect to a specific clinical observation or measurement, which is coded as a clinical concept ID.  
- **Easy Image-Target Pairing**: Facilitates seamless loading of image and target pairs via the dataloader, making it convenient for model training.  
- **Seamless Integration with MONAI Pipeline**: Built on MONAI’s dataset class, allowing users to apply flexible preprocessing and transformations as part of their data pipeline.
- **Support for Multiple Imaging Modalities and Devices**: Handles images from various devices, models, imaging techniques (OCT, IR, CFP, OCTA, etc.), and anatomical regions. Users can specify which device or imaging type to load, making it easy to extract specific datasets via the API.  


# 🔧 Installation

This guide walks you through setting up a Python environment and installing the `aireadi-loader` package along with its dependencies.

1. **Create environment:**

    ```sh
    conda create -n aireadi python=3.10 -y
    conda activate aireadi
    ```

2. **Install `aireadi-loader`:**

    ```sh
    pip install aireadi-loader
    ```

3. **Install remaining dependencies and example training code:**

    ```sh
    git clone https://github.com/AI-READI/aireadi_loader.git
    cd aireadi_loader    
    pip install -r requirements.txt
    ```


### ⚠️ Platform & Environment Notes

- This dataloader currently supports **Linux** systems only.  
  (Installation has **not been tested** on Windows or macOS.)
- Make sure you have a **CUDA-compatible GPU** and the correct **NVIDIA drivers** installed for GPU acceleration.
- Tested with:
  - **Python 3.10 or 3.11**
  - **CUDA 11.8 or 12.1**


>🛠️ If you're using Windows, WSL2 with Ubuntu may work, but we recommend Linux for best compatibility and performance.



---

# Usage

This example demonstrates how to utilize the AIREADI dataloader for various input configurations.  
**Note:** The core focus of this release is the **dataloader itself**, not model training.  
However, we include example training scripts to showcase how the dataloader can be integrated into practical workflows.

### Step 1: Choose a Training Target

Decide which clinical variable (training target) to predict. Each target is encoded using a `concept_id`.  
You can find the corresponding `concept_id` for your variable of interest in the [AIREADI Data Domain Table](https://docs.aireadi.org/v2-dataDomainTable).  

For example, to predict HbA1c, search for `"Hba1c"` in the table — the `TARGET_CONCEPT_ID` will be `3004410`.

**Please make sure you understand your prediction target** — is it a classification (e.g., normal vs. abnormal, or multiclass) or a regression task (e.g., predicting a continuous value like HbA1c level)? This will affect how you configure nb_classes and choose the appropriate loss function.


### Step 2: Select Imaging Conditions

Choose the imaging condition(s) you want to train on, such as:

- Device model
- Imaging modality
- Anatomical region

Valid combinations can be found in the [AIREADI Dataloader Access Table](https://github.com/uw-biomedical-ml/AIREADI_dataloader/blob/main_merged_bug/dataloader_access_table.csv).

For an overview and available images, please refer to the following link:
- [OCT](https://docs.aireadi.org/docs/1/dataset/retinal-oct/)
- [OCTA](https://docs.aireadi.org/docs/1/dataset/retinal-octa/)
- [CFP/IR](https://docs.aireadi.org/docs/1/dataset/retinal-photography/)


### Step 3: Define Transformations

This dataloader uses a dictionary-based transformation pipeline compatible with [MONAI](https://monai.io/), where inputs are expected in the form of a dictionary (e.g., `{"frames": ..., "label": ...}`). MONAI's `Compose` and dictionary-style transforms (like `Resized`, `RandRotated`, etc.) are used to apply preprocessing consistently across multimodal or sequence-based data. By default, the `ToTensord` transform is applied after any user-defined transforms.

#### Example Transform (for Training)

```python
train_transform = Compose([
    Resized(
        keys=["frames"],
        spatial_size=(input_size, input_size),
        mode="bilinear",
    ),
    ScaleIntensityd(keys=["frames"]),
    RandRotated(
        keys=["frames"],
        range_x=(-0.17, 0.17),
        prob=0.5,
        mode="bilinear",
    ),
])
```

### Step 4: Build the Dataset

Use the `build_dataset()` function defined in `build_dataset.py` to construct your dataset.  

**Note:**
The AI-READI dataset comes with a predetermined train/validation/test split to support reproducible research and fair benchmarking. Below is the breakdown of the number of patients in each split by demographic categories such as sex, race, and diabetes status.

**Note:** `shuffle=True` has no effect when using `PatientFastAccessDataset`, which inherits from PyTorch's `IterableDataset`.


Example:

```python
from build_dataset import build_dataset
from monai.data import DataLoader

train_dataset = build_dataset(is_train=True, args=args)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

test_dataset = build_dataset(is_train=False, args=args)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
```



### Step 5: Train the Model

During training, data samples are accessed using:

```python
for batch in dataloader:
    images = batch["frames"].to(device)
    labels = batch["label"].to(device).long() // for classification
    ...
```

---
## Example: 2D, 3D, and Multimodal Training

- 2D Model Training: For slice-based inputs such as OCT/OCTA center slices, OCTA slabs, and fundus images (CFP/IR).
- 3D Model Training: For volume-based models that take full OCT or OCTA scans as input.
- Multimodal 2D Training: Combines multiple 2D image types (e.g., CFP + OCTA slab) for multimodal fusion.

Each script shows how to:

- Initialize the dataset and dataloader with selected modalitie(s)
- Iterate through singlemodal/multimodal batches

Refer to the example training scripts for full workflows. 
