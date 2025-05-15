# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------
import os
import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter
from typing import List
import json

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

from aireadi_loader.dataloader import PatientDataset
from examples.build_dataset import build_transform_2d, build_transform_3d

from monai.data.dataset import Dataset
from monai.transforms import (
    Compose, Resized, RandRotated, RandFlipd, RandAffined, EnsureChannelFirstd, ScaleIntensityd,
    ToTensord, RandAdjustContrastd,RandGaussianNoised,RandGaussianSmoothd
)
from monai.transforms import apply_transform


class MultiModalDataset(Dataset):
    def __init__(self, modality_datasets: List[Dataset]):
        # Need to match visits across modalities by both patient ID and laterality
        self.modalities = modality_datasets

        # Extract all (pat_id, laterality) keys for each modality
        def get_eye_keys(d):
            keys = set()
            for visit in d.visits_dict.values():
                pat_id = visit["pat_id"]
                # find metadata with laterality info
                for imd in [s + "_metadata" for s in ["oct", "octa", "cfp", "ir", "faf"]]:
                    if imd in visit:
                        metadata = visit[imd][0]
                        break
                else:
                    raise KeyError("Missing a metadata key in visit")

                lat = metadata.get("laterality")
                if lat not in ("L", "R"):
                    raise ValueError(f"Unexpected laterality: {lat}")
                keys.add((pat_id, lat))
            return keys

        # Find common (pat_id, laterality) pairs
        common_eye_keys = set.intersection(*[get_eye_keys(d) for d in modality_datasets])
        self.eye_keys = sorted(common_eye_keys)

        # Index samples by matching the first visit_idx per (pat_id, laterality)
        self.samples = []
        for pat_id, lat in self.eye_keys:
            visit_indices = []
            for d in modality_datasets:
                for vidx, visit in d.visits_dict.items():
                    if visit["pat_id"] == pat_id:
                        for imd in [s + "_metadata" for s in ["oct", "octa", "cfp", "ir", "faf"]]:
                            if imd in visit:
                                metadata = visit[imd][0]
                                break
                        else:
                            raise KeyError("Missing a metadata key in visit")
                        if metadata.get("laterality") == lat:
                            visit_indices.append(vidx)
                            break
                else:
                    raise ValueError(f"No visit found for {(pat_id, lat)} in one of the modalities")
            self.samples.append(((pat_id, lat), visit_indices))
        print("Number of eyes in multimodal data: %d" % len(self.eye_keys))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        (pat_id, lat), visit_idxs = self.samples[index]

        frames = []
        label = None
        for i, (d, visit_idx) in enumerate(zip(self.modalities, visit_idxs)):
            visit = d.visits_dict[visit_idx]
            try:
                transformed_sample = apply_transform(d.transform, visit)
            except Exception as e:
                print(f"[ERROR] Exception during apply_transform: {e}", flush=True)
                raise
            frames.append(transformed_sample["frames"])
            if label is None:
                label = transformed_sample["label"]
            else:
                assert label == transformed_sample["label"] # label should be the same for all modalities

        #pat_id, visit_idxs = self.samples[index]
        return {
            "frames": frames,  # list of tensors, one per modality
            "label": label,
        }


## multimodal assume the same patient_dataset_type (in this example)
def build_multimodal_dataset(is_train, args):
    split = "train" if is_train else "test"
    modality_configs = get_modality_configs(args, num_modalities=len(args.modalities))

    modality_datasets = []
    for i, config in enumerate(modality_configs):
        setattr(args, "patient_dataset_type", config["patient_dataset_type"])
        setattr(args, "imaging", config["imaging"])
        setattr(args, "anatomic_region", config["anatomic_region"])
        setattr(args, "manufacturers_model_name", config["manufacturers_model_name"])
        setattr(args, "octa_enface_imaging", config["octa_enface_imaging"])

        if args.patient_dataset_type=="volume":
            args.num_frames = 60
            transform = build_transform_3d(is_train, args)
        else:
            transform = build_transform_2d(is_train, args)

        dataset = PatientDataset(
            root_dir=args.data_path,
            split=split,
            mode=args.patient_dataset_type,
            imaging=args.imaging,
            anatomic_region=args.anatomic_region,
            imaging_device=args.manufacturers_model_name,
            octa_enface_imaging=args.octa_enface_imaging,
            concept_id=args.concept_id,
            num_workers=args.num_workers,
            transform=transform,
        )

        modality_datasets.append(dataset)

    return MultiModalDataset(modality_datasets)


def get_modality_configs(args, num_modalities=2):
    modality_configs = []

    for mod_str in args.modalities:
        config = json.loads(mod_str)
        modality_configs.append(config)

    return modality_configs
