# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------


from monai.transforms.transform import MapTransform

from .datasets import *
from copy import deepcopy
import torch


class CenterSlice_Volume(MapTransform):
    def __call__(self, data_dict):
        volume = deepcopy(data_dict["frames"])
        num_frames = volume.shape[0]
        middle_index = (num_frames // 2) - 1 if num_frames % 2 == 0 else num_frames // 2
        data_dict["frames"] = volume[middle_index]
        return data_dict


class GetLabel(MapTransform):
    def __init__(self, concept_id, **kwargs):
        self.concept_id = concept_id
        super().__init__(**kwargs)

    def __call__(self, data_dict):
        data_dict = deepcopy(data_dict)
        if self.concept_id > 0:
            data_dict["label"] = float(data_dict["source_values"][0])
        elif self.concept_id < 0:
            for imd in [s + "_metadata" for s in ["oct", "octa", "cfp", "ir", "faf"]]:
                if imd in data_dict:
                    metadata = data_dict[imd][0]
                    break
            else:
                raise KeyError("Missing a metadata key")
            # Simplified laterality check
            laterality = metadata.get("laterality")
            label = 1 if laterality == "L" else 0
            data_dict["label"] = int(label)
        else:
            data_dict["label"] = int(data_dict["class_idx"])
        return data_dict


class ToFloat(MapTransform):
    def __call__(self, data_dict):
        data_dict = deepcopy(data_dict)
        data_dict["frames"] = data_dict["frames"].to(torch.float)
        return data_dict


class ToRGB(MapTransform):
    def __call__(self, data_dict):
        frame = data_dict["frames"]
        frame = torch.stack((frame, frame, frame), dim=0)  # To RGB [C,H,W]
        data_dict["frames"] = frame
        return data_dict


class SliceVolume(MapTransform):
    def __call__(self, data_dict):
        data_dict = deepcopy(data_dict)
        idx = data_dict["slice_index"]
        data_dict["frames"] = data_dict["frames"][idx, :, :]
        return data_dict
