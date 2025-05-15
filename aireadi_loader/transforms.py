# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------

import numpy as np
import pydicom
import copy

from monai.transforms.transform import MapTransform
from monai.transforms.io.dictionary import LoadImageD

from .datasets import *
import torch.nn.functional as F
import time
from threading import Lock
import tracemalloc


class Slice_Volume(MapTransform):
    def __call__(self, data_dict):
        idx = data_dict["slice_index"]
        data_dict["frames"] = data_dict["frames"][:, idx, :, :]
        return data_dict


class CenterSlice_Volume(MapTransform):
    def __call__(self, data_dict):
        volume = data_dict["frames"]
        num_frames = volume.shape[0]
        middle_index = (num_frames // 2) - 1 if num_frames % 2 == 0 else num_frames // 2
        data_dict["frames"] = volume[middle_index]
        return data_dict


class GetLabel(MapTransform):
    def __init__(self, concept_id, **kwargs):
        self.concept_id = concept_id
        super().__init__(**kwargs)

    def __call__(self, data_dict):
        if self.concept_id >= 0:
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
        data_dict["frames"] = data_dict["frames"].to(torch.float)
        return data_dict


class FilterFramesLabel(MapTransform):
    def __call__(self, data_dict):
        ndd = dict()
        ndd["frames"] = data_dict["frames"]
        ndd["label"] = data_dict["label"]
        return ndd


# NOTE: Extraordinarily slow
class LoadSlice(MapTransform):
    def __call__(self, data):
        slice_idx = data["slice_index"]

        file_path = data["frames"]

        vol = pydicom.dcmread(file_path).pixel_array
        slice_data = vol[slice_idx]

        h, w = slice_data.shape
        final_frame = np.broadcast_to(slice_data[np.newaxis, ...], (3, h, w)).copy()

        data["frames"] = final_frame
        return data


# For looking at stuff
class Inspector(MapTransform):
    def __call__(self, data_dict):
        print(data_dict["frames"].shape)
        return data_dict


class LoadInspect(MapTransform):
    def __call__(self, data_dict):
        im = pydicom.dcmread(data_dict["frames"]).pixel_array
        print("in load inspect", flush=True)
        breakpoint()
        data_dict["frames"] = im
        return data_dict


class ToRGB(MapTransform):
    def __call__(self, data_dict):
        frame = data_dict["frames"]
        frame = torch.stack((frame, frame, frame), dim=0)  # To RGB [C,H,W]
        data_dict["frames"] = frame
        return data_dict


class SliceVolume(MapTransform):
    def __call__(self, data_dict):
        idx = data_dict["slice_index"]
        data_dict["frames"] = data_dict["frames"][idx, :, :]
        return data_dict


class CacheLoadSlice(MapTransform):
    def clear_cache(self):
        import gc

        with self.cache_lock:
            self.vol_cache.clear()
            self.access_counter.clear()
            self.zero_hits = []
            # Force garbage collection
            gc.collect()

    def __init__(
        self,
    ):
        self.times = {
            k: []
            for k in [
                "init",
                "cleanup",
                "readfile",
                "pixel array",
                "data grab",
                "contiguous",
            ]
        }
        self.vol_cache = {}
        self.access_counter = {}
        self.zero_hits = []
        self.cache_lock = Lock()
        self.shape = None

        # tracemalloc.start()

    # TODO: Partial caching will result in a memory leak. Need a way to clean up after transform is used or limit cache size
    def __call__(self, data):
        st = time.monotonic()
        slice_idx = data["slice_index"]
        file_path = data["frames"]

        et = time.monotonic()
        self.times["init"].append(et - st)
        st = time.monotonic()
        # Caching
        with self.cache_lock:
            # if (file_path := self.root_dir + data_path) not in self.vol_cache:
            if file_path not in self.vol_cache:
                st = time.monotonic()
                self.vol_cache[file_path] = pydicom.dcmread(file_path)
                self.times["readfile"].append(et - st)
                st = time.monotonic()
                self.vol_cache[file_path] = self.vol_cache[file_path].pixel_array
                et = time.monotonic()
                self.times["pixel array"].append(et - st)
                self.access_counter[file_path] = copy.copy(
                    self.vol_cache[file_path].shape[0]
                )

            slice_data = self.vol_cache[file_path][slice_idx]
            self.access_counter[file_path] -= 1

            # Zero check inside the lock
            if self.access_counter[file_path] == 0:
                self.vol_cache[file_path] = None
                del self.vol_cache[file_path]
                print("deleting!", flush=True)
                self.zero_hits.append(file_path)
        et = time.monotonic()
        self.times["data grab"].append(et - st)

        st = time.monotonic()
        h, w = slice_data.shape
        final_frame = np.stack([slice_data, slice_data, slice_data], axis=0)
        final_frame = np.broadcast_to(slice_data[np.newaxis, ...], (3, h, w)).copy()
        et = time.monotonic()
        self.times["contiguous"].append(et - st)
        data["frames"] = final_frame
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics("lineno")

        # for stat in top_stats[:5]:
        #     print(stat)

        return data


class LoadCenterSlice(MapTransform):
    def __init__(
        self,
        mapping_visit2patient,
        data_type,
        root_dir,
        concept_id,
        return_patient_id,
    ):
        self.mapping_visit2patient = mapping_visit2patient
        self.data_type = data_type
        self.root_dir = root_dir
        self.concept_id = concept_id
        self.return_patient_id = return_patient_id

    def __call__(self, data_dict):
        data_path = data_dict["frames"][0]
        ## Process images
        dicom_file = pydicom.dcmread(self.root_dir + data_path)
        if (
            self.data_type == "oct" or self.data_type == "octa"
        ):  # dicom file inclues volume data
            volume = dicom_file.pixel_array
            num_frames = volume.shape[0]
            middle_index = (
                (num_frames // 2) - 1 if num_frames % 2 == 0 else num_frames // 2
            )
            frame = volume[middle_index]
            frame = np.stack([frame] * 3, axis=0)  # To RGB [H, W, C]

            frame = frame.transpose(
                2, 0, 1
            )  # monai require channel first [C, H, W], either torch.tensor or Numpy
            frame = np.asarray(frame)
        else:
            raise ValueError(
                f"LoadCenterSlice data type must be 'oct' or 'octa', but instead got '{self.data_type}'"
            )

        return data_dict


class LoadVolume(MapTransform):
    def __init__(
        self,
        visits_dict,
        mapping_visit2patient,
        data_type,
        root_dir,
        concept_id,
        return_patient_id,
        imaging,
        volume_resize,
    ):
        self.visits_dict = visits_dict
        self.mapping_visit2patient = mapping_visit2patient
        self.data_type = data_type
        self.root_dir = root_dir
        self.concept_id = concept_id
        self.return_patient_id = return_patient_id
        self.imaging = imaging
        self.volume_resize = volume_resize

    def __call__(self, idx):
        data_dict = self.visits_dict[idx]
        patient_id = self.mapping_visit2patient[idx]

        data_path = data_dict["frames"][0]
        ## Process images
        dicom_file = pydicom.dcmread(self.root_dir + data_path)
        volume = dicom_file.pixel_array  # [D, H, W]

        volume = (
            torch.tensor(volume).unsqueeze(1).float()
            if len(volume.shape) == 3
            else torch.tensor(volume).float()
        )
        if self.volume_resize:
            volume = F.interpolate(
                volume, size=self.volume_resize, mode="bicubic", align_corners=False
            )
        volume = volume[:, 0, :, :]

        volume = volume.unsqueeze(
            0
        )  # [C, D, H, W] Monai 3D transform expects this shape (C=1).

        if self.concept_id >= 0:
            try:
                if len(data_dict["source_values"]) > 1:
                    raise ValueError(
                        "More than one values are detected fot the patient: %s"
                        % patient_id
                    )
            except ValueError as e:
                print(f"ValueError: {e}")
            sample = {
                "image": volume,
                "label": float(data_dict["source_values"][0]),
            }
        else:
            sample = {"image": volume, "label": int(data_dict["class_idx"])}

        if self.return_patient_id:
            sample_pid = {"metadata": patient_id}
            sample = {**sample, **sample_pid}

        return sample
