# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------

import pydicom
import copy

import numpy as np
import torch


from torch.utils.data import IterableDataset
from monai.data.dataset import CacheDataset, Dataset
from monai.transforms import (
    ToTensord,
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    SelectItemsd,
    Identityd
)
import torch.nn.functional as F
from typing import Optional, Callable

from .datasets import (
    location_mapping,
    load_ai_readi_clinical_data,
    check_concept_table,
    filter_clinical_table,
    get_patient,
    load_ai_readi_data,
    get_aireadi_setting,
    filter_patient_dict,
)

from .transforms import (
    CenterSlice_Volume,
    GetLabel,
    SliceVolume,
    ToRGB,
)


import random
from typing import Any

"""
A MONAI Dataset for loading and processing AI-READI, with support for multiple modes of data representation
(slice, center_slice, volume).
The dataset is filtered based on device, anatomical region, imaging type, and concept IDs.
"""


class PatientDatasetInit:
    mode: str
    imaging: str
    data_type: str
    location: str
    pre_patient_cohort: str
    imaging_device: str
    root_dir: str
    concept_id: int
    mapping_patient2visit: dict[int, list[int]]
    visits_dict: dict[int, dict[str, Any]]
    patient_all_dict: dict[int, dict[str, Any]]
    aireadi_patient_id_dict: dict[str, np.ndarray]
    octa_enface_imaging: Optional[str]
    transform: Optional[Callable]
    cache_rate: Optional[float]

    def init_logic(
        self,
        root_dir,
        split="train",
        mode="slice",
        anatomic_region="Macula",
        imaging_device="Maestro2",
        imaging="oct",
        concept_id=-1,
        ignore_values=[777, 999, 555, 888],
        octa_enface_imaging=None,
        cache_rate=None,
        transform=None,
        **kwargs,
    ):
        # Define valid mode and imaging pairs
        valid_modes = {
            "slice": ["cfp", "ir", "oct", "octa"],
            "center_slice": ["oct", "octa"],
            "volume": ["oct", "octa"],
        }

        # Check if mode is valid for the given imaging type
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode: '{mode}'. Allowed values are {list(valid_modes.keys())}."
            )
        if imaging not in valid_modes[mode]:
            raise ValueError(
                f"Invalid mode '{mode}' for imaging type '{imaging}'. "
                f"Allowed imaging types for '{mode}' are {valid_modes[mode]}."
            )

        # Ensure that 'octa' is only used for volume when octa_enface_imaging is None
        if (
            imaging == "octa"
            and (mode == "volume" or mode == "center_slice")
            and octa_enface_imaging is not None
        ):
            raise ValueError(
                f"Invalid configuration: Imaging type 'octa' in 'volume' or 'center_slice' mode requires 'octa_enface_imaging=None', "
                f"but got '{octa_enface_imaging}'. Enface types such as ['superficial', 'deep', 'outer_retina', 'choriocapillaris'] "
                f"are only valid when using 'slice' modes."
            )

        if (
            imaging == "octa"
            and mode == "slice"
            and octa_enface_imaging is not None
            and octa_enface_imaging
            not in ["superficial", "deep", "outer_retina", "choriocapillaris"]
        ):
            raise ValueError(
                f"Invalid enface type: For 'octa' imaging in 'slice' mode, 'octa_enface_imaging' must be one of "
                f"['superficial', 'deep', 'outer_retina', 'choriocapillaris']. Got: '{octa_enface_imaging}'."
            )

        # Define valid imaging, device, and anatomic region combinations
        valid_combinations = {
            "oct": {
                "Spectralis": ["Optic_Disc", "Macula"],
                "Maestro2": ["Macula", "Wide_Field", "Macula_6x6"],
                "Triton": ["Macula_12x12", "Macula_6x6", "Optic_Disc"],
                "Cirrus": ["Optic_Disc_6x6", "Optic_Disc", "Macula", "Macula_6x6"],
            },
            "ir": {
                "Spectralis": ["Optic_Disc", "Macula"],
                "Maestro2": ["Macula_6x6"],
                "Cirrus": ["Optic_Disc", "Macula"],
                "Eidon": ["Macula"],
            },
            "cfp": {
                "Maestro2": ["Macula", "Wide_Field"],
                "Triton": ["Macula_12x12", "Macula_6x6", "Optic_Disc"],
                "Eidon": ["Mosaic", "Macula", "Nasal", "Temporal_Periphery"],
            },
            "faf": {"Eidon": ["Macula"]},
            "octa": {
                "Maestro2": ["Macula_6x6"],
                "Triton": ["Macula_12x12", "Macula_6x6"],
                "Cirrus": ["Optic_Disc_6x6", "Macula_6x6"],
            },
        }

        # Check if the combination is valid
        if imaging not in valid_combinations:
            raise ValueError(
                f"Invalid imaging type: '{imaging}'. Allowed values are {list(valid_combinations.keys())}."
            )

        if imaging_device not in valid_combinations[imaging]:
            raise ValueError(
                f"Invalid imaging device '{imaging_device}' for imaging type '{imaging}'. "
                f"Allowed imaging devices are {list(valid_combinations[imaging].keys())}."
            )

        if anatomic_region not in valid_combinations[imaging][imaging_device]:
            raise ValueError(
                f"Invalid anatomic region '{anatomic_region}' for imaging type '{imaging}' and imaging device '{imaging_device}'. "
                f"Allowed anatomic regions are {valid_combinations[imaging][imaging_device]}."
            )

        self.mode = mode

        self.root_dir = root_dir

        self.concept_id = concept_id

        self.imaging_device = imaging_device
        self.location = location_mapping[anatomic_region]
        self.pre_patient_cohort = imaging_device
        self.imaging = imaging
        self.octa_opthalmic_imaging = octa_enface_imaging
        self.octa_enface_imaging = octa_enface_imaging
        if self.imaging == "oct":  # this is the split in manifest file level.
            self.data_type = "oct"
        elif self.imaging == "octa":  # this is the split in manifest file level.
            self.data_type = "octa"
        elif self.imaging == "cfp" or self.imaging == "ir":
            self.data_type = "cfp"
        else:
            raise NotImplementedError(
                "This imaging %s has not been implemented." % self.imaging
            )
        (
            patient_id_recommended_split,
            self.aireadi_patient_id_dict,
            self.patient_all_dict,
        ) = load_ai_readi_data(self.root_dir, self.data_type, self.octa_enface_imaging)
        self.used_aireadi_condition_list, self.used_aireadi_filtered_patient_list = (
            get_aireadi_setting(
                patient_id_recommended_split,
                self.aireadi_patient_id_dict,
                split=split,
                device_model_name=self.imaging_device,
                location=self.location,
                pre_patient_cohort=self.pre_patient_cohort,
            )
        )
        _, self.patient_list = get_aireadi_setting(
            patient_id_recommended_split,
            self.aireadi_patient_id_dict,
            split="all",
            device_model_name=self.imaging_device,
            location=self.location,
            pre_patient_cohort=self.pre_patient_cohort,
        )

        self.patient_list = sorted(self.patient_list)

        self.used_aireadi_patient_dict = filter_patient_dict(
            self.patient_all_dict,
            self.data_type,
            self.imaging,
            condition=self.used_aireadi_condition_list,
            pre_filtered_patient_id_list=self.used_aireadi_filtered_patient_list,
        )
        self.used_aireadi_patient_list = sorted(
            list(self.used_aireadi_patient_dict.keys())
        )

        self.concept_table = load_ai_readi_clinical_data(self.root_dir)
        if self.concept_table.empty:
            raise ValueError("Error: The concept_id has no entries.")

        invalid_concept_ids = check_concept_table(self.concept_table, ignore_values)
        if int(concept_id) in invalid_concept_ids:
            raise ValueError(
                f"Invalid concept ID detected (All values are empty with this concept ID): {invalid_concept_ids}"
            )

        filtered_concept_table = filter_clinical_table(
            self.concept_table, concept_id, ignore_values
        )
        if filtered_concept_table.empty:
            raise ValueError("Error: The concept ID has no entries.")

        print(f"Loading AI-READI dataset for {self.imaging}")
        print(self.root_dir)
        self.patients, self.visits_dict, self.mapping_patient2visit = get_patient(
            self.root_dir,
            self.used_aireadi_patient_dict,
            self.data_type,
            filtered_concept_table,
        )
        self.mapping_visit2patient = {
            visit_idx: patient_id
            for patient_id, visit_idx_list in self.mapping_patient2visit.items()
            for visit_idx in visit_idx_list
        }

        for key, value in kwargs.items():
            setattr(self, key, value)


        for k in self.visits_dict.keys():
            self.visits_dict[k]['index'] = k

        for k,v in self.visits_dict.items():
            self.visits_dict[k]['path'] = self.visits_dict[k]['frames']

        # For indexing frame level (only used in mode == 'slice', and oct or octa)
        if self.imaging == "octa" and self.octa_enface_imaging is not None:  # slab data
            # slab data is 2D
            if self.mode == "slice":
                loadtransform = [
                    LoadImageD(
                        keys=["frames"],
                        reader="PydicomReader",
                        swap_ij=False,
                        dtype=np.uint8,
                    ),
                    ToRGB(keys=["frames"]),
                    GetLabel(keys=["label"], concept_id=self.concept_id),
                ]
            else:
                raise ValueError(
                    f"Expected mode 'slice' for octa enface imaging (slab), but got {self.mode} instead"
                )
        elif self.imaging == "oct" or self.imaging == "octa":
            if self.mode == "slice":
                # Extend the Volume-Level Dictionary to Slice-Level
                self.make_visits_dict_slice()
                if cache_rate is not None:
                    loadtransform = [] #PatientFastAccessDataloader
                else:

                    self.visits_dict = self.visits_dict_slice
                    loadtransform = [
                        LoadImageD(
                            keys=["frames"],
                            reader="PydicomReader",
                            swap_ij=False,
                            dtype=np.uint8,
                        ),
                        SliceVolume(keys=["frames", "slice_index"]),
                        ToRGB(keys=["frames"]),
                    ]
            elif self.mode == "center_slice":
                loadtransform = [
                    LoadImageD(
                        keys=["frames"],
                        reader="PydicomReader",
                        swap_ij=False,
                        dtype=np.uint8,
                    ),
                    CenterSlice_Volume(keys=["frames"]),
                    ToRGB(keys=["frames"]),
                ]
            elif self.mode == "volume":
                loadtransform = [
                    LoadImageD(
                        keys=["frames"],
                        reader="PydicomReader",
                        swap_ij=False,
                        dtype=np.uint8,
                    ),
                    EnsureChannelFirstD(keys=["frames"]),
                ]
            else:
                raise ValueError(
                    "mode must be one of ['slice', 'center_slice', 'volume']"
                )
        elif (self.imaging == "cfp" or self.imaging == "ir") and self.mode == "slice":
            if (
                self.imaging_device == "Spectralis" or self.imaging_device == "Cirrus"
            ) and self.imaging == "ir":
                loadtransform = [
                    LoadImageD(
                        keys=["frames"],
                        reader="PydicomReader",
                        swap_ij=False,
                        dtype=np.uint8,
                    ),
                    ToRGB(keys=["frames"]),
                ]
            else:
                loadtransform = [
                    LoadImageD(
                        keys=["frames"],
                        channel_dim=2,
                        ensure_channel_first=True,
                        reader="PydicomReader",
                        swap_ij=False,
                        dtype=np.uint8,
                    ),
                ]
        else:
            raise ValueError(
                f"Expected one of ['oct', 'octa', 'ir', 'cfp'], but got {self.imaging}"
            )
        if transform is None:
            transform = Compose([Identityd(keys=["frames", "label"])])

        self.transform = Compose(
            [
                *loadtransform,
                GetLabel(keys=["label"], concept_id=self.concept_id),
                *transform.transforms,
                ToTensord(
                    keys=["frames", "label"],
                    track_meta=False,
                    dtype=torch.float32,
                ),  # Convert image and label to tensor
                SelectItemsd(keys=["frames", "label","index","path"],allow_missing_keys=True),
            ]
        )
        return self

    def check_source_value_types(self):
        """
        Checks the data types of 'source_values' in the visits_dict.

        Args:
            visits_dict (dict): Dictionary containing visit data.

        Returns:
            str: A message indicating whether all source_values are of the same type or mixed types.
        """
        source_value_types = set()

        for visit_id, visit_data in self.visits_dict.items():
            source_values = visit_data.get("source_values", [])
            for value in source_values:
                if value.is_integer():
                    source_value_types.add("int")
                elif isinstance(value, float):
                    source_value_types.add("float")
                else:
                    raise ValueError(
                        f"Source value {value} in visit {visit_id} is of unsupported type: {type(value)}"
                    )

        return source_value_types

    def make_visits_dict_slice(self):
        visits_dict_slice = dict()
        i = 0
        for visit_idx, visit_data in self.visits_dict.items():
            for idx, data_metadata in enumerate(visit_data[f"{self.imaging}_metadata"]):
                resolution = data_metadata["resolution"]
                num_slices = resolution[0]
                for slice_idx in range(num_slices):
                    slice_data = copy.copy(visit_data)
                    slice_data["slice_index"] = slice_idx
                    slice_data["index"] = i #global index of item
                    visits_dict_slice[i] = slice_data
                    i += 1

        self.visits_dict_slice = visits_dict_slice

    def sanity_check(
        self, frame_or_volume, data_dict
    ):  # Sanity Check with laterality label
        ###  CAUTION!!! remove horizontal flip augmentation when you use this target
        laterality_list = []
        for data_metadata in data_dict[f"{self.data_type}_metadata"]:
            if "laterality" in data_metadata:
                laterality_list.append(data_metadata["laterality"])
            else:
                laterality_list.append(None)  # Handle missing 'laterality'
        if laterality_list[0] == "L":
            label = 1
        else:
            label = 0

        sample = {"frames": frame_or_volume, "label": int(label)}

        return sample


class PatientDataset(PatientDatasetInit, Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        mode="slice",
        anatomic_region="Macula",
        imaging_device="Maestro2",
        imaging="oct",
        concept_id=-1,
        ignore_values=[777, 999, 555, 888],
        transform=None,
        octa_enface_imaging=None,
        **kwargs,
    ):
        """
        Args:
            root_dir (str): Path to the dataset directory containing the images and metadata.
            split (str): Dataset split to load. One of {'train', 'val', 'test', 'all'}. Default is 'train'.
            mode (str): Data representation mode. One of {'slice', 'center_slice', 'volume'}.
                        - "slice": Returns individual slices from volumes.
                        - "center_slice": Returns the center slice of each volume.
                        - "volume": Returns the full volume.
            anatomic_region (str): Anatomical region to filter the dataset. Default is 'Macula'.
            imaging_device (str): Device model to filter the dataset. Default is 'Maestro2'.
            imaging (str): Imaging modality to filter the dataset (e.g., 'oct', 'cfp', 'ir'). Default is 'oct'.
            concept_id (int): Concept ID to filter patients. If -1, `cls_idx` is derived from `study_group`. Default is -1.
            ignore_values (list): A list of `source_value` values to ignore during filtering. Default is [777, 999].
            transform (callable, optional): A PyTorch-compatible transform to apply on individual samples. Default is None.
            octa_enface_imaging (str): Ophthalmic image type for OCTA enface to filter the dataset (e.g., superficial, deep, outer_retina, choriocapillaris). Default is 'superficial'.
            **kwargs: Additional arguments to be stored as attributes of the class.

        Attributes:
            patients (dict): A dictionary containing patient information after filtering.
            visits_dict (dict): A dictionary mapping visit indices to metadata.
        """
        super().init_logic(
            root_dir=root_dir,
            split=split,
            mode=mode,
            anatomic_region=anatomic_region,
            imaging_device=imaging_device,
            imaging=imaging,
            concept_id=concept_id,
            ignore_values=ignore_values,
            octa_enface_imaging=octa_enface_imaging,
            transform=transform,
        )

        super().__init__(self.visits_dict, transform=self.transform)


class PatientCacheDataset(PatientDatasetInit, CacheDataset):
    def __init__(
        self,
        root_dir,
        split="train",
        mode="slice",
        anatomic_region="Macula",
        imaging_device="Maestro2",
        imaging="oct",
        concept_id=-1,
        cache_rate=0.2,
        ignore_values=[777, 999, 555, 888],
        num_workers=10,
        transform=None,
        octa_enface_imaging=None,
        **kwargs,
    ):
        """
        Args:
            root_dir (str): Path to the dataset directory containing the images and metadata.
            split (str): Dataset split to load. One of {'train', 'val', 'test', 'all'}. Default is 'train'.
            mode (str): Data representation mode. One of {'slice', 'center_slice', 'volume'}.
                        - "slice": Returns individual slices from volumes.
                        - "center_slice": Returns the center slice of each volume.
                        - "volume": Returns the full volume.
            anatomic_region (str): Anatomical region to filter the dataset. Default is 'Macula'.
            imaging_device (str): Device model to filter the dataset. Default is 'Maestro2'.
            imaging (str): Imaging modality to filter the dataset (e.g., 'oct', 'cfp', 'ir'). Default is 'oct'.
            concept_id (int): Concept ID to filter patients. If -1, `cls_idx` is derived from `study_group`. Default is -1.
            cache_rate (float): Percentage of dataset to hold in cache. Default is 0.2.
            ignore_values (list): A list of `source_value` values to ignore during filtering. Default is [777, 999].
            num_workers: Number of workers to pre-cache data with.
            transform (callable, optional): A PyTorch-compatible transform to apply on individual samples. Default is None.
            octa_enface_imaging (str): Specifies OCTA slab data to access. One of {'superficial', 'deep', 'outer_retina', 'choriocapillaris'}.
            **kwargs: Additional arguments to be stored as attributes of the class.

        Attributes:
            patients (dict): A dictionary containing patient information after filtering.
            visits_dict (dict): A dictionary mapping visit indices to metadata.
        """
        super().init_logic(
            root_dir=root_dir,
            split=split,
            mode=mode,
            anatomic_region=anatomic_region,
            imaging_device=imaging_device,
            imaging=imaging,
            concept_id=concept_id,
            ignore_values=ignore_values,
            octa_enface_imaging=octa_enface_imaging,
            cache_rate=cache_rate,
            transform=transform,
        )

        super().__init__(
            data=self.visits_dict,
            transform=self.transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )


class PatientFastAccessDataset(PatientDatasetInit, IterableDataset):
    def __init__(
        self,
        root_dir,
        split="train",
        mode="slice",
        anatomic_region="Macula",
        imaging_device="Maestro2",
        imaging="oct",
        concept_id=-1,
        cache_rate=0.2,
        ignore_values=[777, 999, 555, 888],
        transform=None,
        shuffle=True,
        **kwargs,
    ):
        """
        PatientFastAccessDataloader caches data at the volume level and randomly returns slices all the cached volumes. Once all slices have been returned from a volume, the volume is discarded until the next epoch. The higher the cache rate, the more volumes can be loaded into memory, and the higher the variance in training data.
        Args:
            root_dir (str): Path to the dataset directory containing the images and metadata.
            split (str): Dataset split to load. One of {'train', 'val', 'test', 'all'}. Default is 'train'.
            mode (str): Data representation mode. One of {'slice', 'center_slice', 'volume'}.
                        - "slice": Returns individual slices from volumes.
                        - "center_slice": Returns the center slice of each volume.
                        - "volume": Returns the full volume.
            anatomic_region (str): Anatomical region to filter the dataset. Default is 'Macula'.
            imaging_device (str): Device model to filter the dataset. Default is 'Maestro2'.
            imaging (str): Imaging modality to filter the dataset (e.g., 'oct', 'cfp', 'ir'). Default is 'oct'.
            concept_id (int): Concept ID to filter patients. If -1, `cls_idx` is derived from `study_group`. Default is -1.
            cache_rate (float): Percentage of dataset to hold in cache. Default is 0.2.
            ignore_values (list): A list of `source_value` values to ignore during filtering. Default is [777, 999].
            transform (callable, optional): A PyTorch-compatible transform to apply on individual samples. Default is None.
            shuffle (bool): Whether or not to shuffle volumes and enable random slice sampling.
            **kwargs: Additional arguments to be stored as attributes of the class.

        Attributes:
            patients (dict): A dictionary containing patient information after filtering.
            visits_dict (dict): A dictionary mapping visit indices to metadata.
        """
        super().init_logic(
            root_dir=root_dir,
            split=split,
            mode=mode,
            anatomic_region=anatomic_region,
            imaging_device=imaging_device,
            imaging=imaging,
            concept_id=concept_id,
            cache_rate=cache_rate,
            ignore_values=ignore_values,
            transform=transform,
        )
        self.shuffle = shuffle
        self.cache_rate = cache_rate
        self.slice_cache = {}
        self.total_slice_cache_size = int(len(self.visits_dict_slice) * cache_rate)
        print(f'Cache size: {self.total_slice_cache_size} slices')
        self.main_volume_list = list(self.visits_dict.keys())
        if self.shuffle:
            random.shuffle(self.main_volume_list)

    def __len__(self):
        """
        Return the total number of slices across all volumes.
        """

        return len(self.visits_dict_slice)

    def __iter__(self):

        return self._slice_gen()
    
    def _slice_gen(self):
        self.worker_init_fn()

        while (len(self.worker_vol_list) > 0) | (len(self.slice_cache) > 0):

            ## Populate cache
            while (len(self.slice_cache) < self.worker_slice_cache_size) and (
                len(self.worker_vol_list) > 0
            ):
                vol_idx = self.worker_vol_list.pop(0)
                data_path = self.visits_dict[vol_idx]["frames"]
                dicom_file = pydicom.dcmread(data_path)
                num_slices = dicom_file.pixel_array.shape[0]
                # add to cache
                for i in range(num_slices):
                    self.slice_cache[(vol_idx, i)] = dicom_file.pixel_array[i].copy()
            # retrieve random slice from cache
            if self.shuffle:
                slice_idx = random.choice(list(self.slice_cache.keys()))
            else:
                slice_idx =  list(self.slice_cache.keys())[0] #first one from ordered dict
            frame = self.slice_cache.pop(slice_idx)

            data_dict = self.visits_dict[slice_idx[0]].copy()

            frame = np.stack([frame] * 3, axis=0)  # To RGB [C,H, W]
            frame = np.asarray(frame, dtype=float)
            data_dict["frames"] = frame
            data_dict["index"] = slice_idx
            
            data_dict = self.transform(data_dict)

            yield data_dict

    def worker_init_fn(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_vol_list = self.main_volume_list.copy()
            self.worker_slice_cache_size = self.total_slice_cache_size
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            vol_idxs = self.main_volume_list

            vols_per_worker = int(np.ceil(len(vol_idxs) / num_workers))
            start = vols_per_worker * worker_id
            end = min(vols_per_worker * (worker_id + 1), len(vol_idxs))
            self.worker_vol_list = vol_idxs[start:end].copy()
            self.worker_slice_cache_size = self.total_slice_cache_size // num_workers

        if self.shuffle:
            random.shuffle(self.worker_vol_list)
