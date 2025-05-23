# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------
from aireadi_loader.dataloader import PatientDataset, PatientCacheDataset, PatientFastAccessDataset
from aireadi_loader.transforms import FilterFramesLabel, ToFloat


from monai.transforms import (
    Compose,
    Resized,
    RandRotated,
    RandFlipd,
    ScaleIntensityd,
    ToTensord,
    RandRotated,
    RandFlipd,
    ToTensord,
)


def build_dataset(is_train, args):
    split = "train" if is_train else "val"
    if (
        args.patient_dataset_type == "center_slice"
        or args.patient_dataset_type == "slice"
    ):
        transform = build_transform_2d(is_train, args)

    else:
        transform = build_transform_3d(is_train, args)
    # if args.transform is None:
    args.transform = transform

    if args.cache_rate != 0:
        if args.patient_dataset_type == "slice" and (
            args.imaging == "oct"
            or (args.imaging == "octa" and args.octa_enface_imaging is None)
        ):
            dataset = PatientFastAccessDataset(
                root_dir=args.data_path,
                split=split,
                mode=args.patient_dataset_type,
                imaging=args.imaging,
                anatomic_region=args.anatomic_region,
                imaging_device=args.manufacturers_model_name,
                concept_id=args.concept_id,  # 374028 (AMD) 437541 (glaucoma)
                num_workers=args.num_workers,
                cache_rate=args.cache_rate,
                transform=args.transform,
                octa_enface_imaging=args.octa_enface_imaging,
            )
        else:
            dataset = PatientCacheDataset(
                root_dir=args.data_path,
                split=split,
                mode=args.patient_dataset_type,
                imaging=args.imaging,
                anatomic_region=args.anatomic_region,
                imaging_device=args.manufacturers_model_name,
                concept_id=args.concept_id,  # 374028 (AMD) 437541 (glaucoma)
                num_workers=args.num_workers,
                cache_rate=args.cache_rate,
                transform=args.transform,
                octa_enface_imaging=args.octa_enface_imaging,
            )
    else:
        dataset = PatientDataset(
            root_dir=args.data_path,
            split=split,
            mode=args.patient_dataset_type,
            imaging=args.imaging,
            anatomic_region=args.anatomic_region,
            imaging_device=args.manufacturers_model_name,
            concept_id=args.concept_id,
            num_workers=args.num_workers,
            transform=args.transform,
            octa_enface_imaging=args.octa_enface_imaging,
        )

    return dataset


def build_transform_2d(is_train, args):
    train_transform, val_transform = create_2d_transforms(args.imaging, args.input_size)
    if is_train == "train":
        return train_transform
    else:
        return val_transform


def create_2d_transforms(imaging, input_size):

    # train transform
    if imaging == "oct":
        train_transform = Compose(
            [
                Resized(
                    keys=["frames"],
                    spatial_size=(input_size, input_size),
                    mode="bilinear",
                ),  # Resize with bilinear interpolation
                # RandFlipd(keys=["frames"], spatial_axis=1, prob=0.5),  # Random horizontal flip
                ScaleIntensityd(keys=["frames"]),
                ToTensord(
                    keys=["frames", "label"], track_meta=False
                ),  # Convert image and label to tensor
                RandRotated(
                    keys=["frames"],
                    range_x=(-0.17, 0.17),
                    prob=0.5,
                    mode="bilinear",
                ),  # Random rotation (in radian)
                FilterFramesLabel(keys=["frames", "label"]),
                ToFloat(keys=["frames"]),
            ]
        )
    else:
        train_transform = Compose(
            [
                Resized(
                    keys=["frames"],
                    spatial_size=(input_size, input_size),
                    mode="bilinear",
                ),  # Resize with bilinear interpolation
                # RandFlipd(keys=["frames"], spatial_axis=1, prob=0.5),  # 0:vertical, 1:horizontal (ignore C dimension [C, H, W])
                ScaleIntensityd(keys=["frames"]),
                ToTensord(
                    keys=["frames", "label"], track_meta=False
                ),  # Convert image and label to tensor
                RandRotated(
                    keys=["frames"],
                    range_x=(-0.17, 0.17),
                    prob=0.5,
                    mode="bilinear",
                ),  # Random rotation (in radian)
                FilterFramesLabel(keys=["frames", "label"]),
                ToFloat(keys=["frames"]),
            ]
        )

    # eval transform
    val_transform = Compose(
        [
            Resized(
                keys=["frames"], spatial_size=(input_size, input_size), mode="bilinear",
            ),
            ScaleIntensityd(keys=["frames"]),
            ToTensord(
                keys=["frames", "label"], track_meta=False
            ),  # Convert image and label to tensor
            FilterFramesLabel(keys=["frames", "label"]),
            ToFloat(keys=["frames"]),
        ]
    )
    return train_transform, val_transform


def build_transform_3d(is_train, args):
    train_transform, val_transform = create_3d_transforms(
        args.num_frames, args.input_size
    )
    if is_train == "train":
        return train_transform
    else:
        return val_transform


def create_3d_transforms(num_frames, input_size):

    train_transform = Compose(
        [
            Resized(
                keys=["frames"],
                spatial_size=(num_frames, input_size, input_size),
                mode=("trilinear"),
            ),
            RandFlipd(
                keys=["frames"], prob=0.5, spatial_axis=0
            ),  # 0:depth  1:vertical, 2:horizontal (ignore channel dimension [C, D, H, W])
            ScaleIntensityd(keys=["frames"]),
            ToTensord(keys=["frames", "label"], track_meta=False),  # Convert frames and label to tensor
            ToFloat(keys=["frames"]),
        ]
    )

    # eval transform
    val_transform = Compose(
        [
            Resized(
                keys=["frames"],
                spatial_size=(num_frames, input_size, input_size),
                mode=("trilinear"),
            ),
            ScaleIntensityd(keys=["frames"]),
            ToTensord(keys=["frames", "label"], track_meta=False),  # Convert frames and label to tensor
        ]
    )

    return train_transform, val_transform
