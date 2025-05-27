import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from examples.build_dataset import build_dataset
import pandas as pd
import copy
from aireadi_loader.datasets import reverse_location_mapping
import os
from monai.data import DataLoader
from PIL import Image
import numpy as np
import tqdm
import time


def try_iterating(d, lim=5):
    i = 0
    for c in tqdm.tqdm(d):
        i += 1
        if i >= lim:
            return c


def save_one_image(path, loader):
    for i, im in enumerate(loader):
        shp = "x".join(str(i) for i in im["frames"].shape)
        im = (
            (im["frames"].detach().cpu().numpy()).astype(np.uint8).transpose(1, 2, 0)
        )  # Transpose to Height, Width, Channels
        im = Image.fromarray(im)

        im.save(f"{path}_{shp}_{i}.png")
        if i > 10:
            return


imaging_mapping = {
    "infrared reflectance": "ir",
    "color photography": "cfp",
    "autofluorescence": "faf",
    "octa": "octa",
    "oct": "oct",
}

valid_modes = {
    "slice": ["cfp", "ir", "oct", "octa"],
    "center_slice": ["oct", "octa"],
    "volume": ["oct", "octa"],
}

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

data_path = "/data/datasets/AIREADI/YEAR2/"


# For storing arguments
class DataSetTestArgs:
    pass


with open("./narrowing_data", "r") as f:
    lines = f.readlines()

item_size = 7
assert len(lines) % item_size == 0, "need line count to be a multiple of item_size"
narrowing_data = []
for i in range(len(lines) // item_size):
    datum = {}
    for d in lines[i * item_size : i * item_size + 5]:
        k, v = d.strip().split(": ")
        datum[k] = v

    narrowing_data.append(datum)
concept_id = -1
split = "train"
ignore_values = [777, 999, 555, 888]
volume_resize = (224, 224)
save_images = False


def is_valid_combo(args):
    if (
        args.imaging == "octa"
        and (args.mode == "volume" or args.mode == "center_slice")
        and args.octa_enface_imaging is not None
    ):
        return False

    elif (
        args.imaging == "octa"
        and args.mode == "slice"
        and args.octa_enface_imaging is not None
        and args.octa_enface_imaging
        not in ["superficial", "deep", "outer_retina", "choriocapillaris"]
    ):
        return False
    # Check if the combination is valid
    elif args.imaging not in valid_combinations:
        print(f"Invalid {args.imaging} for combination")
        return False

    elif args.manufacturers_model_name not in valid_combinations[args.imaging]:
        print(
            f"Invalid device {args.manufacturers_model_name} for imaging {args.imaging}"
        )
        return False

    elif (
        args.anatomic_region
        not in valid_combinations[args.imaging][args.manufacturers_model_name]
    ):
        print(
            f"Invalid region {args.anatomic_region} for imaging {args.imaging} and device {args.manufacturers_model_name}"
        )
        return False
    # Check if mode is valid for the given imaging type
    elif args.mode not in valid_modes:
        print(f"Invalid mode {args.mode}, not in {valid_modes.keys()}")
        return False
    elif args.imaging not in valid_modes[args.mode]:
        print(f"Invalid imaging {args.imaging} for mode {args.mode}")
        return False
    else:
        return True


rows = []
if not os.path.exists("visualized_data_types"):
    os.makedirs("visualized_data_types")
for cache_rate in [0.00, 0.01]:
    for mode in ["volume", "slice", "center_slice"]:
        transform = None
        for datum in narrowing_data:
            for region in eval(datum["Unique Anatomic Regions"]):
                row = copy.copy(datum)
                row["Anatomic Region"] = reverse_location_mapping[region]
                del row["Unique Anatomic Regions"]
                if row["octa_enface_imaging"] == "None":
                    row["octa_enface_imaging"] = None

                # For looking at specific items
                # if row["octa_enface_imaging"] is None:
                #     continue
                # if row != {
                #     "Manufacturer": "Topcon",
                #     "Model": "Maestro2",
                #     "Imaging": "OCTA",
                #     "octa_enface_imaging": "superficial",
                #     "Anatomic Region": "Macula_6x6",
                # }:
                #     continue
                # if mode != "slice":
                #     continue
                row["caching"] = cache_rate
                row["mode"] = mode
                args = DataSetTestArgs()
                args.input_size = 224
                args.split = split
                args.mode = mode
                args.anatomic_region = reverse_location_mapping[region]
                args.imaging = imaging_mapping[datum["Imaging"].lower()]
                args.manufacturers_model_name = datum["Model"]
                args.num_workers = 10
                args.concept_id = concept_id  # 374028 (AMD) 437541 (glaucoma)
                args.ignore_values = ignore_values
                args.volume_resize = volume_resize
                args.transform = transform
                args.cache_rate = cache_rate
                args.root_dir = data_path
                args.data_path = data_path
                args.patient_dataset_type = mode
                args.octa_enface_imaging = row["octa_enface_imaging"]
                args.num_frames = 60
                print(f"Trying row: {row}")
                print(f"Mode: {mode}")
                if is_valid_combo(args):
                    print("Valid Combo", flush=True)
                    row["Valid Combination"] = True
                else:
                    print("Invalid Combo", flush=True)
                    row["Valid Combination"] = False
                print()

                try:

                    def tm():
                        return time.monotonic()

                    print("building dataset...")
                    st = tm()

                    d = build_dataset(is_train=True, args=args)
                    print(f"built dataset in {tm() - st}")

                    print("iterating dataset...")
                    dataset_iters = 5
                    st = tm()
                    item = try_iterating(d, lim=dataset_iters)
                    print(
                        f"iterated dataset {dataset_iters} times in {tm() - st} ({(tm() - st )/ dataset_iters}s/it)"
                    )
                    st = tm()
                    print("building dataloader...")
                    loader = DataLoader(
                        d,
                        batch_size=16,
                        num_workers=10,
                        drop_last=True,
                    )
                    print(f"built dataloader in {tm() - st}")
                    dataloader_iters = 5
                    try_iterating(loader, 5)
                    print(
                        f"iterated dataset {dataloader_iters} times in {tm() - st} ({(tm() - st )/ dataloader_iters}s/it)"
                    )
                    if save_images:
                        if mode != "volume":
                            if datum["octa_enface_imaging"] == "None":
                                save_one_image(
                                    f"./visualized_data_types/{args.anatomic_region}_{args.manufacturers_model_name}_{args.imaging}",
                                    d,
                                )
                            else:
                                save_one_image(
                                    f"./visualized_data_types/{args.anatomic_region}_{args.manufacturers_model_name}_{args.imaging}_{datum['octa_enface_imaging']}",
                                    d,
                                )

                    row["success"] = True
                    print("Loading Success!")
                    row["error"] = None
                except ValueError as e:

                    row["success"] = False
                    print(f"Loading failed! {e}")
                    row["error"] = e
                    print(str(e))
                rows.append(row)
                df = pd.DataFrame(rows)
                # Will sort them in order, meaning the last category to be sorted by will be the main sorted category
                df.sort_values(
                    [
                        "success",
                        "Anatomic Region",
                        "Imaging",
                        "Model",
                        "Manufacturer",
                        "Valid Combination",
                    ],
                    inplace=True,
                )
                df.to_csv("./dataloader_access_table.csv")


df = pd.DataFrame(rows)
df.sort_values(
    [
        "Valid Combination",
        "Manufacturer",
        "Model",
        "Imaging",
        "Anatomic Region",
    ],
    inplace=True,
    ascending=False,
)
df.to_csv("./dataloader_access_table.csv")
