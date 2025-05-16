#!/bin/bash
# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------

BSZ=32
INPUTSIZE=224
EPOCHS=50
CACHE_RATE=0.01
CONCEPT_ID=-1
PROCESS_TYPE=slice
DEVICE=Cirrus
LOCATION=Macula_6x6
IMAGING=octa
OPHTHALMIC_IMAGING=superficial

OUTPUT_DIR=./outputs/finetune_aireadi_2d_${CONCEPT_ID}_${PROCESS_TYPE}_${DEVICE}_${LOCATION}_${IMAGING}_${OPHTHALMIC_IMAGING}/
YOUR_DATASET_PATH=/data/datasets/AIREADI/YEAR2/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="0,1" python -m examples.train_2d --nb_classes 2 \
    --data_path  $YOUR_DATASET_PATH \
    --imaging $IMAGING \
    --manufacturers_model_name $DEVICE \
    --anatomic_region $LOCATION \
    --concept_id $CONCEPT_ID \
    --cache_rate $CACHE_RATE \
    --octa_enface_imaging $OPHTHALMIC_IMAGING \
    --input_size $INPUTSIZE \
    --log_dir ./logs_ft/ \
    --output_dir $OUTPUT_DIR \
    --batch_size $BSZ \
    --patient_dataset_type $PROCESS_TYPE \
    --epochs $EPOCHS \
    --num_workers 10 \
