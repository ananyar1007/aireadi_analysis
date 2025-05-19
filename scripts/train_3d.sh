#!/bin/bash
# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------

BSZ=16
INPUTSIZE=224
EPOCHS=5
CONCEPT_ID=0
CACHE_RATE=0.00
PROCESS_TYPE=volume
DEVICE=Spectralis
LOCATION=Macula
IMAGING=oct

OUTPUT_DIR=./outputs/finetune_aireadi_3d_${CONCEPT_ID}_${PROCESS_TYPE}_${DEVICE}_${LOCATION}_${IMAGING}/
YOUR_DATASET_PATH=/data/datasets/AIREADI/YEAR2/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="1,3" python -m examples.train_3d --nb_classes 4  \
    --data_path  $YOUR_DATASET_PATH \
    --imaging $IMAGING \
    --manufacturers_model_name $DEVICE \
    --anatomic_region $LOCATION \
    --concept_id $CONCEPT_ID \
    --cache_rate $CACHE_RATE \
    --input_size $INPUTSIZE \
    --num_frames 60 \
    --log_dir ./logs_ft/ \
    --output_dir $OUTPUT_DIR \
    --batch_size $BSZ \
    --patient_dataset_type $PROCESS_TYPE \
    --epochs $EPOCHS \
    --num_workers 10 \
