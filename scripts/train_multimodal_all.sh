#!/bin/bash
# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------

BSZ=32
INPUTSIZE=256
EPOCHS=50
RATIO=0.9
CONCEPT_ID=-1

YOUR_DATASET_PATH=/data/datasets/AIREADI/YEAR2/
LOG_DIR=./logs_ft/

# Set GPU devices
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="1,4,5,6"


# === First combination: CFP + OCTA-superficial ===
MODALITIES=(
    '{"patient_dataset_type": "slice", "imaging": "cfp", "manufacturers_model_name": "Maestro2", "anatomic_region": "Macula", "octa_enface_imaging": null}'
    '{"patient_dataset_type": "slice", "imaging": "octa", "manufacturers_model_name": "Cirrus", "anatomic_region": "Macula_6x6", "octa_enface_imaging": "superficial"}'
)

OUTPUT_DIR=./outputs/finetune_aireadi_2d_${CONCEPT_ID}_cfp_octa/
python -m examples.train_multimodal \
    --nb_classes 1 \
    --data_path "$YOUR_DATASET_PATH" \
    --modalities "${MODALITIES[@]}" \
    --concept_id "$CONCEPT_ID" \
    --input_size "$INPUTSIZE" \
    --log_dir "$LOG_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BSZ" \
    --warmup_epochs 10 \
    --epochs "$EPOCHS" \
    --num_workers 10

# === Second combination: OCTA modalities ===
MODALITIES=(
    '{"patient_dataset_type": "slice", "imaging": "octa", "manufacturers_model_name": "Cirrus", "anatomic_region": "Macula_6x6", "octa_enface_imaging": "superficial"}'
    '{"patient_dataset_type": "slice", "imaging": "octa", "manufacturers_model_name": "Cirrus", "anatomic_region": "Macula_6x6", "octa_enface_imaging": "deep"}'
    '{"patient_dataset_type": "slice", "imaging": "octa", "manufacturers_model_name": "Cirrus", "anatomic_region": "Macula_6x6", "octa_enface_imaging": "outer_retina"}'
    '{"patient_dataset_type": "slice", "imaging": "octa", "manufacturers_model_name": "Cirrus", "anatomic_region": "Macula_6x6", "octa_enface_imaging": "choriocapillaris"}'
)
OUTPUT_DIR=./outputs/finetune_aireadi_2d_${CONCEPT_ID}_octa_multi/
python -m examples.train_multimodal \
    --nb_classes 1 \
    --data_path "$YOUR_DATASET_PATH" \
    --modalities "${MODALITIES[@]}" \
    --concept_id "$CONCEPT_ID" \
    --input_size "$INPUTSIZE" \
    --log_dir "$LOG_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BSZ" \
    --warmup_epochs 10 \
    --epochs "$EPOCHS" \
    --num_workers 10
