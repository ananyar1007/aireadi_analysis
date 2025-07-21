@echo off
setlocal

rem === Configuration ===
set "INPUTSIZE=224"

set "CACHE_RATE=0"
set "CONCEPT_ID=-1"
set "PROCESS_TYPE=slice"
set "DEVICE=Eidon"
set "LOCATION=Macula"
set "IMAGING=cfp"
rem set "OPHTHALMIC_IMAGING=superficial"

rem Adjust paths as needed for Windows

set "OUTPUT_DIR=.\outputs\finetune_aireadi_2d_%CONCEPT_ID%_%PROCESS_TYPE%_%DEVICE%_%LOCATION%_%IMAGING%_%OPHTHALMIC_IMAGING%"
set "YOUR_DATASET_PATH=C:\Users\preet\Documents\AI_READI\"

rem === CUDA / Python Invocation ===
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_VISIBLE_DEVICES=0"

python -m examples.train_2d --nb_classes 2 ^
--data_path "C:\Users\preet\Documents\AI_READI\\" ^
--imaging cfp ^
--manufacturers_model_name "Eidon" ^
--anatomic_region "Macula" ^
--concept_id -1 ^
--cache_rate 0 ^
--octa_enface_imaging "superficial" ^
--input_size 224 ^
--log_dir ".\logs_ft" ^
--output_dir "%OUTPUT_DIR%" ^
--batch_size 16 ^
--patient_dataset_type "slice" ^
--epochs 10 ^
--num_workers 4 ^
--label "mhoccur_amd, Age-related macular degeneration (AM"   ^
--dataset_config_path "C:\\Users\\preet\\Documents\\AI_READI\\dataset_config\\cfp_icare_eidon_png.csv" ^
--img_dir "C:\\Users\\preet\\Documents\\AI_READI\\retinal_photography\\cfp\\icare_eidon_png" ^
--img_type "cfp" ^
--lr 1e-4

endlocal
