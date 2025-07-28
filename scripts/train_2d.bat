@echo off
setlocal



python -m examples.train_2d --nb_classes 2 ^
--experiment_name "ir_all" ^
--output_dir "train_logs" ^
--data_path "C:\Users\preet\Documents\AI_READI\\" ^
--imaging cfp ^
--manufacturers_model_name "Eidon" ^
--anatomic_region "Macula" ^
--concept_id -1 ^
--cache_rate 0 ^
--octa_enface_imaging "superficial" ^
--input_size 224 ^
--log_dir ".\logs_ft" ^
--batch_size 16 ^
--val_batch_size 16 ^
--patient_dataset_type "slice" ^
--epochs 20 ^
--num_workers 0 ^
--label "mhoccur_ca, Cancer (any type)" ^
--dataset_config_path "C:\Users\preet\Documents\AI_READI\dataset_config\cfp_icare_eidon_ir_heidelberg_spectralis_png_binned.csv" ^
--cfp_img_path "C:\Users\preet\Documents\AI_READI\retinal_photography\cfp\topcon_triton_png3" ^
--ir_img_path "C:\Users\preet\Documents\AI_READI\retinal_photography\ir\heidelberg_spectralis_png" ^
--img_type "cfp and ir" ^
--lr 1e-5 ^
--num_layers 4 ^
--dropout 0.3 
endlocal
