# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------
import pandas as pd
import os

location_mapping = {
    "Macula": "Macula",
    "Macula_6x6": "Macula, 6 x 6",
    "Macula_12x12": "Macula, 12 x 12",
    "Wide_Field": "Wide Field",
    "Optic_Disc": "Optic Disc",
    "Temporal_Periphery": "Temporal Periphery",
    "Optic_Disc_6x6": "Optic Disc, 6 x 6",
    "Macula_or_Optic_Disc": "Macula or Optic Disc",
    "Mosaic": "Mosaic",
    "Nasal": "Nasal",
}

reverse_location_mapping = {v:k for k,v in location_mapping.items()}

aireadi_label_mapping = {
    'healthy': 0,
    'pre_diabetes_lifestyle_controlled': 1,
    'oral_medication_and_or_non_insulin_injectable_medication_controlled': 2,
    'insulin_dependent': 3,
}


def load_ai_readi_data(dataset_directory, data_type, octa_enface_imaging):

    participants_tsv = dataset_directory + 'participants.tsv'
    participants_json = dataset_directory + 'participants.json'
    participants_df = pd.read_csv(participants_tsv, sep='\t')

    ## Image data
    oct_retinal_directory = dataset_directory + 'retinal_oct/'
    cfp_retinal_directory = dataset_directory + 'retinal_photography/'
    octa_retinal_directory = dataset_directory + 'retinal_octa/'
    # retinal_flio_directory = dataset_directory + 'retinal_flio/'

    if data_type == 'oct':
        manifest_tsv = oct_retinal_directory + 'manifest.tsv'
    elif data_type == 'cfp':
        manifest_tsv = cfp_retinal_directory + 'manifest.tsv'
    elif data_type == 'octa':
        manifest_tsv = octa_retinal_directory + 'manifest.tsv'
    else:
        raise NotImplementedError("This imaging has not been implemented.")
    manifest_df = pd.read_csv(manifest_tsv, sep='\t')

    aireadi_patient_all_dict = get_patient_dict(data_type, octa_enface_imaging, participants_df, manifest_df, label_mapping=aireadi_label_mapping)

    # Get patient-lvel recommended split
    patient_id_recommended_split = {}
    for idx, row in participants_df.iterrows():
        patient_id = row['participant_id']
        patient_id_recommended_split[patient_id] = row['recommended_split']

    aireadi_patient_id_dict = store_patient_id(manifest_df, data_type)

    return patient_id_recommended_split, aireadi_patient_id_dict, aireadi_patient_all_dict



def store_patient_id(manifest_df, imaging):
    # Store the patient ID into a dictionary
    aireadi_patient_id_dict = {}
    num_all_patients = manifest_df['participant_id'].unique()
    #print('Number of all patients: ', len(num_all_patients))
    aireadi_patient_id_dict['All'] = num_all_patients
    num_pat_has_hei = manifest_df[manifest_df['manufacturer'] == 'Heidelberg']['participant_id'].unique()
    #print('Number of patients with Heidelberg OCT: ', len(num_pat_has_hei))
    aireadi_patient_id_dict['Heidelberg'] = num_pat_has_hei
    num_pat_has_topcon = manifest_df[manifest_df['manufacturer'] == 'Topcon']['participant_id'].unique()
    aireadi_patient_id_dict['Topcon'] = num_pat_has_topcon
    #print('Number of patients with Topcon OCT: ', len(num_pat_has_topcon))
    num_pat_has_maestro = manifest_df[manifest_df['manufacturers_model_name'] == 'Maestro2']['participant_id'].unique()
    aireadi_patient_id_dict['Maestro'] = num_pat_has_maestro
    #print('Number of patients with Maestro OCT: ', len(num_pat_has_maestro))
    num_pat_has_triton = manifest_df[manifest_df['manufacturers_model_name'] == 'Triton']['participant_id'].unique()
    #print('Number of patients with Triton OCT: ', len(num_pat_has_triton))
    aireadi_patient_id_dict['Triton'] = num_pat_has_triton
    num_pat_has_zei = manifest_df[manifest_df['manufacturer'] == 'Zeiss']['participant_id'].unique()
    #print('Number of patients with Heidelberg OCT: ', len(num_pat_has_hei))
    aireadi_patient_id_dict['Zeiss'] = num_pat_has_zei
    num_pat_has_ica = manifest_df[manifest_df['manufacturer'] == 'iCare']['participant_id'].unique()
    #print('Number of patients with Heidelberg OCT: ', len(num_pat_has_hei))
    aireadi_patient_id_dict['iCare'] = num_pat_has_ica
    num_pat_has_opt = manifest_df[manifest_df['manufacturer'] == 'Optomed']['participant_id'].unique()
    #print('Number of patients with Heidelberg OCT: ', len(num_pat_has_hei))
    aireadi_patient_id_dict['Optomed'] = num_pat_has_opt

    return aireadi_patient_id_dict


def load_ai_readi_clinical_data(dataset_directory):
    """
    Loads and processes clinical data from condition, measurement, and observation files.
    Extracts relevant columns and combines them into a single table.

    Args:
        dataset_directory (str): Path to the dataset directory containing clinical data.

    Returns:
        pd.DataFrame: A combined table with columns ['person_id', 'concept_id', 'source_value', 'definition'].
    """
    clinical_directory = dataset_directory + "clinical_data/"
    condition_csv = clinical_directory + "condition_occurrence.csv"
    measurement_csv = clinical_directory + "measurement.csv"
    observation_csv = clinical_directory + "observation.csv"

    observation_df = pd.read_csv(observation_csv)
    observation_df.columns = observation_df.columns.str.strip()
    condition_df = pd.read_csv(condition_csv)
    condition_df.columns = condition_df.columns.str.strip()

    if "condition_concept_id" in condition_df.columns:
        condition_matches = condition_df["condition_concept_id"].isin(observation_df["qualifier_concept_id"])
        matching_observations = observation_df[observation_df["qualifier_concept_id"].isin(condition_df["condition_concept_id"])]

        condition_pairs = matching_observations[["person_id", "qualifier_concept_id", "value_as_number", "observation_source_value"]].rename(
            columns={"qualifier_concept_id": "concept_id", "value_as_number": "source_value", "observation_source_value": "definition"}
        )

        used_rows = observation_df["qualifier_concept_id"].isin(matching_observations["qualifier_concept_id"])
        observation_remaining = observation_df[~used_rows]
    else:
        condition_pairs = pd.DataFrame(columns=["person_id", "concept_id", "source_value", "definition"])
        observation_remaining = observation_df

    if "observation_concept_id" in observation_remaining.columns and "value_as_number" in observation_remaining.columns:
        observation_pairs = observation_remaining[["person_id", "observation_concept_id", "value_as_number", "observation_source_value"]].rename(
            columns={"observation_concept_id": "concept_id", "value_as_number": "source_value", "observation_source_value": "definition"}
        )
    else:
        observation_pairs = pd.DataFrame(columns=["person_id", "concept_id", "source_value", "definition"])

    measurement_df = pd.read_csv(measurement_csv)
    measurement_df.columns = measurement_df.columns.str.strip()

    if "measurement_concept_id" in measurement_df.columns and "value_as_number" in measurement_df.columns:
        measurement_pairs = measurement_df[["person_id", "measurement_concept_id", "value_as_number", "measurement_source_value"]].rename(
            columns={"measurement_concept_id": "concept_id", "value_as_number": "source_value", "measurement_source_value": "definition"}
        )
    else:
        measurement_pairs = pd.DataFrame(columns=["person_id", "concept_id", "source_value", "definition"])

    combined_table = pd.concat([condition_pairs, observation_pairs, measurement_pairs], ignore_index=True)
    combined_table['concept_id'] = combined_table['concept_id'].astype(int)

    return combined_table

'''
def check_concept_table(combined_table, ignore_values):
    """
    Identifies concept_ids where source_value is missing or contains invalid dummy codes (ignore_values).
    Saves invalid concept_ids to a CSV file.

    Args:
        combined_table (pd.DataFrame): The processed clinical data table.
        ignore_values (list): A list of values to ignore (e.g., dummy codes like 555, 888, etc.).

    Returns:
        list: A list of concept_ids that contain missing or invalid source_value entries.
    """
    # Identify concept_ids where source_value is empty or one of the ignore values
    invalid_concepts = combined_table.groupby("concept_id")["source_value"].apply(lambda x: ((x.isna()) | (x == "") | (x.astype(str).isin(map(str, ignore_values)))).any())
    invalid_concept_ids = invalid_concepts[invalid_concepts].index.tolist()
    #pd.DataFrame(invalid_concept_ids, columns=["concept_id"]).to_csv("invalid_concepts.csv", index=False)

    return invalid_concept_ids
'''
def check_concept_table(combined_table, ignore_values, min_valid_count=16):
    """
    Identifies concept_ids where source_value is missing or contains invalid dummy codes (ignore_values).
    A concept_id is considered invalid only if the number of valid samples is below min_valid_count.

    Args:
        combined_table (pd.DataFrame): The processed clinical data table.
        ignore_values (list): A list of values to ignore (e.g., dummy codes like 555, 888, etc.).
        min_valid_count (int, optional): Minimum valid samples required for a concept_id to be considered valid. Defaults to 16.

    Returns:
        list: A list of concept_ids that contain too many invalid source_value entries.
    """
    # Convert ignore values to strings for comparison
    ignore_values = set(map(str, ignore_values))

    # Define a mask for invalid values
    is_invalid = combined_table["source_value"].isna() | (combined_table["source_value"] == "") | (combined_table["source_value"].astype(str).isin(ignore_values))

    # Count valid and invalid entries per concept_id
    concept_counts = combined_table.groupby("concept_id")["source_value"].count()
    invalid_counts = combined_table[is_invalid].groupby("concept_id")["source_value"].count()

    # Fill missing values with 0 for concept_ids that have no invalid values
    invalid_counts = invalid_counts.reindex(concept_counts.index, fill_value=0)

    # Compute valid counts
    valid_counts = concept_counts - invalid_counts

    # Identify concept_ids with too few valid samples
    invalid_concept_ids = valid_counts[valid_counts < min_valid_count].index.tolist()

    return invalid_concept_ids


def filter_clinical_table(table, concept_id, ignore_values):
    """
    Filters a table by a specific concept_id and removes rows with source_value in the ignore list.

    Args:
        table (pd.DataFrame): The input table.
        concept_id (float): The concept_id to filter.
        ignore_values (list): A list of source_value values to ignore.

    Returns:
        pd.DataFrame: The filtered table.
    """
    if concept_id <= 0:
        return table
    # check if concept_id is likely a string or contains non-numeric characters
    try:
        float(concept_id)
    except ValueError:
        print("Invalid concept ID: %s"%concept_id)

    filtered_table = table[
        (table['concept_id'] == float(concept_id)) &
        (~table['source_value'].isin(ignore_values)) &
        (table['source_value'].notna())
    ]

    return filtered_table


def get_aireadi_setting(patient_id_recommended_split, aireadi_patient_id_dict, split='train', device_model_name='All', location='All', pre_patient_cohort='All'):
    # split: 'train', 'val', 'test'
    # device_model_name: 'Spectralis', 'Maestro2', 'Triton', 'Cirrus', 'All'
    # location: 'Macula', 'Disc', 'Macula all 6', 'Macula all', 'Macula 12', 'All'
    # pre_patient_cohort: 'All_have', 'Spectralis', 'Maestro2', 'Triton', 'Cirrus', 'All',

    spectralis_macula = ('Spectralis', 'Macula')
    spectralis_optic_disc = ('Spectralis', 'Optic Disc')

    maestro_macula = ('Maestro2', 'Macula')
    maestro_macula_6 = ('Maestro2', 'Macula, 6 x 6')
    maestro_wide_field = ('Maestro2', 'Wide Field')

    triton_macula_6 = ('Triton', 'Macula, 6 x 6')
    triton_macula_12 = ('Triton', 'Macula, 12 x 12')
    triton_optic_disc = ('Triton', 'Optic Disc')

    cirrus_macula = ('Cirrus', 'Macula')
    cirrus_macula_6 = ('Cirrus', 'Macula, 6 x 6')
    cirrus_optic_disc = ('Cirrus', 'Optic Disc')
    cirrus_optic_disc_6 = ('Cirrus', 'Optic Disc, 6 x 6')

    ## Additional device for CFP
    eidon_mosaic = ('Eidon', 'Mosaic')
    eidon_macula = ('Eidon', 'Macula')
    eidon_nasal = ('Eidon', 'Nasal')
    eidon_temporal = ('Eidon', 'Temporal Periphery')

    aurora_macula = ('Aurora', 'Macula or Optic Disc')
    # aurora_optic_disc = ('Aurora', 'Optic Disc')

    condition_list = []

    if location == 'Macula':
        if device_model_name == 'Spectralis':
            condition_list.append(spectralis_macula)
        elif device_model_name == 'Maestro2':
            condition_list.append(maestro_macula)
        elif device_model_name == 'Triton':
            condition_list.append(triton_macula)
        elif device_model_name == 'Cirrus':
            condition_list.append(cirrus_macula)
        elif device_model_name == 'Eidon':
            condition_list.append(eidon_macula)
        #elif device_model_name == 'Aurora':
        #    condition_list.append(aurora_macula)
        elif device_model_name == 'All':
            condition_list.append(spectralis_macula)
            condition_list.append(maestro_macula)
            condition_list.append(triton_macula)
            condition_list.append(cirrus_macula)
            condition_list.append(eidon_macula)
            condition_list.append(aurora_macula_or_disc)
    elif location == 'Macula, 6 x 6':
        if device_model_name == 'Maestro2':
            condition_list.append(maestro_macula_6)
        elif device_model_name == 'Triton':
            condition_list.append(triton_macula_6)
        elif device_model_name == 'Cirrus':
            condition_list.append(cirrus_macula_6)
        elif device_model_name == 'All':
            condition_list.append(maestro_macula_6)
            condition_list.append(triton_macula_6)
            condition_list.append(cirrus_macula_6)
    elif location == 'Macula, 12 x 12':
        if device_model_name == 'Triton':
            condition_list.append(triton_macula_12)
    elif location == 'Optic Disc':
        if device_model_name == 'Spectralis':
            condition_list.append(spectralis_optic_disc)
        elif device_model_name == 'Maestro2':
            condition_list.append(maestro_wide_field)
        elif device_model_name == 'Triton':
            condition_list.append(triton_optic_disc)
        elif device_model_name == 'Cirrus':
            condition_list.append(cirrus_optic_disc)
        elif device_model_name == 'Aurora':
            condition_list.append(aurora_macula_or_disc)
        elif device_model_name == 'All':
            condition_list.append(spectralis_optic_disc)
            condition_list.append(maestro_wide_field)
            condition_list.append(triton_optic_disc)
            condition_list.append(cirrus_optic_disc)
            #condition_list.append(aurora_macula_or_disc)
    elif location == 'Macula all 6':
        condition_list.append(maestro_macula)
        condition_list.append(triton_macula)
        condition_list.append(spectralis_macula)
        condition_list.append(maestro_macula_6)
        condition_list.append(cirrus_macula)
        condition_list.append(cirrus_macula_6)
    elif location == 'Optic Disc, 6 x 6':
        condition_list.append(cirrus_optic_disc_6)
    elif location == 'Mosaic':
        condition_list.append(eidon_mosaic)
    elif location == 'Nasal':
        condition_list.append(eidon_nasal)
    elif location == 'Temporal Periphery':
        condition_list.append(eidon_temporal)
    elif location == 'Wide Field':
        condition_list.append(maestro_wide_field)
    elif location == 'All':
        if device_model_name == 'Spectralis':
            condition_list.append(spectralis_macula)
            condition_list.append(spectralis_optic_disc)
        elif device_model_name == 'Maestro2':
            condition_list.append(maestro_macula)
            condition_list.append(maestro_macula_6)
            condition_list.append(maestro_wide_field)
        elif device_model_name == 'Triton':
            condition_list.append(triton_macula)
            condition_list.append(triton_macula_12)
            condition_list.append(triton_optic_disc)
        elif device_model_name == 'Cirrus':
            condition_list.append(cirrus_macula)
            condition_list.append(cirrus_macula_6)
            condition_list.append(cirrus_optic_disc)
            condition_list.append(cirrus_optic_disc_6)
        elif device_model_name == 'Eidon':
            condition_list.append(eidon_mosaic)
            condition_list.append(eidon_macula)
            condition_list.append(eidon_nasal)
            condition_list.append(eidon_temporal)
        elif device_model_name == 'Aurora':
            condition_list.append(aurora_macula_or_disc)
        elif device_model_name == 'All':
            condition_list.append(spectralis_macula)
            condition_list.append(maestro_macula)
            condition_list.append(triton_macula)
            condition_list.append(cirrus_macula)
            condition_list.append(maestro_macula_6)
            condition_list.append(cirrus_macula_6)
            condition_list.append(triton_macula_12)
            condition_list.append(maestro_wide_field)
            condition_list.append(triton_optic_disc)
            condition_list.append(spectralis_optic_disc)
            condition_list.append(cirrus_optic_disc)
            condition_list.append(cirrus_optic_disc_6)
            condition_list.append(eidon_mosaic)
            condition_list.append(eidon_macula)
            condition_list.append(eidon_nasal)
            condition_list.append(eidon_temporal)
            condition_list.append(aurora_macula_or_disc)
        else:
            raise ValueError('Unknown device_model_name')
    else:
        raise ValueError('Unknown location: %s'%location)

    if pre_patient_cohort == 'All_have':
        patient_list = aireadi_patient_id_dict['All_devices']
        # print('Number of patients in the split:', len(patient_list))

    elif pre_patient_cohort == 'Spectralis':
        patient_list = aireadi_patient_id_dict['Heidelberg']
    elif pre_patient_cohort == 'Maestro2':
        patient_list = aireadi_patient_id_dict['Maestro']
    elif pre_patient_cohort == 'Triton':
        patient_list = aireadi_patient_id_dict['Triton']
    elif pre_patient_cohort == 'Cirrus':
        patient_list = aireadi_patient_id_dict['Zeiss']
    elif pre_patient_cohort == 'Eidon':
        patient_list = aireadi_patient_id_dict['iCare']
    elif pre_patient_cohort == 'Aurora':
        patient_list = aireadi_patient_id_dict['Optomed']
    elif pre_patient_cohort == 'All':
        patient_list = aireadi_patient_id_dict['All']
    else:
        raise ValueError('Unknown pre_patient_cohort')
    if split.lower() == 'all':
        return condition_list, patient_list
    else:
        splited_patient_list = []
        for patient_id in patient_list:
            if patient_id_recommended_split[patient_id] == split:
                splited_patient_list.append(patient_id)
        print('Number of patients in the split:', split, len(splited_patient_list))
        return condition_list, splited_patient_list


def get_patient(dataset_dir: str, used_aireadi_patient_dict, data_type: str, concept_table):
    """
    Load patients and their corresponding information based on concept_id and data_type.

    Args:
        dataset_dir (str): Directory containing the dataset.
        used_aireadi_patient_dict (dict): Dictionary with patient data.
        data_type (str): The data type to process, e.g., 'oct' or 'cfp'.
        concept_table (pd.DataFrame): Table containing person_id, concept_id, and source_value.

    Returns:
        tuple: patients (dict), visits_dict (dict), mapping_patient2visit (dict)
    """
    if data_type not in ['oct', 'cfp', 'octa']:
        raise ValueError("Invalid data_type. Please choose either 'oct', 'octa', or 'cfp'.")

    patients = {}
    visits_dict = {}
    mapping_patient2visit = {}
    visit_idx = 0

    for patient_id, patient_data in used_aireadi_patient_dict.items():
        # Filter the concept_table for the current patient_id
        patient_info = concept_table[concept_table["person_id"] == patient_id]
        if patient_info.empty:
            continue  # Skip if no relevant data in concept_table for this patient_id

        source_values = patient_info["source_value"].tolist()

        # Validate data_type existence in patient_data
        if data_type not in patient_data:
            continue  # Skip if the specified data_type is not present

        data_list = patient_data[data_type]
        patient_metadata = patient_data['metadata']
        label = patient_metadata['label']
        #label = 1 if label > 1 else 0
        class_name = patient_metadata['study_group']

        for data_dict in data_list:
            data_file = os.path.join(dataset_dir,data_dict['file'].lstrip('/'))
            data_metadata = data_dict['metadata']

            if patient_id not in patients:
                patients[patient_id] = {
                    'class_idx': label,
                    'class': class_name,
                    'frames': [data_file],
                    'pat_metadata': patient_metadata,
                    f'{data_type}_metadata': [data_metadata],
                    'pat_id': patient_id,
                    'source_values': source_values,  # Store source_value(s)
                }
                mapping_patient2visit[patient_id] = [visit_idx]
                visits_dict[visit_idx] = {
                    'class_idx': label,
                    'class': class_name,
                    'frames': data_file,
                    'pat_metadata': patient_metadata,
                    f'{data_type}_metadata': [data_metadata],
                    'pat_id': patient_id,
                    'source_values': source_values,  # Store source_value(s)
                }
                visit_idx += 1
            else:
                patients[patient_id]['frames'].append(data_file)
                patients[patient_id][f'{data_type}_metadata'].append(data_metadata)

                mapping_patient2visit[patient_id].append(visit_idx)
                visits_dict[visit_idx] = {
                    'class_idx': label,
                    'class': class_name,
                    'frames': data_file,
                    'pat_metadata': patient_metadata,
                    f'{data_type}_metadata': [data_metadata],
                    'pat_id': patient_id,
                    'source_values': source_values,  # Store source_value(s)
                }
                visit_idx += 1

    return patients, visits_dict, mapping_patient2visit


def get_patient_dict(data_type, octa_enface_imaging, participants_df, manifest_df, label_mapping, verbose=False):
    patient_dict = {}

    for idx, row in participants_df.iterrows():
        patient_id = row['participant_id']
        recommended_split = row['recommended_split']
        study_group = row['study_group']
        age = row['age']
        label = label_mapping[study_group]

        metadata_dict = {
            'recommended_split': recommended_split,
            'study_group': study_group,
            'age': age,
            'label': label,
        }

        # Initialize patient dictionary structure
        patient_dict[patient_id] = {
            'metadata': metadata_dict,
            'oct': [],
            'cfp': [],
            'octa': [],
            'oct_stats': {},
            'cfp_stats': {},
            'octa_stats': {}
        }

        if data_type == "oct":
            # Process OCT data
            if row['retinal_oct']:
                oct_files = manifest_df[manifest_df['participant_id'] == patient_id]
                process_oct_data(patient_dict, patient_id, oct_files, verbose)
        elif data_type == "cfp":
            # Process CFP data
            if row['retinal_photography']:
                cfp_files = manifest_df[manifest_df['participant_id'] == patient_id]
                process_cfp_data(patient_dict, patient_id, cfp_files, verbose)
        elif data_type == "octa":
            # Process OCTA data
            if row['retinal_octa']:
                octa_files = manifest_df[manifest_df['participant_id'] == patient_id]
                process_octa_data(patient_dict, patient_id, octa_files, octa_enface_imaging, verbose)

        '''
        # Determine availability of laterality
        has_L = any(item['metadata']['laterality'] == 'L' for item in patient_dict[patient_id]['oct'] + patient_dict[patient_id]['cfp'])
        has_R = any(item['metadata']['laterality'] == 'R' for item in patient_dict[patient_id]['oct'] + patient_dict[patient_id]['cfp'])

        if has_L and has_R:
            patient_dict[patient_id]['metadata']['avail_laterality'] = 'B'
        elif has_L:
            patient_dict[patient_id]['metadata']['avail_laterality'] = 'L'
        elif has_R:
            patient_dict[patient_id]['metadata']['avail_laterality'] = 'R'
        else:
            raise ValueError('No laterality found for patient: ', patient_id)
        '''
    return patient_dict


def process_oct_data(patient_dict, patient_id, oct_files, verbose=False):
    # Add OCT-specific data processing and statistics
    patient_dict[patient_id]['oct_stats'] = update_device_counts(oct_files)
    # Update with more detailed logic similar to your original code
    for oct_idx, oct_row in oct_files.iterrows():
        metadata = extract_oct_metadata(oct_row)
        patient_dict[patient_id]['oct'].append({'file': oct_row['filepath'], 'metadata': metadata})


def process_cfp_data(patient_dict, patient_id, cfp_files, verbose=False):
    # Add CFP-specific data processing and statistics
    patient_dict[patient_id]['cfp_stats'] = update_device_counts(cfp_files)
    # Update with more detailed logic similar to your original code
    for cfp_idx, cfp_row in cfp_files.iterrows():
        metadata = extract_cfp_metadata(cfp_row)
        patient_dict[patient_id]['cfp'].append({'file': cfp_row['filepath'], 'metadata': metadata})


def process_octa_data(patient_dict, patient_id, octa_files, octa_enface_imaging, verbose=False):
    # Add OCT-specific data processing and statistics
    patient_dict[patient_id]['octa_stats'] = update_device_counts(octa_files)

    for octa_idx, octa_row in octa_files.iterrows():
        metadata = extract_octa_metadata(octa_row)
        if octa_enface_imaging is None:
            patient_dict[patient_id]['octa'].append({'file': octa_row['flow_cube_file_path'], 'metadata': metadata})
        else:
            patient_dict[patient_id]['octa'].append({'file': metadata[f'{octa_enface_imaging}_file_path'], 'metadata': metadata})


def update_device_counts(files):
    # Initialize counters
    num_spectralis, num_spectralis_macula, num_spectralis_optic_disc = 0, 0, 0
    num_maestro, num_maestro_macula, num_maestro_macula_6, num_maestro_wide_field = 0, 0, 0, 0
    num_triton, num_triton_macula_6, num_triton_macula_12, num_triton_optic_disc = 0, 0, 0, 0
    num_cirrus, num_cirrus_macula, num_cirrus_optic_disc, num_cirrus_macula_6, num_cirrus_optic_disc_6 = 0, 0, 0, 0, 0
    num_eidon, num_eidon_mosaic, num_eidon_macula, num_eidon_nasal, num_eidon_temporal = 0, 0, 0, 0, 0
    num_aurora, num_aurora_macula, num_aurora_optic_disc = 0, 0, 0
    # Process each row in the data
    for idx, row in files.iterrows():
        if row['manufacturer'] == 'Heidelberg':
            num_spectralis += 1
            if row['anatomic_region'] == 'Macula':
                num_spectralis_macula += 1
            elif row['anatomic_region'] == 'Optic Disc':
                num_spectralis_optic_disc += 1
        elif row.get('manufacturers_model_name') == 'Maestro2':
            num_maestro += 1
            if row['anatomic_region'].startswith('Macula, 6'):
                num_maestro_macula_6 += 1
            elif row['anatomic_region'] == 'Macula':
                num_maestro_macula += 1
            elif row['anatomic_region'] == 'Wide Field':
                num_maestro_wide_field += 1
        elif row.get('manufacturers_model_name') == 'Triton':
            num_triton += 1
            if row['anatomic_region'].startswith('Macula, 6'):
                num_triton_macula_6 += 1
            elif row['anatomic_region'].startswith('Macula, 12'):
                num_triton_macula_12 += 1
            elif row['anatomic_region'] == 'Optic Disc':
                num_triton_optic_disc += 1
        elif row['manufacturer'] == 'Zeiss':
            num_cirrus += 1
            if row['anatomic_region'] == 'Macula':
                num_cirrus_macula += 1
            elif row['anatomic_region'] == 'Optic Disc':
                num_cirrus_optic_disc += 1
            elif row['anatomic_region'].startswith('Macula, 6'):
                num_cirrus_macula_6 += 1
            elif row['anatomic_region'].startswith('Optic Disc, 6'):
                num_cirrus_optic_disc_6 += 1
        elif row['manufacturer'] == 'iCare':
            num_eidon += 1
            if row['anatomic_region'] == 'Mozaic':
                num_eidon_mosaic += 1
            elif row['anatomic_region'] == 'Macula':
                num_eidon_macula += 1
            elif row['anatomic_region'] == 'Nasal':
                num_eidon_nasal += 1
            elif row['anatomic_region'] == 'Temporal Periphery':
                num_eidon_temporal += 1
        elif row['manufacturer'] == 'Optomed':
            num_aurora += 1
            if row['anatomic_region'] == 'Macula':
                num_aurora_macula += 1
            elif row['anatomic_region'] == 'Optic Disc':
                num_aurora_optic_disc += 1

    # Return results as a dictionary
    return {
        "num_spectralis": num_spectralis,
        "num_spectralis_macula": num_spectralis_macula,
        "num_spectralis_optic_disc": num_spectralis_optic_disc,
        "num_maestro": num_maestro,
        "num_maestro_macula": num_maestro_macula,
        "num_maestro_macula_6": num_maestro_macula_6,
        "num_maestro_wide_field": num_maestro_wide_field,
        "num_triton": num_triton,
        "num_triton_macula_6": num_triton_macula_6,
        "num_triton_macula_12": num_triton_macula_12,
        "num_triton_optic_disc": num_triton_optic_disc,
        "num_cirrus": num_cirrus,
        "num_cirrus_macula": num_cirrus_macula,
        "num_cirrus_optic_disc": num_cirrus_optic_disc,
        "num_cirrus_macula_6": num_cirrus_macula_6,
        "num_cirrus_optic_disc_6": num_cirrus_optic_disc_6
    }


def extract_oct_metadata(oct_row):
    # Extract and return metadata for OCT files
    return {
        'imaging': oct_row['imaging'],
        'anatomic_region': oct_row['anatomic_region'],
        'manufacturer': oct_row['manufacturer'],
        'manufacturers_model_name': oct_row['manufacturers_model_name'],
        'filepath': oct_row['filepath'],
        'sop_instance_uid': oct_row['sop_instance_uid'],
        'resolution': (oct_row['number_of_frames'], oct_row['height'], oct_row['width']),
        'laterality': oct_row['laterality'],
    }


def extract_cfp_metadata(cfp_row):
    # Extract and return metadata for CFP files
    return {
        'imaging': cfp_row['imaging'],
        'anatomic_region': cfp_row['anatomic_region'],
        'manufacturer': cfp_row['manufacturer'],
        'manufacturers_model_name': cfp_row['manufacturers_model_name'],
        'filepath': cfp_row['filepath'],
        'sop_instance_uid': cfp_row['sop_instance_uid'],
        'resolution': (cfp_row['color_channel_dimension'], cfp_row['height'], cfp_row['width']),
        'laterality': cfp_row['laterality'],
    }



def extract_octa_metadata(octa_row):
    # Extract and return metadata for OCT files
    return {
        'imaging': octa_row['imaging'],
        'anatomic_region': octa_row['anatomic_region'],
        'manufacturer': octa_row['manufacturer'],
        'manufacturers_model_name': octa_row['manufacturers_model_name'],
        'flow_cube_filepath': octa_row['flow_cube_file_path'],
        'sop_instance_uid': octa_row['flow_cube_sop_instance_uid'],
        'resolution': (octa_row['flow_cube_number_of_frames'], octa_row['flow_cube_height'], octa_row['flow_cube_width']),
        'laterality': octa_row['laterality'],
        'superficial_file_path': octa_row['associated_enface_1_file_path'],
        'deep_file_path': octa_row['associated_enface_2_file_path'],
        'choriocapillaris_file_path': octa_row['associated_enface_3_file_path'],
        'outer_retina_file_path': octa_row['associated_enface_4_file_path'],
    }


def filter_patient_dict(patient_dict, data_type, imaging, condition=None, pre_filtered_patient_id_list=None, verbose=False):
    """
    Filters patient dictionary based on conditions for OCT or CFP data.
    Filters specific imaging type here.

    Args:
        patient_dict (dict): Dictionary containing patient data.
        data_type (str): Type of data to filter ('oct', 'octa', 'cfp', or 'ir').
        condition (list of tuples): List of (manufacturers_model_name, anatomic_region) pairs to filter: e.g. condition=[('Spectralis', 'Macula')].
        pre_filtered_patient_id_list (list, optional): List of pre-filtered patient IDs to include.
        verbose (bool, optional): If True, prints filtering details.

    Returns:
        dict: Filtered subset of patient dictionary with updated statistics.
    """

    if condition is None:
        condition = []

    return_patient_dict = {}

    for patient_id, patient_info in patient_dict.items():
        if pre_filtered_patient_id_list is not None and patient_id not in pre_filtered_patient_id_list:
            continue

        # Choose the correct data key based on data_type
        data_list = patient_info.get(data_type, [])
        patient_metadata = patient_info['metadata']

        # Initialize counters dynamically
        stats = {
            'num_spectralis': 0,
            'num_spectralis_macula': 0,
            'num_spectralis_optic_disc': 0,
            'num_maestro': 0,
            'num_maestro_macula': 0,
            'num_maestro_macula_6': 0,
            'num_maestro_wide_field': 0,
            'num_triton': 0,
            'num_triton_macula_6': 0,
            'num_triton_macula_12': 0,
            'num_triton_optic_disc': 0,
            'num_cirrus': 0,
            'num_cirrus_macula': 0,
            'num_cirrus_macula_6': 0,
            'num_cirrus_optic_disc': 0,
            'num_cirrus_optic_disc_6': 0,
        }

        filtered_data_list = []

        for data_dict in data_list:
            metadata = data_dict['metadata']
            manufacturers_model_name = metadata.get('manufacturers_model_name', '')
            anatomic_region = metadata.get('anatomic_region', '')
            imaging_name = metadata.get('imaging', '')

            if (manufacturers_model_name, anatomic_region) in condition:
                # Check if imaging matches
                if (
                    (imaging_name == 'Color Photography' and imaging == 'cfp') or
                    (imaging_name == 'Infrared Reflectance' and imaging == 'ir') or
                    (imaging_name == 'Autofluorescence' and imaging == 'faf') or
                    (imaging_name == 'OCTA' and imaging == 'octa')or
                    (imaging_name == 'OCT' and imaging == 'oct')
                ):

                    filtered_data_list.append(data_dict)

                    # Update counters based on manufacturers_model_name and anatomic_region
                    if metadata['manufacturer'] == 'Heidelberg':
                        stats['num_spectralis'] += 1
                        if anatomic_region == 'Macula':
                            stats['num_spectralis_macula'] += 1
                        elif anatomic_region == 'Optic Disc':
                            stats['num_spectralis_optic_disc'] += 1
                    elif manufacturers_model_name == 'Maestro2':
                        stats['num_maestro'] += 1
                        if anatomic_region.startswith('Macula, 6'):
                            stats['num_maestro_macula_6'] += 1
                        elif anatomic_region == 'Macula':
                            stats['num_maestro_macula'] += 1
                        elif anatomic_region == 'Wide Field':
                            stats['num_maestro_wide_field'] += 1
                    elif manufacturers_model_name == 'Triton':
                        stats['num_triton'] += 1
                        if anatomic_region.startswith('Macula, 6'):
                            stats['num_triton_macula_6'] += 1
                        elif anatomic_region.startswith('Macula, 12'):
                            stats['num_triton_macula_12'] += 1
                        elif anatomic_region == 'Optic Disc':
                            stats['num_triton_optic_disc'] += 1
                    elif metadata['manufacturer'] == 'Zeiss':
                        stats['num_cirrus'] += 1
                        if anatomic_region == 'Macula':
                            stats['num_cirrus_macula'] += 1
                        elif anatomic_region.startswith('Macula, 6'):
                            stats['num_cirrus_macula_6'] += 1
                        elif anatomic_region == 'Optic Disc':
                            stats['num_cirrus_optic_disc'] += 1
                        elif anatomic_region.startswith('Optic Disc, 6'):
                            stats['num_cirrus_optic_disc_6'] += 1

        if len(filtered_data_list) > 0:
            return_patient_dict[patient_id] = {
                data_type: filtered_data_list,
                'metadata': patient_metadata,
                f'{data_type}_stats': stats,
            }
            if verbose:
                print(f"{patient_id}: {stats}")

    if verbose:
        print('Number of patients:', len(return_patient_dict))
    return return_patient_dict


def get_visit_idx(self, patient_id_list):
    visit_idx_list = []
    for patient_id in patient_id_list:
        visit_idx_list += self.mapping_patient2visit[patient_id]
    return visit_idx_list
