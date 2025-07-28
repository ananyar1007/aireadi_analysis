import argparse
def get_args_parser():

    parser = argparse.ArgumentParser(
        "Image classification parser", add_help=False
    )

    parser.add_argument(
        "--patient_dataset_type",
        default="slice",
        type=str,
        choices=["slice", "center_slice", "volume"],
        help="patient dataset type",
    )
    parser.add_argument(
        "--imaging",
        default="oct",
        type=str,
        choices=["oct", "cfp", "ir", "octa"],
        help="imaging type",
    )
    parser.add_argument(
        "--manufacturers_model_name",
        default="Spectralis",
        type=str,
        choices=["Spectralis", "Maestro2", "Triton", "Cirrus", "Eidon", "All"],
        help="device type",
    )
    parser.add_argument(
        "--anatomic_region",
        default="Macula",
        type=str,
        help="anatomic region to process",
    )
    parser.add_argument(
        "--octa_enface_imaging",
        default=None,
        type=str,
        choices=["superficial", "deep", "outer_retina", "choriocapillaris", None],
        help="OCTA enface slab type",
    )

    parser.add_argument(
        "--concept_id", default=-1, type=int, help="anatomic region to process"
    )


    parser.add_argument(
        "--cache_rate",
        default=0.,
        type=float,
        help="Proportion of dataset to cache between epochs",
    )
    parser.add_argument(
        "--cfp_img_path",
        default=None,
        type=str,
        help="cfp image path",
    )
    parser.add_argument(
        "--ir_img_path",
        default=None,
        type=str,
        help="ir image path",
    )
    parser.add_argument('--clinical_data', type=eval, default='[]',
                                 help='Types of PE to include.')
    # Training parameters
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU")
    parser.add_argument(
        "--val_batch_size", default=16, type=int, help="Validation batch size"
    )
    parser.add_argument(
        "--epochs", default=50, type=int, help="Number of training epochs"
    )

    # Model parameters
    parser.add_argument("--input_size", default=224, type=int, help="Input image size")

    parser.add_argument(
        "--nb_classes",
        default=2,
        type=int,
        help="Number of classification categories",
    )

    # Optimizer & Learning Rate
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/path/to/data/", type=str, help="Dataset path"
    )

    # Device and computation settings
    parser.add_argument(
        "--device", default="cuda", help="Device to use for training/testing"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers", default=10, type=int, help="Number of data loading workers"
    )

    # Checkpointing & Logging
    parser.add_argument(
        "--output_dir", default="./output", help="Path to save model checkpoints"
    )
    parser.add_argument("--log_dir", default="./logs", help="Path for logging")

    # Evaluation mode
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")

    #label
    parser.add_argument(
        "--label", default="mhoccur_ca, Cancer (any type)", help="label to predict"
    )

    parser.add_argument(
        "--dataset_config_path", default="/path/to/data/", type=str, help="Dataset path"
    )
    parser.add_argument(
        "--dropout", default=0, type=float, help="dropout"
    ) 
    parser.add_argument(
        "--num_layers", default=2, type=int, help="number of layers"
    ) 
    
    parser.add_argument("--experiment_name", default="train", help="experiment name")
    parser.add_argument( "--img_type", default="cfp",type=str, help = "image modality")
    args = parser.parse_args()

    return parser