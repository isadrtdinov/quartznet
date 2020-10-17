def set_params():
    params = {
        # The most important parameter
        'random_seed': 270072,

        # System params
        'verbose': True,
        'num_workers': 8,

        # Wandb params
        'wandb_project': 'quartznet',
        'num_examples': 5,

        # Data location
        'data_root': 'data/wavs/',
        'metadata_file': 'data/metadata.csv',

        # Checkpoints
        'checkpoint_dir': 'checkpoints/',
        'checkpoint_template': 'checkpoints/quartznet{}.pt',
        'model_checkpoint': 'checkpoints/quartznet30.pt',
        'load_model': True,

        # Data processing
        'valid_ratio': 0.2,
        'sample_rate': 22050,
        'num_mels': 128,
        'max_audio_length': 216000,
        'max_target_length': 200,

        # Augmentation params:
        'pitch_shift': 2.0, 'noise_scale': 0.005,
        'gain_db': (-10.0, 30.0), 'audio_scale': 0.15,

        # QuartzNet params:
        'num_blocks': 5, 'num_cells': 5,
        'input_kernel': 33, 'input_channels': 128,
        'head_kernel': 87, 'head_channels': 256,
        'block_kernels': (33, 39, 51, 63, 75),
        'block_channels': (128, 128, 128, 256, 256),
        'dropout_rate': 0.3,

        # Optimizer params:
        'lr': 3e-4, 'weight_decay': 1e-3,
        'batch_size': 208, 'num_epochs': 30,
        'start_epoch': 31,
    }

    return params
