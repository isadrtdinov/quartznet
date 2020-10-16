def set_params():
    params = {
        # The most important parameter
        'random_seed': 270072,
        'verbose': True,

        # Wandb configs
        'wandb_project': 'quartznet',

        # Data location
        'data_root': 'data/wavs/',
        'metadata_file': 'data/metadata.csv',

        # Checkpoints
        'checkpoint_dir': 'checkpoints/',
        'checkpoint_template': 'checkpoints/quartznet{}.pt',
        'model_checkpoint': 'checkponts/quartznet1.pt',
        'load_model': False,

        # Data processing
        'valid_ratio': 0.2,
        'sample_rate': 22050,
        'num_mels': 128,
        'max_audio_length': 216000,
        'max_target_length': 200,

        # QuartzNet params:
        'num_blocks': 5, 'num_cellls': 5,
        'input_kernel': 33, 'input_channels': 256,
        'head_kernel': 87, 'head_channels': 512,
        'block_kernels': (33, 39, 51, 63, 75),
        'block_channels': (256, 256, 256, 512, 512),

        # Optimizer params:
        'lr': 1e-4, 'weight_decay': 1e-3,
        'batch_size': 64, 'num_epochs': 1,
    }

    return params
