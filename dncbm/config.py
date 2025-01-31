

autoencoder_input_dim_dict = {'clip_RN50_out': 1024,
                              'clip_ViT-B16_out': 512,  
                              'clip_ViT-L14_out': 768, }

data_dir_root = './data'
save_dir_root = './SAE'
probe_cs_save_dir_root = './probe'
vocab_dir = './vocab'
analysis_dir = './analysis'



probe_dataset_root_dir_dict = {
    "places365": "/gpfs/scratch1/shared/FACT_2025_group1",
    "imagenet": "/gpfs/scratch1/shared/FACT_2025_group1/tiny-imagenet-200/",
    "cifar10": "/gpfs/home2/pnair/fact-1/data",
    "cifar100": "/gpfs/home2/pnair/fact-1/data",
}

probe_dataset_nclasses_dict = {"places365": 365,
                              #'imagenet': 1000,
                               "imagenet": 200,
                               "cifar10": 10,
                               "cifar100": 100}
