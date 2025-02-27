import os

# List of class folders to check
class_folders = [
    "n01784675", "n01855672", "n01882714", "n01910747", "n01917289", "n01944390", "n01945685", "n01950731", "n01983481", "n01984695", 
    "n02002724", "n02056570", "n02058221", "n02074367", "n02085620", "n02094433", "n02099601", "n02099712", "n02106662", "n02113799", 
    "n02123045", "n02123394", "n02124075", "n02125311", "n02129165", "n02132136", "n02165456", "n02190166", "n02206856", "n02226429", 
    "n02231487", "n02233338", "n02236044", "n02268443", "n02279972", "n02281406", "n02321529", "n02364673", "n02395406", "n02403003", 
    "n02410509", "n02415577", "n02423022", "n02437312", "n02480495", "n02481823", "n02486410", "n02504458", "n02509815", "n02666196", 
    "n02669723", "n02699494", "n02730930", "n02769748", "n02788148", "n02791270", "n02793495", "n02795169", "n02802426", "n02808440", 
    "n02814533", "n02814860", "n02815834", "n02823428", "n02837789", "n02841315", "n02843684", "n02883205", "n02892201", "n02906734", 
    "n02909870", "n02917067", "n02927161", "n02948072", "n02950826", "n02963159", "n02977058", "n02988304", "n02999410", "n03014705", 
    "n03026506", "n03042490", "n03085013", "n03089624", "n03100240", "n03126707", "n03160309", "n03179701", "n03201208", "n03250847", 
    "n03255030", "n03355925", "n03388043", "n03393912", "n03400231", "n03404251", "n03424325", "n03444034", "n03447447", "n03544143", 
    "n03584254", "n03599486", "n03617480", "n03637318", "n03649909", "n03662601", "n03670208", "n03706229", "n03733131", "n03763968", 
    "n03770439", "n03796401", "n03804744", "n03814639", "n03837869", "n03838899", "n03854065", "n03891332", "n03902125", "n03930313", 
    "n03937543", "n03970156", "n03976657", "n03977966", "n03980874", "n03983396", "n03992509", "n04008634", "n04023962"
]

# Root directory of the dataset
probe_dataset_root_dir = "/gpfs/scratch1/shared/FACT_2025_group1/tiny-imagenet-200/train"

# Check for missing folders
# missing_folders = [folder for folder in class_folders if not os.path.isdir(os.path.join(probe_dataset_root_dir, folder))]

# # Print missing folders
# if missing_folders:
#     print("Missing folders:")
#     for folder in missing_folders:
#         print(folder)
# else:
#     print("All folders are present.")

# List the number of folders present
print(f"Number of folders present: {len(os.listdir(probe_dataset_root_dir))}")
print(f"Folders present: {os.listdir(probe_dataset_root_dir)}")

