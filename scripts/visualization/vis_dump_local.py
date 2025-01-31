import sys
import os
import matplotlib.pyplot as plt
from dncbm.data_utils import probe_classnames
# Add the repository root to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
# Print the Python path
print("Python sys.path:", sys.path)

# Check if the 'clip' directory exists
print("Does 'clip' exist?:", os.path.exists("clip"))

# Try importing clip
try:
    import clip
    print("Clip module imported successfully!")
    print("Clip module path:", clip.__file__)
except ModuleNotFoundError as e:
    print("Error:", e)

import os.path as osp
import torch
import torchvision
import os
import clip

from dncbm import method_utils, arg_parser, config
from dncbm.visualization.drawing_functions import to_numpy_img, to_numpy
from dncbm.utils import get_printable_class_name, common_init, get_probe_dataset

parser = arg_parser.get_common_parser()
parser.add_argument("--which_ckpt", type=str, default='final')
args = parser.parse_args()
common_init(args, disable_make_dirs=True)

topk = 5 # number of text explanations per image
num_samples = 1000 # total number of plots to be done

embeddings_path = osp.join(args.vocab_dir,f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth" )
vocab_txt_path = osp.join(args.vocab_dir, "clipdissect_20k.txt")
print(f"Current working directory: {os.getcwd()}")

_, preprocess = clip.load(args.img_enc_name[5:], device=args.device)
dataset = get_probe_dataset(args.probe_dataset, args.probe_split, args.probe_dataset_root_dir, preprocess_fn=preprocess)
un_normalize = torchvision.transforms.Normalize((-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711), (1/0.26862954, 1/0.26130258, 1/0.27577711))

method_obj = method_utils.get_method(args.method_name, args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=True)

all_concepts = method_obj.get_concepts()
all_concepts = all_concepts.to(args.device)
all_labels = method_obj.get_labels()
all_labels = all_labels.to(args.device)

classifier_weights = method_obj.get_classifier_weights()
classifier_weights = classifier_weights.to(args.device)
data_dir = osp.join(config.analysis_dir, 'local_explanations', args.img_enc_name_for_saving, args.probe_dataset, 'data')

local_images = {}
local_class_name = {}
local_concepts = {}
local_img_idxs = {}

randperm = torch.randperm(len(all_labels))[:num_samples]
# Number of images to display
num_images_to_display = 5

# Prepare lists to store images and predicted classes
local_images = []
predicted_classes = []

for img_pos_idx, img_idx in enumerate(randperm):
    # if img_pos_idx >= num_images_to_display:
    #     break  # Stop after displaying the specified number of images

    concept_strengths = all_concepts[img_idx]
    contribs = concept_strengths * classifier_weights
    pred_class = torch.argmax(contribs.sum(dim=1)).item()
    print("Predicted class index:", pred_class)
    class_name = probe_classnames.sun397_classes[pred_class]

    print(f"Processing image {img_pos_idx + 1}/{num_samples} with index {img_idx} and predicted class name {class_name}")

    # Get the image and convert it to a numpy array
    img = dataset[img_idx][0].unsqueeze(0)
    img = to_numpy_img(un_normalize(img)[0])  # Assuming to_numpy_img and un_normalize are defined

    # Store the image and predicted class
    local_images.append(img)

    predicted_classes.append(class_name)

# Plot the images and their predicted classes
# fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 5))
# for ax, img, pred_class in zip(axes, local_images, predicted_classes):
#     ax.imshow(img)
#     ax.axis('off')  # Hide axes
#     ax.set_title(f'Predicted Class: {pred_class}')  # Display predicted class

# plt.tight_layout()
# plt.savefig("local_image_dump_sun397.png")
# os.makedirs(data_dir, exist_ok=True) 
# torch.save(local_class_name, osp.join(data_dir, f'local_class_name_{args.probe_split}.pt'))
# torch.save(local_images, osp.join(data_dir, f'local_images_{args.probe_split}.pt'))
# torch.save(local_concepts, osp.join(data_dir, f'local_concepts_{args.probe_split}.pt'))
# torch.save(local_img_idxs, osp.join(data_dir, f'local_img_idxs_{args.probe_split}.pt'))

        
