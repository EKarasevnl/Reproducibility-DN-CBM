import os.path as osp
import torch
import torchvision
import clip
import matplotlib.pyplot as plt
import random
import os
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image, to_tensor
from dncbm import method_utils, arg_parser, config
from dncbm.utils import get_printable_class_name, common_init, get_probe_dataset, get_sae_ckpt
from sparse_autoencoder import SparseAutoencoder
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
output_dir = osp.join(os.getcwd(), 'scripts/visualization/improved_combined_colors')
os.makedirs(output_dir, exist_ok=True)

# Initialize arguments and common settings
parser = arg_parser.get_common_parser()
parser.add_argument("--which_ckpt", type=str, default='final')
args = parser.parse_args()
common_init(args, disable_make_dirs=True)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

# Load the dataset
model, preprocess = clip.load(args.img_enc_name[5:], device=args.device)
dataset = get_probe_dataset(args.probe_dataset, args.probe_split, args.probe_dataset_root_dir, preprocess_fn=None)

# Load method object
embeddings_path = osp.join(args.vocab_dir, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth")
vocab_txt_path = osp.join(args.vocab_dir, "clipdissect_20k.txt")
method_obj = method_utils.get_method(args.method_name, args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=True)


# Get original concepts and weights
all_concepts = method_obj.get_concepts()
all_concepts = all_concepts.to(args.device)
all_labels = method_obj.get_labels()
all_labels  = all_labels.to(args.device)
classifier_weights = method_obj.get_classifier_weights()
classifier_weights = classifier_weights.to(args.device)



# Load the SAE model
autoencoder_input_dim = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim, n_learned_features=n_learned_features, n_components=len(args.hook_points)).to(args.device)
autoencoder = get_sae_ckpt(args, autoencoder)

# Un-normalize transformation for images
un_normalize = torchvision.transforms.Normalize(
    mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
    std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
)

print(f'size of dataset {len(dataset)}')
num_images = 365
# # Select amount of random images

labels = [label for _, label in dataset]
random_indices, labels = train_test_split(range(len(dataset)), train_size=num_images, stratify=labels, random_state=args.seed)
relevant_images = [dataset[idx] for idx in random_indices]


accuracies_original_classes ={}
accuracies_grayscale_classes = {}
accuracies_inverted_classes = {}

overall_accuracies = {}
overall_accuracies['original'] = 0
overall_accuracies['grayscale'] = 0
overall_accuracies['inverted'] = 0

count_of_labels = {}
for i in range(len(relevant_images)):
    img, label = relevant_images[i]
    if label in count_of_labels:
        count_of_labels[label] += 1
    else:
        count_of_labels[label] = 1

# Define transformations
resize_transform = torchvision.transforms.Resize((224, 224))
center_crop_transform = torchvision.transforms.CenterCrop(224)
grayscale_transform = torchvision.transforms.Compose([
    resize_transform,
    center_crop_transform,
    torchvision.transforms.Grayscale(num_output_channels=3),  # Convert to grayscale (3 channel)
    torchvision.transforms.ToTensor(),  # Convert to a tensor
])

# Apply grayscale transformation to relevant images
grayscale_images = [
    (grayscale_transform(img), label) for img, label in relevant_images
]

# Convert tensors back to PIL images before normalization
grayscale_images = [
    (to_pil_image(img), label) for img, label in grayscale_images
]

# Normalize the grayscale images
grayscale_images = [
    (preprocess(img), label) for img, label in grayscale_images
]


# Inverted original image
def invert_image(image):
    return ImageOps.invert(image)


# Invert the original images
inverted_images = [
    (invert_image(center_crop_transform(resize_transform(img)) ), label) for img, label in relevant_images
]

inverted_images = [
    (invert_image(img), label) for img, label in relevant_images
]

# Normalize the inverted images
inverted_images = [
    (preprocess(img), label) for img, label in inverted_images
]

print(f'len of relevant images {len(relevant_images)}')
print(f'len of grayscale images {len(grayscale_images)}')
print(f'len of inverted images {len(inverted_images)}')



def generate_prediction(image_index, real_idx):
    model.eval()
    img, label = relevant_images[image_index]
    concepts = all_concepts[real_idx]
    topk=10

    ### Original image
    with torch.no_grad():
        contribs = concepts * classifier_weights
        pred_class_original = torch.argmax(contribs.sum(dim=1)).item()
        pred_class_name_original = get_printable_class_name(args.probe_dataset, pred_class_original)


    ### Grayscale image
    img_grayscale, label_grayscale = grayscale_images[image_index]
    img_tensor = img_grayscale.unsqueeze(0).to(device)

    with torch.no_grad():
        grayscale_features = model.encode_image(img_tensor).squeeze(0)
        grayscale_concepts, _ = autoencoder(grayscale_features.unsqueeze(0))
        grayscale_concepts = grayscale_concepts.view(-1)
        grayscale_contribs = grayscale_concepts * classifier_weights

        pred_class_grayscale = torch.argmax(grayscale_contribs.sum(dim=1)).item()
        pred_class_name_grayscale = get_printable_class_name(args.probe_dataset, pred_class_grayscale)




    ### Inverted image
    img_inv, label_inv = inverted_images[image_index]
    img_tensor = img_inv.unsqueeze(0).to(device)

    with torch.no_grad():
        inverted_features = model.encode_image(img_tensor).squeeze(0)
        inverted_concepts, _ = autoencoder(inverted_features.unsqueeze(0))
        inverted_concepts = inverted_concepts.view(-1)
        inverted_contribs = inverted_concepts * classifier_weights

        pred_class_inverted = torch.argmax(inverted_contribs.sum(dim=1)).item()
        pred_class_name_inverted = get_printable_class_name(args.probe_dataset, pred_class_inverted)
    
    return label,pred_class_original, pred_class_grayscale, pred_class_inverted, pred_class_name_original, pred_class_name_grayscale, pred_class_name_inverted
        

for idx,real_idx in tqdm(enumerate(random_indices)):

    print(f'idx {idx}')
    print(f'real_idx {real_idx}')

    label,pred_class_original, pred_class_grayscale, pred_class_inverted, pred_class_name_original, pred_class_name_grayscale, pred_class_name_inverted = generate_prediction(idx, real_idx)
    if pred_class_original == label:
        overall_accuracies['original'] += 1
        if pred_class_name_original in accuracies_original_classes:
            accuracies_original_classes[pred_class_name_original] += 1
        else:
            accuracies_original_classes[pred_class_name_original] = 1
    if pred_class_grayscale == label:
        overall_accuracies['grayscale'] += 1
        if pred_class_name_grayscale in accuracies_grayscale_classes:
            accuracies_grayscale_classes[pred_class_name_grayscale] += 1
        else:
            accuracies_grayscale_classes[pred_class_name_grayscale] = 1
    if pred_class_inverted == label:
        overall_accuracies['inverted'] += 1
        if pred_class_name_inverted in accuracies_inverted_classes:
            accuracies_inverted_classes[pred_class_name_inverted] += 1
        else:
            accuracies_inverted_classes[pred_class_name_inverted] = 1
    # print every 100 images
    if idx % 100 == 0:
        print(f'idx {idx}')
        print(f'real_idx {real_idx}')
        print(f'original {pred_class_name_original}')
        print(f'grayscale {pred_class_name_grayscale}')
        print(f'inverted {pred_class_name_inverted}')
        print('-----------------')

# divide accuracies first map labels to names
count_of_labels = {get_printable_class_name(args.probe_dataset, label): count for label, count in count_of_labels.items()}

print('count of labels')
print(count_of_labels)
for key in accuracies_original_classes:
    accuracies_original_classes[key] = accuracies_original_classes[key]/count_of_labels[key]
for key in accuracies_grayscale_classes:
    accuracies_grayscale_classes[key] = accuracies_grayscale_classes[key]/count_of_labels[key]
for key in accuracies_inverted_classes:
    accuracies_inverted_classes[key] = accuracies_inverted_classes[key]/count_of_labels[key]

for key in overall_accuracies:
    overall_accuracies[key] = overall_accuracies[key]/num_images

print('Total accuracies')
print(f'Original accuracy: {overall_accuracies["original"]}')
print(f'Grayscale accuracy: {overall_accuracies["grayscale"]}')
print(f'Inverted accuracy: {overall_accuracies["inverted"]}')


print('Class accuracies')
print('original')
print(accuracies_original_classes)
print('grayscale')
print(accuracies_grayscale_classes)
print('inverted')
print(accuracies_inverted_classes)
















