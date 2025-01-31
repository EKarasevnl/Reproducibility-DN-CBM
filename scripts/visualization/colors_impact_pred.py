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


print(f'size of dataset {len(dataset)}')
# Select amount of random images
num_images = 7300  
labels = [label for _, label in dataset]
random_indices, labels = train_test_split(range(len(dataset)), train_size=num_images, stratify=labels, random_state=args.seed)
#random_indices = [2,5,7,15]
# Load only the relevant images
relevant_images = [dataset[idx] for idx in random_indices]




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

notes = []

def generate_prediction(image_index,real_idx):
    #print(f"Image nr in sample: {image_index} - Real index: {real_idx}")
    correct_original = 0
    correct_grayscale = 0
    correct_inverted = 0
    model.eval()
    img,label = relevant_images[image_index]

    concteps = all_concepts[real_idx]

    topk = 20

    ### Original image
    with torch.no_grad():
        contribs = concteps * classifier_weights
        pred_class_original = torch.argmax(contribs.sum(dim=1)).item()
        top_strengths_original, top_indices_original = torch.topk(contribs[pred_class_original], k=topk)
        pred_class_name_original = get_printable_class_name(args.probe_dataset, pred_class_original)
        top_concepts_original = [method_obj.get_concept_name(idx)[0] for idx in top_indices_original]

        # print("New image -------------------")
        # print(f"gt_class_name: {get_printable_class_name(args.probe_dataset, int(label))}")
        # print(f"pred_class_name: {pred_class_name_original}")

        if pred_class_original == int(label):
            correct_original += 1
        

    ### Grayscale image
    img,label = grayscale_images[image_index]
    
    img_tensor = img.unsqueeze(0).to(device)
   
    with torch.no_grad():
        grayscale_features = model.encode_image(img_tensor).squeeze(0)

        grayscale_concepts, _ = autoencoder(grayscale_features.unsqueeze(0))
        grayscale_concepts = grayscale_concepts.view(-1) # (1,1,8192) -> (8192)
        grayscale_concepts = grayscale_concepts.squeeze(0)
        grayscale_contribs = grayscale_concepts * classifier_weights
        
        pred_class_grayscale = torch.argmax(grayscale_contribs.sum(dim=1)).item()
        top_strengths_grayscale, top_indices_grayscale = torch.topk(grayscale_contribs[pred_class_grayscale], k=topk)
        pred_class_name_grayscale = get_printable_class_name(args.probe_dataset, pred_class_grayscale)
        top_concepts_grayscale = [method_obj.get_concept_name(idx)[0] for idx in top_indices_grayscale]
        #print(f"pred_class_name: {pred_class_name_grayscale}")

        if pred_class_grayscale == int(label):
            correct_grayscale += 1

    ### Inverted image
    img,label = inverted_images[image_index]
    img_tensor = img.unsqueeze(0).to(device)
    with torch.no_grad():
        inverted_features = model.encode_image(img_tensor).squeeze(0)

        inverted_concepts, _ = autoencoder(inverted_features.unsqueeze(0))
        inverted_concepts = inverted_concepts.view(-1) # (1,1,8192) -> (8192)
        inverted_concepts = inverted_concepts.squeeze(0)
        inverted_contribs = inverted_concepts * classifier_weights

        pred_class_inverted = torch.argmax(inverted_contribs.sum(dim=1)).item()
        top_strengths_inverted, top_indices_inverted = torch.topk(inverted_contribs[pred_class_inverted], k=topk)
        pred_class_name_inverted = get_printable_class_name(args.probe_dataset, pred_class_inverted)
        top_concepts_inverted = [method_obj.get_concept_name(idx)[0] for idx in top_indices_inverted]
        #print(f"pred_class_name: {pred_class_name_inverted}")

        if pred_class_inverted == int(label):
            correct_inverted += 1

        notes.append({
            'info':{
                'gt_class_name': get_printable_class_name(args.probe_dataset, int(label)),
                'image_index': image_index,
                'real_idx': real_idx
            },
            'original': {
                'pred_class': pred_class_original,
                'pred_class_name': pred_class_name_original,
                'top_concepts': top_concepts_original,
                'top_strengths': top_strengths_original
            },
            'grayscale': {
                'pred_class': pred_class_grayscale,
                'pred_class_name': pred_class_name_grayscale,
                'top_concepts': top_concepts_grayscale,
                'top_strengths': top_strengths_grayscale
            },
            'inverted': {
                'pred_class': pred_class_inverted,
                'pred_class_name': pred_class_name_inverted,
                'top_concepts': top_concepts_inverted,
                'top_strengths': top_strengths_inverted
            }
        })



    return correct_original, correct_grayscale, correct_inverted




accuracy = {}
accuracy['original'] = 0
accuracy['grayscale'] = 0
accuracy['inverted'] = 0
for nr, real_idx in tqdm(enumerate(random_indices), total=len(random_indices), desc="Processing"):
    correct_original, correct_grayscale, correct_inverted = generate_prediction(nr,real_idx)
    accuracy['original'] += correct_original
    accuracy['grayscale'] += correct_grayscale
    accuracy['inverted'] += correct_inverted

print(f"Original accuracy: {accuracy['original'] / num_images}")
print(f"Grayscale accuracy: {accuracy['grayscale'] / num_images}")
print(f"Inverted accuracy: {accuracy['inverted'] / num_images}")


# Save notes as txt
notes_path = osp.join(output_dir, 'notes.txt')
with open(notes_path, 'w') as f:
    for note in notes:
        f.write(str(note))
        f.write('\n')








