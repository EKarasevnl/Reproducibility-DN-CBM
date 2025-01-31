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


# color dictionary
colors = {
    "red": 0,
    "blue": 0,
    "green": 0,
    "yellow": 0,
    "orange": 0,
    "purple": 0,
    "pink": 0,
    "brown": 0,
    "black": 0,
    "white": 0,
    "gray": 0,
    "cyan": 0,
    "magenta": 0,
    "beige": 0,
    "gold": 0,
    "silver": 0,
    "teal": 0,
    "navy": 0,
    "lime": 0,
    "turquoise": 0,
    "maroon": 0,
    "olive": 0,
    "violet": 0,
    "indigo": 0,
    "lavender": 0,
    "peach": 0,
    "amber": 0,
    "jade": 0,
    "plum": 0,
    "ruby": 0,
    "sapphire": 0,
    "ivory": 0,
    "coral": 0,
    "charcoal": 0,
    "noir":0
}


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
labels = [label for _, label in dataset]

print(f'size of dataset {len(dataset)}')
tran_concepts ={}
untran_concepts = {}
tran_concept_values = {}
untran_concept_values = {}
num_images = 7300
random_indices, _ = train_test_split(range(len(dataset)), train_size=num_images, stratify=labels, random_state=args.seed)

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

# Load method object
embeddings_path = osp.join(args.vocab_dir, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth")
vocab_txt_path = osp.join(args.vocab_dir, "clipdissect_20k.txt")
method_obj = method_utils.get_method(args.method_name, args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=True)
# Get original concepts
all_concepts = method_obj.get_concepts()
all_concepts = all_concepts.to(args.device)


# Load the SAE model
autoencoder_input_dim = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim, n_learned_features=n_learned_features, n_components=len(args.hook_points)).to(args.device)
autoencoder = get_sae_ckpt(args, autoencoder)

# Create output directory in the current path
output_dir = osp.join(os.getcwd(), 'scripts/visualization/transcending_images/')
output_dir_untran = osp.join(os.getcwd(), 'scripts/visualization/unique_concepts/')

os.makedirs(output_dir, exist_ok=True)

# Un-normalize transformation for images
un_normalize = torchvision.transforms.Normalize(
    mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
    std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
)

def split_concepts(concepts, max_length=40):
    """Split concepts into multiple lines to fit within the plot title."""
    lines = []
    current_line = []
    current_length = 0
    for concept in concepts:
        if current_length + len(concept) + 2 > max_length:  # +2 for ", "
            lines.append(", ".join(current_line))
            current_line = [concept]
            current_length = len(concept)
        else:
            current_line.append(concept)
            current_length += len(concept) + 2  # +2 for ", "
    if current_line:
        lines.append(", ".join(current_line))
    return "\n".join(lines)

# Load the model for concept extraction
model.eval()

count_of_transcending = 0


for i, (img, label) in enumerate(relevant_images):
    img_idx = random_indices[i]

    ############################
    # Reuse precomputed concepts for original images
    original_concept_strengths = all_concepts[img_idx]
    topk = 10  # Number of top concepts to retrieve
    original_top_strengths, original_top_indices = torch.topk(original_concept_strengths, k=topk)
    original_concept_names = set(method_obj.get_concept_name(concept_idx)[0] for concept_idx in original_top_indices)

    ############################
    # Grayscale image concepts

    grayscale_img, _ = grayscale_images[i]
    img_tensor = grayscale_img.unsqueeze(0).to(device)
   
    with torch.no_grad():
        # Do CLIP to obtain features
        grayscale_features = model.encode_image(img_tensor).squeeze(0)
        # Autoencoder to obtain concepts
        grayscale_concepts, _ = autoencoder(grayscale_features.unsqueeze(0))
        grayscale_concepts = grayscale_concepts.view(-1) # (1,1,8192) -> (8192)
        grayscale_concepts = grayscale_concepts.squeeze(0)
    # Get top concepts
    grayscale_top_strengths, grayscale_top_indices = torch.topk(grayscale_concepts, k=topk)
    # Get concept names
    grayscale_concept_names = set(method_obj.get_concept_name(concept_idx)[0] for concept_idx in grayscale_top_indices)

    ############################
    # Inverted colormap image concepts
    img_tensor = to_tensor(img).unsqueeze(0).to(device)
    img_tensor = resize_transform(img_tensor)
    inverted_img = ImageOps.invert(to_pil_image(img_tensor.squeeze(0)))
    inverted_img_tensor = preprocess(inverted_img).unsqueeze(0).to(device)
    with torch.no_grad():
        inverted_features = model.encode_image(inverted_img_tensor).squeeze(0)
        inverted_concepts, _ = autoencoder(inverted_features.unsqueeze(0))
        inverted_concepts = inverted_concepts.view(-1)
        inverted_concepts = inverted_concepts.squeeze(0)
    # Get top concepts
    inverted_top_strengths, inverted_top_indices = torch.topk(inverted_concepts, k=topk)
    # Get concept names
    inverted_concept_names = set(method_obj.get_concept_name(concept_idx)[0] for concept_idx in inverted_top_indices)

    ############################
    # Transcending color image concepts
    overlap_all_three = original_concept_names.intersection(grayscale_concept_names, inverted_concept_names)
    print(f'Image {img_idx} has {len(overlap_all_three)} overlapping concepts')
    print(f'Those concepts are {overlap_all_three}')
    if len(overlap_all_three) > 0:
        tran_concepts[img_idx] = overlap_all_three
        for concept in overlap_all_three:
            if concept not in tran_concept_values:
                tran_concept_values[concept] = 0  # Initialize to 0 if not already in the dictionary
            tran_concept_values[concept] += 1 

    # Unique concepts for original image
    unique_for_original = original_concept_names.difference(grayscale_concept_names, inverted_concept_names)
    print(f'Image {img_idx} has {len(unique_for_original)} unique concepts')
    print(f'Those concepts are {unique_for_original}')
    if len(unique_for_original) > 0:
        untran_concepts[img_idx] = unique_for_original
        for concept in unique_for_original:
            if concept not in untran_concept_values:
                untran_concept_values[concept] = 0  # Initialize to 0 if not already in the dictionary
            untran_concept_values[concept] += 1

    ############################
    # Save images if large overlap
    if len(overlap_all_three) > 5:
        count_of_transcending += 1
        # Combine the images
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Display the original image
        img = img_tensor.squeeze(0).cpu()
        img = to_pil_image(img)
        axs[0].imshow(img)
        axs[0].set_title(f"Original Image Index: {img_idx}\n Transcending concepts: {split_concepts(overlap_all_three)}", fontsize=10)
        axs[0].axis('off')

        # Display the grayscale image
        grayscale_img = un_normalize(grayscale_img.cpu())
        grayscale_img = to_pil_image(grayscale_img)
        axs[1].imshow(grayscale_img, cmap='gray')  # Use 'gray' colormap for grayscale image
        axs[1].axis('off')

        # Display the inverted colormap image
        inverted_img = un_normalize(inverted_img_tensor.squeeze(0).cpu())
        inverted_img = to_pil_image(inverted_img)
        axs[2].imshow(inverted_img)
        axs[2].axis('off')

        # Save the combined image
        output_path = osp.join(output_dir, f'combined_image_{img_idx}_concepts.png')
        plt.savefig(output_path)
        plt.close(fig)
    if len(unique_for_original) > 5:
        # Combine the images
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Display the original image
        img = img_tensor.squeeze(0).cpu()
        img = to_pil_image(img)
        axs[0].imshow(img)
        axs[0].set_title(f"Original Image Index: {img_idx}\n Unique concepts: {split_concepts(unique_for_original)}", fontsize=10)
        axs[0].axis('off')

        # Display the grayscale image
        grayscale_img = un_normalize(grayscale_img.cpu())
        grayscale_img = to_pil_image(grayscale_img)
        axs[1].imshow(grayscale_img, cmap='gray')  # Use 'gray' colormap for grayscale image
        axs[1].axis('off')

        # Display the inverted colormap image
        inverted_img = un_normalize(inverted_img_tensor.squeeze(0).cpu())
        inverted_img = to_pil_image(inverted_img)
        axs[2].imshow(inverted_img)
        axs[2].axis('off')

        # Save the combined image
        output_path = osp.join(output_dir_untran, f'combined_image_{img_idx}_unique_concepts.png')
        plt.savefig(output_path)



# save the concepts and their quantities descending order
tran_concept_values = dict(sorted(tran_concept_values.items(), key=lambda item: item[1], reverse=True))
with open(osp.join(output_dir, 'concept_quantities.txt'), 'w') as f:
    for concept, quantity in tran_concept_values.items():
        f.write(f"{concept}: {quantity}\n")
# save concepts for images
with open(osp.join(output_dir, 'image_concepts.txt'), 'w') as f:
    for img_idx, concepts in tran_concepts.items():
        f.write(f"Image {img_idx}: {concepts}\n")

# save the concepts and their quantities descending order
untran_concept_values = dict(sorted(untran_concept_values.items(), key=lambda item: item[1], reverse=True))
with open(osp.join(output_dir, 'unique_concept_quantities.txt'), 'w') as f:
    for concept, quantity in untran_concept_values.items():
        f.write(f"{concept}: {quantity}\n")
# save concepts for images
with open(osp.join(output_dir, 'image_unique_concepts.txt'), 'w') as f:
    for img_idx, concepts in untran_concepts.items():
        f.write(f"Image {img_idx}: {concepts}\n")

print(f'Number of transcending images: {count_of_transcending}')
print(f'Proportion of transcending images: {count_of_transcending/num_images}')




