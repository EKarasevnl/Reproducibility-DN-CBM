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
# Select amount of random images
num_images = 7300
# _, random_indices = train_test_split(range(len(dataset)), train_size=num_images, stratify=labels, random_state=args.seed)
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
output_dir = osp.join(os.getcwd(), 'scripts/visualization/improved_combined_colors')
os.makedirs(output_dir, exist_ok=True)

# Un-normalize transformation for images
un_normalize = torchvision.transforms.Normalize(
    mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
    std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
)

average_concept_importance_original = {}
average_concept_importance_grayscale = {}
average_concept_importance_inverted = {}

average_rank_original = {}
average_rank_grayscale = {}
average_rank_inverted = {}

concepts_for_images = {}
concepts_for_images['original'] = {}
concepts_for_images['grayscale'] = {}
concepts_for_images['inverted'] = {}

concepts_within_top5={}
concepts_within_top5['original'] = {}
concepts_within_top5['grayscale'] = {}
concepts_within_top5['inverted'] = {}




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

# Colors for the concepts
colors_original = colors.copy()
colors_grayscale = colors.copy()
colors_inverted = colors.copy()


for i, (img, label) in enumerate(relevant_images):
    img_idx = random_indices[i]
    # Reuse precomputed concepts for original images
    original_concept_strengths = all_concepts[img_idx]
    topk = 20  # Number of top concepts to retrieve
    original_top_strengths, original_top_indices = torch.topk(original_concept_strengths, k=topk)
    original_concept_names = [method_obj.get_concept_name(concept_idx)[0] for concept_idx in original_top_indices]

    # Get rank of colors if in


    # combine concept names with strengths
    combined_original = [f"{concept_name} ({strength:.2f})" for concept_name, strength in zip(original_concept_names, original_top_strengths)]

    #Add concepts to the dictionary
    concepts_for_images['original'][img_idx] = combined_original


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
    grayscale_concept_names = [method_obj.get_concept_name(concept_idx)[0] for concept_idx in grayscale_top_indices]

    #Add concepts to the dictionary

    combined_grayscale = [f"{concept_name} ({strength:.2f})" for concept_name, strength in zip(grayscale_concept_names, grayscale_top_strengths)]
    concepts_for_images['grayscale'][img_idx] = combined_grayscale


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
    inverted_concept_names = [method_obj.get_concept_name(concept_idx)[0] for concept_idx in inverted_top_indices]



    #Add concepts to the dictionary
    combined_inverted = [f"{concept_name} ({strength:.2f})" for concept_name, strength in zip(inverted_concept_names, inverted_top_strengths)]
    concepts_for_images['inverted'][img_idx] = combined_inverted



    # Calculate use of colors in concepts
    test = []
    index_original, index_grayscale, index_inverted = 1, 1, 1
    for concept in original_concept_names:
        if concept in colors.keys():
            colors_original[concept] += 1
            # Get the rank of colors in the concepts
            if concept in average_rank_original.keys():
                average_rank_original[concept] += index_original
                average_concept_importance_original[concept] += original_top_strengths[index_original-1]
                if index_original <= 5:
                    if concept in concepts_within_top5['original'].keys():
                        concepts_within_top5['original'][concept] += 1
                    else:
                        concepts_within_top5['original'][concept] = 1
            else:
                average_rank_original[concept] = index_original
                average_concept_importance_original[concept] = original_top_strengths[index_original-1]
                test.append((concept, index_original))
        index_original += 1
    if i == 0:
        print(f"Original Image Concepts: {original_concept_names}")
        print(f"Rank of Colors in Original Image: {test}")
        print(f"Average Concept Importance in Original Image: {average_concept_importance_original}")
        print(f"Average Rank of Colors in Original Image: {average_rank_original}")
    # print(f"Original Image Concepts: {original_concept_names}")
    # print(f"Rank of Colors in Original Image: {test}")
        


    for concept in grayscale_concept_names:
        if concept in colors.keys():
            colors_grayscale[concept] += 1
            # Get the rank of colors in the concepts
            if concept in average_rank_grayscale.keys():
                average_rank_grayscale[concept] += index_grayscale
                average_concept_importance_grayscale[concept] += grayscale_top_strengths[index_grayscale-1]
                if index_grayscale <= 5:
                    if concept in concepts_within_top5['grayscale'].keys():
                        concepts_within_top5['grayscale'][concept] += 1
                    else:
                        concepts_within_top5['grayscale'][concept] = 1
            else:
                average_rank_grayscale[concept] = index_grayscale
                average_concept_importance_grayscale[concept] = grayscale_top_strengths[index_grayscale-1]
        index_grayscale += 1

    for concept in inverted_concept_names:
        if concept in colors.keys():
            colors_inverted[concept] += 1
            # Get the rank of colors in the concepts
            if concept in average_rank_inverted.keys():
                average_rank_inverted[concept] += index_inverted
                average_concept_importance_inverted[concept] += inverted_top_strengths[index_inverted-1]
                if index_inverted <= 5:
                    if concept in concepts_within_top5['inverted'].keys():
                        concepts_within_top5['inverted'][concept] += 1
                    else:
                        concepts_within_top5['inverted'][concept] = 1
            else:
                average_rank_inverted[concept] = index_inverted
                average_concept_importance_inverted[concept] = inverted_top_strengths[index_inverted-1]
        index_inverted += 1




### Total Colors
print(colors_original)
print(colors_grayscale)
print(colors_inverted)

### Average Rank of Colors
for key in average_rank_original.keys():
    average_rank_original[key] = average_rank_original[key] / colors_original[key]
    print(f"Average Rank of {key} in Original Image: {average_rank_original[key]}")
for key in average_rank_grayscale.keys():
    average_rank_grayscale[key] = average_rank_grayscale[key] / colors_grayscale[key]
    print(f"Average Rank of {key} in Grayscale Image: {average_rank_grayscale[key]}")
for key in average_rank_inverted.keys():
    average_rank_inverted[key] = average_rank_inverted[key] / colors_inverted[key]
    print(f"Average Rank of {key} in Inverted Colormap Image: {average_rank_inverted[key]}")

### Average Concept Importance
for key in average_concept_importance_original.keys():
    average_concept_importance_original[key] = average_concept_importance_original[key] / colors_original[key]
    print(f"Average Concept Importance of {key} in Original Image: {average_concept_importance_original[key]}")
for key in average_concept_importance_grayscale.keys():
    average_concept_importance_grayscale[key] = average_concept_importance_grayscale[key] / colors_grayscale[key]
    print(f"Average Concept Importance of {key} in Grayscale Image: {average_concept_importance_grayscale[key]}")
for key in average_concept_importance_inverted.keys():
    average_concept_importance_inverted[key] = average_concept_importance_inverted[key] / colors_inverted[key]
    print(f"Average Concept Importance of {key} in Inverted Colormap Image: {average_concept_importance_inverted[key]}")

for key in concepts_within_top5['original'].keys():
    concepts_within_top5['original'][key] = concepts_within_top5['original'][key] / colors_original[key]
    print(f"Concepts within Top 5 for {key} in Original Image: {concepts_within_top5['original'][key]}")
for key in concepts_within_top5['grayscale'].keys():
    concepts_within_top5['grayscale'][key] = concepts_within_top5['grayscale'][key] / colors_grayscale[key]
    print(f"Concepts within Top 5 for {key} in Grayscale Image: {concepts_within_top5['grayscale'][key]}")
for key in concepts_within_top5['inverted'].keys():
    concepts_within_top5['inverted'][key] = concepts_within_top5['inverted'][key] / colors_inverted[key]
    print(f"Concepts within Top 5 for {key} in Inverted Colormap Image: {concepts_within_top5['inverted'][key]}")
    

### Calculated Colors Count
# Original Image
total_original = sum(colors_original.values())

# Grayscale Image
total_grayscale = sum(colors_grayscale.values())

# Inverted Colormap Image
total_inverted = sum(colors_inverted.values())

print(f"Original Image Total Colors: {total_original}")
print(f"Grayscale Image Total Colors: {total_grayscale}")
print(f"Inverted Colormap Image Total Colors: {total_inverted}")


#Save the colors for the images
with open(osp.join(output_dir, 'colors_for_images.txt'), 'w') as f:
    f.write("Original Image Colors\n")
    f.write(str(colors_original))
    f.write("\n\n")
    f.write("Grayscale Image Colors\n")
    f.write(str(colors_grayscale))
    f.write("\n\n")
    f.write("Inverted Colormap Image Colors\n")
    f.write(str(colors_inverted))
    f.write("\n\n")

#Save the average rank of colors for the images
with open(osp.join(output_dir, 'average_rank_colors_for_images.txt'), 'w') as f:
    f.write("Average Rank of Colors in Original Image\n")
    f.write(str(average_rank_original))
    f.write("\n\n")
    f.write("Average Rank of Colors in Grayscale Image\n")
    f.write(str(average_rank_grayscale))
    f.write("\n\n")
    f.write("Average Rank of Colors in Inverted Colormap Image\n")
    f.write(str(average_rank_inverted))
    f.write("\n\n")

#Save the average concept importance for the images
with open(osp.join(output_dir, 'average_concept_importance_for_images.txt'), 'w') as f:
    f.write("Average Concept Importance in Original Image\n")
    f.write(str(average_concept_importance_original))
    f.write("\n\n")
    f.write("Average Concept Importance in Grayscale Image\n")
    f.write(str(average_concept_importance_grayscale))
    f.write("\n\n")
    f.write("Average Concept Importance in Inverted Colormap Image\n")
    f.write(str(average_concept_importance_inverted))
    f.write("\n\n")



#Save the concepts for the images
with open(osp.join(output_dir, 'concepts_for_images.txt'), 'w') as f:
    for img_idx in concepts_for_images['original']:
        f.write(f"Image Index: {img_idx}\n")
        f.write(f"Original Image Concepts: {concepts_for_images['original'][img_idx]}\n")
        f.write(f"Grayscale Image Concepts: {concepts_for_images['grayscale'][img_idx]}\n")
        f.write(f"Inverted Colormap Image Concepts: {concepts_for_images['inverted'][img_idx]}\n")
        f.write("\n\n")

#Save the concepts within top 5 for the images
with open(osp.join(output_dir, 'concepts_within_top5_for_images.txt'), 'w') as f:
    f.write("Original Image Concepts within Top 5\n")
    f.write(str(concepts_within_top5['original']))
    f.write("\n\n")
    f.write("Grayscale Image Concepts within Top 5\n")
    f.write(str(concepts_within_top5['grayscale']))
    f.write("\n\n")
    f.write("Inverted Colormap Image Concepts within Top 5\n")
    f.write(str(concepts_within_top5['inverted']))
    f.write("\n\n")
