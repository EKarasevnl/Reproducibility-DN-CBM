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
# Select amount of random images



#relevant_indices = [3335, 21518,26825,36355,35989,29675,29581,28761,28229,28003,23810,22533,21245,477,2949,3011,5296,6415]
relevant_indices = [3335,21518,2949,28761]
relevant_images = [dataset[i] for i in relevant_indices]




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

        # Top 10 concepts for original
        #top10_concepts_original = torch.topk(contribs.sum(dim=1), 10).indices
        #top10_concepts_original = [get_printable_class_name(args.probe_dataset, idx.item()) for idx in top10_concepts_original]
        original_top_strengths, original_top_indices = torch.topk(concepts, k=topk)
        original_concept_names = [method_obj.get_concept_name(concept_idx)[0] for concept_idx in original_top_indices]
        top10_concepts_original = split_concepts(original_concept_names)

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

        # Top 10 concepts for grayscale
        #top10_concepts_grayscale = torch.topk(grayscale_contribs.sum(dim=1), 10).indices
        # top10_concepts_grayscale = [get_printable_class_name(args.probe_dataset, idx.item()) for idx in top10_concepts_grayscale]
        top_strengths_grayscale, top_indices_grayscale = torch.topk(grayscale_concepts, k=topk)
        grayscale_concept_names = [method_obj.get_concept_name(concept_idx)[0] for concept_idx in top_indices_grayscale]

        top10_concepts_grayscale = split_concepts(grayscale_concept_names)

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

        # Top 10 concepts for inverted
        #top10_concepts_inverted = torch.topk(inverted_contribs.sum(dim=1), 10).indices
        #top10_concepts_inverted = [get_printable_class_name(args.probe_dataset, idx.item()) for idx in top10_concepts_inverted]
        
        top_strengths_inverted, top_indices_inverted = torch.topk(inverted_concepts, k=topk)
        inverted_concept_names = [method_obj.get_concept_name(concept_idx)[0] for concept_idx in top_indices_inverted]
        top10_concepts_inverted = split_concepts(inverted_concept_names)

    # Save the images
    ground_truth_label = f"Ground Truth: {get_printable_class_name(args.probe_dataset, int(label))}"

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), constrained_layout=True)
    fig.suptitle(ground_truth_label, fontsize=16, fontweight='bold',y=0.97)

    # Original image
    axes[0].imshow(img)
    axes[0].set_title(f"Original: {pred_class_name_original}", fontsize=14, pad=8)
    axes[0].axis('off')
    axes[0].text(0.5, -0.02, top10_concepts_original, fontsize=13, ha='center', va='top', transform=axes[0].transAxes)

    # Grayscale image
    axes[1].imshow(to_pil_image(un_normalize(img_grayscale)))
    axes[1].set_title(f"Grayscale: {pred_class_name_grayscale}", fontsize=14, pad=8)
    axes[1].axis('off')
    axes[1].text(0.5, -0.02, top10_concepts_grayscale, fontsize=13, ha='center', va='top', transform=axes[1].transAxes)

    # Inverted image
    axes[2].imshow(to_pil_image(un_normalize(img_inv)))
    axes[2].set_title(f"Inverted: {pred_class_name_inverted}", fontsize=14, pad=8)
    axes[2].axis('off')
    axes[2].text(0.5, -0.02, top10_concepts_inverted, fontsize=13, ha='center', va='top', transform=axes[2].transAxes)

    plt.show()

    # Save the images
    fig.savefig(osp.join(output_dir, f"image_{real_idx}.png"))
    print(f"Saved image_{real_idx}.png at {output_dir}")
    plt.close(fig)
    return None




for i in range(len(relevant_images)):
    generate_prediction(i,relevant_indices[i])

    














