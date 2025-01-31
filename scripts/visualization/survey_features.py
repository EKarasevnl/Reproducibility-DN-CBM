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
from time import time
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

start_time = time()
# Initialize arguments and common settings
parser = arg_parser.get_common_parser()
parser.add_argument("--which_ckpt", type=str, default='final')
args = parser.parse_args()
common_init(args, disable_make_dirs=True)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device



# # Load the dataset
model, preprocess = clip.load(args.img_enc_name[5:], device=args.device)
dataset = get_probe_dataset(args.probe_dataset, args.probe_split, args.probe_dataset_root_dir, preprocess_fn=preprocess)
# labels = [label for _, label in dataset]


# Load method object
embeddings_path = osp.join(args.vocab_dir, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth")
vocab_txt_path = osp.join(args.vocab_dir, "clipdissect_20k.txt")

method_obj = method_utils.get_method(args.method_name, args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=True)


# Un-normalize transformation for images
un_normalize = torchvision.transforms.Normalize(
    mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
    std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
)

def show_imgs_per_word():
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    all_concepts = []
    for images, _ in dataloader:

        images = images.to(device)
        image_features = model.encode_image(images).detach().cpu()
        all_concepts.append(image_features)
    all_concepts = torch.cat(all_concepts)
    all_concepts = all_concepts.to(device)
    print(all_concepts.shape)

    ### Cosine similarity
    all_embeddings = torch.cat(method_obj.all_embeddings).to(device)
    all_embeddings = F.normalize(all_embeddings, dim=1)  # Normalize along the embedding dimension
    all_concepts = F.normalize(all_concepts, dim=1)  # Normalize along the embedding dimension
    
    all_embeddings = all_embeddings.to(torch.float32)
    all_concepts = all_concepts.to(torch.float32)

    # Compute cosine similarity 
    # `cosine_similarity` will have shape (num_images, vocab_size)
    cosine_similarity = torch.matmul(all_embeddings, all_concepts.T)

    print(f"all_embeddings: {all_embeddings.shape}")
    print(f"all_concepts: {all_concepts.shape}")
    print(f"cosine similarity: {cosine_similarity.shape}")
    # Example: get the top 5 most similar words for the first image

    cols = 4  
    rows = 3

    top_k = 12

    #words_idx = [69, 169, 1069, 7695] # Indices s
    words_idx = [982]
    for word_idx in tqdm(words_idx): # indices should correspond to indices in vocab file

        word = method_obj.vocab_txt_all[0][word_idx]
        top_indices = torch.topk(cosine_similarity[word_idx], top_k).indices
        print(f"Top {top_k} images for word {word} ({word_idx}): {top_indices}")
        
        # Create a figure with a grid of subplots
        fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 5))
        axs = axs.flatten()  # Flatten to easily index into the grid
        
        for i, idx in enumerate(top_indices):
            img = dataset[idx][0]


            if not isinstance(img, Image.Image):
                
                img = un_normalize(img)
                img = to_pil_image(img)
                

            axs[i].imshow(img)
            axs[i].set_title(f"Image {i+1}")
            axs[i].axis('off')
        
        # Turn off any unused subplots
        for i in range(len(top_indices), len(axs)):
            axs[i].axis('off')
        
        plt.suptitle(f"Top {top_k} Images for word: {word}")
        
        # Save the combined image
        output_dir = "scripts/visualization/survey_features"
        os.makedirs(output_dir, exist_ok=True)
        output_path = osp.join(output_dir, f'top_images_{args.probe_dataset}_for_word_{word_idx}_{word}.png')
        plt.savefig(output_path)
        plt.show()

    return None

show_imgs_per_word()

