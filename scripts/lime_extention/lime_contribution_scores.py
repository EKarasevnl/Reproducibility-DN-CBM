import os.path as osp
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
import clip
from dncbm.utils import common_init, get_probe_dataset, get_sae_ckpt
from dncbm import arg_parser, method_utils
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
from sparse_autoencoder import SparseAutoencoder
from dncbm.utils import get_printable_class_name
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb

# segmentation_fn = lambda img: felzenszwalb(img, scale=100, sigma=0.8, min_size=50)
# segmentation_fn = lambda img: slic(img, n_segments=1000, compactness=8)

def generate_lime_explanation(image_index, args):
    """
    Generate LIME explanations for the image at a specific index in the dataset.
    
    Args:
    - image_index (int): Index of the image to explain.
    - args: Command-line arguments or config with necessary parameters.
    
    Returns:
    - output_path (str): Path where the explanation image is saved.
    """
    
    common_init(args, disable_make_dirs=True)

    # Determine the device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Load the CLIP model and preprocessing function
    model, preprocess = clip.load(args.img_enc_name[5:], device=args.device)

    # Load the dataset
    dataset = get_probe_dataset(
        args.probe_dataset, 
        args.probe_split, 
        args.probe_dataset_root_dir, 
        preprocess_fn=preprocess
    )

    # Load a single image from the dataset
    image, label = dataset[image_index]
    img_tensor = image.unsqueeze(0).to(device)  # Add batch dimension and send to device

    # Load method object
    embeddings_path = osp.join(
        args.vocab_dir, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth"
    )
    vocab_txt_path = osp.join(args.vocab_dir, "clipdissect_20k.txt")
    method_obj = method_utils.get_method(
        args.method_name, args, 
        embeddings_path=embeddings_path, 
        vocab_txt_path=vocab_txt_path, 
        use_fixed_sae=True
    )

    # Load the SAE model
    autoencoder_input_dim = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
    n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
    autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim, n_learned_features=n_learned_features, n_components=len(args.hook_points)).to(args.device)
    autoencoder = get_sae_ckpt(args, autoencoder)

    # Get all concepts (embeddings) for the original dataset
    all_concepts = method_obj.get_concepts().to(args.device)
    concept_strengths = all_concepts[image_index]

    all_labels = method_obj.get_labels()
    all_labels = all_labels.to(args.device)
    image_label = get_printable_class_name(args.probe_dataset, int(all_labels[image_index].item()))

    classifier_weights = method_obj.get_classifier_weights()
    classifier_weights = classifier_weights.to(args.device)

    topk = 5  # Number of top concepts to retrieve
    contribs = concept_strengths * classifier_weights
    pred_class = torch.argmax(contribs.sum(dim=1)).item()
    top_strengths, top_indices = torch.topk(contribs[pred_class], k=topk)

    pred_class_name = get_printable_class_name(args.probe_dataset, pred_class)
    top_concepts = [method_obj.get_concept_name(idx)[0] for idx in top_indices]

    print(f"gt_class_name: {image_label}")
    print(f"pred_class_name: {pred_class_name}")
    for concept, strength in zip(top_concepts, top_strengths):
        print(f"Concept: {concept}, Strength: {strength}")

    # Un-normalize transformation for images
    un_normalize = T.Normalize(
        mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
        std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
    )

    # Artificial classifier that outputs concept strengths
    def artificial_classifier(image):
        """
        This classifier will dynamically extract the concepts from the image.
        """
        image = image.squeeze(0)
        image = np.transpose(image, (2, 0, 1))  # Move color channel to the first dimension
        image_tensor = torch.tensor(image).to(torch.float32)
        image = to_pil_image(image_tensor)
        image = image.convert('RGB')
        img_tensor = preprocess(image).unsqueeze(0).to(device)  # Preprocess the image and send to device
        
        with torch.no_grad():
            # Extract CLIP features and compute all concepts
            clip_features = model.encode_image(img_tensor).squeeze(0)
            concepts = method_obj.get_concepts_from_features(clip_features)
         
            contribs = concepts * classifier_weights
        
            top_strengths = contribs[pred_class]
        
        return top_strengths.cpu().numpy().reshape(1, -1)

    # Create LIME image explainer
    explainer = lime_image.LimeImageExplainer()
    image = image.squeeze(0) 
    image = un_normalize(image)
    image = to_pil_image(image)
    image = image.convert('RGB')

    # Generate explanation using the artificial classifier
    explanation = explainer.explain_instance(
        np.array(image).astype(np.float32), 
        artificial_classifier, 
        top_labels=5, 
        num_samples=5000,
        batch_size=1
    )

    # Format concepts with strengths
    top_concepts_with_strengths = [f"{concept} ({strength:.4f})" for concept, strength in zip(top_concepts, top_strengths)]

    # Prepare and save the visualization
    model.eval()
    img = un_normalize(img_tensor.squeeze(0).cpu())
    img = to_pil_image(img)

    # Prepare a single visualization with the original image and top 5 concepts
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))  # 1 row, 6 columns for all images
    axes = axes.ravel()  # Flatten the array to make it easier to loop through

    # Plot the original image
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title("Original Image", fontsize=10)

    # Visualize LIME explanation for the top 5 concepts
    for i, top_label in enumerate(explanation.top_labels[:5]):
        temp, mask = explanation.get_image_and_mask(
            top_label, positive_only=True, num_features=3, hide_rest=False, min_weight=0.0001
        )
        
        # Normalize `temp` to range [0, 1]
        temp = temp / 255.0  # Normalize to the [0, 1] range (if it's in the 0-255 range)
        
        # Apply the mask to the image
        boundary_image = mark_boundaries(temp, mask)

        # Add concept and contribution strength to subtitle
        axes[i + 1].imshow(boundary_image)
        axes[i + 1].axis('off')
        axes[i + 1].set_title(top_concepts_with_strengths[i], fontsize=10)

    # Set the overall title for the figure
    fig.suptitle(
        f"Image Index: {image_index}\nPredicted: {pred_class_name} | Ground Truth: {image_label}",
        fontsize=14
    )

    # Save the final visualization
    output_dir = osp.join(os.getcwd(), "analysis/lime_extension/contribution")
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, f'lime_explanation_alpha_{image_index}_all_in_one.png')
    plt.savefig(output_path)
    plt.close()

    print(f"LIME explanation with all images and contribution scores saved to {output_path}")
    return output_path

import random

parser = arg_parser.get_common_parser()
parser.add_argument("--which_ckpt", type=str, default='final', help='Checkpoint of Sparse autoencoder to load')
args = parser.parse_args()

# Generate 20 random numbers in the range 1 to 3000
random_numbers = random.sample(range(1, 30001), 50)

nums = [35258, 14483, 36359, 31069, 12143, 14240, 23601]
# Call the function 20 times
for num in random_numbers:
# Example call to the function
    generate_lime_explanation(image_index=num, args=args)

# generate_lime_explanation(image_index=2137, args=args)
