# This file loads images relating to CLIP features for a given word index across all datasets
import os
import os.path as osp
import torch
import torchvision
import clip
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from tqdm import tqdm

from dncbm import method_utils, arg_parser, config
from dncbm.utils import common_init, get_probe_dataset

# Initialize arguments and common settings
parser = arg_parser.get_common_parser()
parser.add_argument("--which_ckpt", type=str, default='final')
args = parser.parse_args()
common_init(args, disable_make_dirs=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

# Load CLIP
model, preprocess = clip.load(args.img_enc_name[5:], device=args.device)
model.eval()

# Define datasets and splits
split_config = {
    "imagenet": "val",
    "cifar10": "test",
    "cifar100": "test",
    "places365": "val"
}
all_probe_datasets = ["imagenet", "cifar10", "cifar100", "places365"]

datasets_dict = {}
features_dict = {}

# Un-normalize transformation (CLIP RN50)
un_normalize = torchvision.transforms.Normalize(
    mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
    std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
)

# Load method object and embeddings
embeddings_path = osp.join(args.vocab_dir, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth")
vocab_txt_path = osp.join(args.vocab_dir, "clipdissect_20k.txt")
method_obj = method_utils.get_method(
    args.method_name,
    args,
    embeddings_path=embeddings_path,
    vocab_txt_path=vocab_txt_path,
    use_fixed_sae=True
)

all_embeddings = torch.cat(method_obj.all_embeddings).to(device)  # (vocab_size, embed_dim)
all_embeddings = F.normalize(all_embeddings, dim=1).float()

# Load each dataset via get_probe_dataset and compute features
for ds_name in tqdm(all_probe_datasets):
    print(f"Loading dataset: {ds_name}")
    ds_split = split_config[ds_name]
    ds_root_dir = config.probe_dataset_root_dir_dict[ds_name]

    ds = get_probe_dataset(ds_name, ds_split, ds_root_dir, preprocess_fn=preprocess)
    datasets_dict[ds_name] = ds

    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    feats_list = []
    for images, _ in loader:
        images = images.to(device)
        with torch.no_grad():
            img_features = model.encode_image(images).cpu()
        feats_list.append(img_features)
    feats = torch.cat(feats_list, dim=0).float().to(device)
    feats = F.normalize(feats, dim=1)
    features_dict[ds_name] = feats

# Show top images for a given word index across all datasets
def show_imgs_per_word(word_idx=982, top_k=4, output_dir = "scripts/visualization/survey_features"):
    word = method_obj.vocab_txt_all[0][word_idx]
    os.makedirs(output_dir, exist_ok=True)
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f"Top {top_k} Images per Dataset for Word: {word} (idx={word_idx})")

    for row_idx, ds_name in enumerate(all_probe_datasets):
        ds_feats = features_dict[ds_name]
        # Cosine similarity with the current word embedding
        cos_sim = torch.matmul(all_embeddings[word_idx].unsqueeze(0), ds_feats.t())  # (1, num_images)
        top_indices = torch.topk(cos_sim.squeeze(0), top_k).indices

        ds = datasets_dict[ds_name]
        for col_idx, img_idx in enumerate(top_indices):
            img, _ = ds[img_idx]
            if isinstance(img, torch.Tensor):
                img = un_normalize(img)
                img = to_pil_image(img)
            axs[row_idx, col_idx].imshow(img)
            axs[row_idx, col_idx].set_title(f"{ds_name} #{img_idx.item()}")
            axs[row_idx, col_idx].axis("off")

    output_path = osp.join(output_dir, f'clip_{word_idx}_{word}.png')
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    word_idx = [1960, 11953, 1368, 939, 1403, 4391] #pink, sweater, tree, lake, flowers, chairs
    word_idx = [6744, 8102, 12968, 3656, 2337, 3945] #cloud 4029, fog, sunlight, doors, golden, bathroom
    for idx in word_idx:
        show_imgs_per_word(word_idx=idx, top_k=4)
