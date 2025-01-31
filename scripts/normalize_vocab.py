#!/usr/bin/env python3

import os
import math
import torch
import clip

from dncbm import arg_parser, utils

def main():
    # 1. Parse args
    parser = arg_parser.get_common_parser()
    args = parser.parse_args()
    utils.common_init(args)

    # 2. Load model
    model, preprocess = utils.get_img_model(args)
    model.eval()  # put the model in eval mode

    # 3. Load vocabulary
    vocab_path = os.path.join(args.vocab_dir, "wiki_100k.pth")
    txt_file = os.path.join(args.vocab_dir, "wiki-100k-cleaned.txt")

    with open(txt_file, "r") as f:
        words = [line.strip() for line in f]

    # 4. Tokenize text
    text_tokens = clip.tokenize(words)
    text_tokens = text_tokens.to(args.device)

    # 5. Encode in batches to avoid OOM
    batch_size = 512  # tweak as needed
    all_embeddings = []

    with torch.no_grad():
        num_batches = math.ceil(len(text_tokens) / batch_size)
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch = text_tokens[start:end]

            # Encode text
            batch_embeddings = model.encode_text(batch)

            # Convert to float32 and normalize
            batch_embeddings = batch_embeddings.float()
            batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True)

            # Move to CPU to free GPU memory
            all_embeddings.append(batch_embeddings.cpu())

    # 6. Concatenate all batch embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # 7. Save to disk
    output_path = os.path.join(
        args.vocab_dir, 
        f"embeddings_{args.img_enc_name_for_saving}_wiki_100k.pth"
    )
    torch.save(all_embeddings, output_path)

    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    main()
