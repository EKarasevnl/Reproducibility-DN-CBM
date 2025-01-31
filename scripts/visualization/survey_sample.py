import os
import os.path as osp
import shutil
import random
from tqdm import tqdm
from survey import show_imgs_per_word

# Sample 24 random concepts from CSV
concepts_csv = "/scratch-shared/fact_1_2025/fact-1/Assigned Names/clip_RN50_concept_name.csv"
with open(concepts_csv, "r") as f:
    lines = f.readlines()

random_concepts = random.sample(lines, 24)
random_concept_indices = [int(c.split(",")[0]) for c in random_concepts]
concept_names = [c.split(",")[1].strip().lower() for c in random_concepts]

print(f"Random concept names: {concept_names}")
print(f"Random concept indices: {random_concept_indices}")

# Directory containing DN-CBM PDF files
dncbm_dir = "/scratch-shared/fact_1_2025/fact-1/analysis/task_agnosticity/clip_RN50/vis"

# Vocab file
vocab_path = "/scratch-shared/fact_1_2025/fact-1/vocab/clipdissect_20k.txt"
with open(vocab_path, "r") as vf:
    vocab_lines = [v.strip() for v in vf.readlines()]

# Survey images output base
survey_images_base = "/scratch-shared/fact_1_2025/fact-1/analysis/survey_images"

for concept_idx, concept_name in tqdm(zip(random_concept_indices, concept_names), desc="Processing concepts"):
    print(f"Processing concept: {concept_name}, index: {concept_idx}")

    # Create a folder to store results for this concept
    concept_folder_path = osp.join(survey_images_base, concept_name)
    os.makedirs(concept_folder_path, exist_ok=True)

    # Find matching PDF files containing this concept index
    for file in os.listdir(dncbm_dir):
        if file.endswith(".pdf"):
            # Parse concept indices from filename
            try:
                idx_part = file.split("_")[2].split("[")[1].split(",")
                idx_clean = [x.replace("]", "").strip() for x in idx_part]
                concept_indices = [int(c) for c in idx_clean]
            except (IndexError, ValueError):
                print(f"Error: Could not parse concept indices from {file}.")
                continue

            if concept_idx in concept_indices:
                source_pdf = osp.join(dncbm_dir, file)
                target_pdf = osp.join(concept_folder_path, f"dncbm_{concept_name}.pdf")
                if osp.isfile(source_pdf):
                    shutil.copy2(source_pdf, target_pdf)
                else:
                    print(f"Error: {source_pdf} is not a valid file.")

    # Find this concept_name in the vocabulary
    if concept_name in vocab_lines:
        vocab_index = vocab_lines.index(concept_name)
        print(f"Vocab index: {vocab_index}")
        show_imgs_per_word(word_idx=vocab_index, top_k=4, output_dir=concept_folder_path)
        print(f"CLIP images saved to {concept_folder_path}")
    else:
        print(f"Concept name '{concept_name}' not found in the vocabulary.")
