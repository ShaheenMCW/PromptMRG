import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from torchvision import transforms

# ---------------------------
# 1. Setup Device and Paths
# ---------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pretrained_path = 'data/mimic_cxr/clip-imp-pretrained_128_6_after_4.pt'  
clip_text_features_path = "data/mimic_cxr/clip_text_features.json"  # Precomputed CLIP text features
image_folder = "test_image"  # Folder containing images

# ---------------------------
# 2. Load CLIP Model & Pretrained Weights
# ---------------------------
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
state_dict = torch.load(pretrained_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f" Loaded CLIP checkpoint from {pretrained_path}")

# ---------------------------
# 3. Load Precomputed CLIP Text Features
# ---------------------------
with open(clip_text_features_path, "r") as f:
    all_text_features = np.array(json.load(f))  # Expected shape: (N, 512)

# Convert to PyTorch tensor and ensure float32 dtype
all_text_features = torch.tensor(all_text_features, dtype=torch.float32).to(device)
print(f" Loaded CLIP text features, shape: {all_text_features.shape}")

# ---------------------------
# 4. Iterate Over All Images in Folder
# ---------------------------
top_k = 21  # Number of closest text embeddings to retrieve

for image_file in os.listdir(image_folder):
    if image_file.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_folder, image_file)
        print(f"\n Processing Image: {image_file}")

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

            # Encode image feature using CLIP
            with torch.no_grad():
                image_feature = model.encode_image(image_tensor)
                image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)  # Normalize
                image_feature = image_feature.to(torch.float32)  # Convert to float32

            print(f"    Image feature extracted, shape: {image_feature.shape}")

            # Compute Cosine Similarity
            similarity = torch.matmul(image_feature, all_text_features.T)  # Shape: [1, N]

            # Retrieve top-k most similar text embeddings
            topk_values, topk_indices = torch.topk(similarity, k=top_k, dim=-1)

            # Print results
            print(f"   🔹 Top-{top_k} CLIP indices:", topk_indices.squeeze().tolist())
        
        except Exception as e:
            print(f" Error processing {image_file}: {e}")

print("\n Retrieval completed for all images in the folder.")
