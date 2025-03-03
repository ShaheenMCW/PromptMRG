import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from models.blip import blip_decoder  # BLIP model for report generation
import torch.nn.functional as F

# ---------------------------
# 1. Setup Device and Paths
# ---------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pretrained_clip_path = 'data/mimic_cxr/clip-imp-pretrained_128_6_after_4.pt'  # Pre-trained CLIP weights
clip_text_features_path = "data/mimic_cxr/clip_text_features.json"  # Precomputed CLIP text features
image_path = "data/iu_xray/images/CXR3191_IM-1505/1.png"  # Target Image
blip_model_path = "checkpoints/stanford/chexbert/model_promptmrg_20240305.pth"  # BLIP Model for Report Generation

# ---------------------------
# 2. Load CLIP Model & Pretrained Weights
# ---------------------------
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
state_dict = torch.load(pretrained_clip_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f" Loaded CLIP checkpoint from {pretrained_clip_path}")

# ---------------------------
# 3. Load and Preprocess Image
# ---------------------------
image = Image.open(image_path).convert("RGB")
image_tensor = preprocess(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

# Encode image feature using CLIP
with torch.no_grad():
    image_feature = model.encode_image(image_tensor)
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)  # Normalize
    image_feature = image_feature.to(torch.float32)  # Convert to float32

print(f" Image feature extracted, shape: {image_feature.shape}")

# ---------------------------
# 4. Load Precomputed CLIP Text Features
# ---------------------------
# Load CLIP text features from JSON file
with open(clip_text_features_path, "r") as f:
    all_text_features = np.array(json.load(f))  # Expected shape: (N, 512)

# Convert to PyTorch tensor and ensure float32 dtype
all_text_features = torch.tensor(all_text_features, dtype=torch.float32).to(device)
print(f" Loaded CLIP text features, shape: {all_text_features.shape}")

# ---------------------------
# 5. Compute Cosine Similarity & Retrieve Top-K Matches
# ---------------------------
# Compute Cosine Similarity
similarity = torch.matmul(image_feature, all_text_features.T)  # Shape: [1, N]

# Retrieve top 21 closest text embeddings
top_k = 21
topk_values, topk_indices = torch.topk(similarity, k=top_k, dim=-1)

# Debugging Outputs
print(" Top-k similarity scores:", topk_values.squeeze().tolist())
print(" Top-k indices:", topk_indices.squeeze().tolist())

# Form the CLIP memory from these retrieved indices.
clip_memory = all_text_features[topk_indices.squeeze()]  # Shape: [clip_k, D]
clip_memory = clip_memory.unsqueeze(0)  # Shape: [1, clip_k, D]
clip_memory = clip_memory.float()  # Ensure dtype consistency
print(" CLIP Memory Shape:", clip_memory.shape)

# ---------------------------
# 6. Load BLIP Tokenizer & Setup Special Tokens
# ---------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_special_tokens({'bos_token': '[DEC]'})
tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])

# SCORES mapping for disease classification tokens
SCORES = ['[BLA]', '[POS]', '[NEG]', '[UNC]']

# ---------------------------
# 7. Define Arguments (simulate argparse)
# ---------------------------
class Args:
    image_size = 224
    beam_size = 3
    gen_max_len = 150
    gen_min_len = 100
    clip_k = 21
    batch_size = 1

args = Args()

# ---------------------------
# 8. Load the BLIP Model (for Report Generation & Disease Classification)
# ---------------------------
# We initialize the model with a prompt of 18 "[BLA]" tokens.
prompt = " ".join(["[BLA]"] * 18) + " "
blip_model = blip_decoder(args=args, tokenizer=tokenizer, image_size=args.image_size, prompt=prompt).to(device)

state_dict = torch.load(blip_model_path, map_location=device)
blip_model.load_state_dict(state_dict, strict=False)
blip_model.eval()
print(f" Loaded BLIP model from {blip_model_path}")

# ---------------------------
# 9. Preprocess the Target Image for BLIP
# ---------------------------
blip_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
image_blip = blip_transform(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

# ---------------------------
# 10. Run Inference: Generate Report and Predict Diseases
# ---------------------------
captions, cls_preds, cls_preds_logits = blip_model.generate(
    image_blip, clip_memory, sample=False, num_beams=args.beam_size,
    max_length=args.gen_max_len, min_length=args.gen_min_len
)

# ---------------------------
# 11. Post-Process Disease Classification Outputs
# ---------------------------
# Define disease labels (first 14 categories)
disease_labels = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural Thickening",
    "Pneumonia", "Pneumothorax"
]

# Convert discrete classification outputs (cls_preds) into tokens
predicted_token_classes = [SCORES[val] for val in cls_preds[0][:14]]
# Retrieve the positive class probabilities (cls_preds_logits) for each disease
disease_probs = cls_preds_logits[0].tolist()  # Expecting 14 values

# ---------------------------
# 12. Print the Results
# ---------------------------
print("\n Generated Report:")
print(captions)

print("\n Disease Classification (Positive-Class Probabilities):")
for i, name in enumerate(disease_labels):
    print(f"   {name}: {disease_probs[i]:.4f}")

print("\n Discrete Disease Predictions:")
for i, token in enumerate(predicted_token_classes):
    print(f"   {disease_labels[i]}: {token}")
