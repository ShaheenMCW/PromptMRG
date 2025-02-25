import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from models.blip import blip_decoder
import numpy as np
import json
import os
import pandas as pd
import torch.nn.functional as F

# ---------------------------
# 1. Setup Device and Paths
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/stanford/chexbert/model_promptmrg_20240305.pth"
clip_memory_path = "data/mimic_cxr/clip_text_features.json"  # Precomputed CLIP text embeddings

# ---------------------------
# 2. Load Tokenizer &  Special Tokens
# ---------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_special_tokens({'bos_token': '[DEC]'})
tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])

# SCORES mapping for disease classification tokens
SCORES = ['[BLA]', '[POS]', '[NEG]', '[UNC]']

# ---------------------------
# 3. Define Arguments
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
# 4. Load the Model (BLIP Decoder with Disease Classification)
# ---------------------------
# Use a prompt of 18 "[BLA]" tokens to initialize prompt length.
prompt = " ".join(["[BLA]"] * 18) + " "
model = blip_decoder(args=args, tokenizer=tokenizer, image_size=args.image_size, prompt=prompt).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ---------------------------
# 5. Preprocess the Input Images
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ---------------------------
# 6. Define Input/Output Paths
# ---------------------------
image_folder = "test_image"  # Folder with images
output_results = []

# ---------------------------
# 7. Load CLIP Memory from JSON File
# ---------------------------
with open(clip_memory_path, "r") as f:
    clip_features = np.array(json.load(f))

# ---------------------------
# 8. Define Disease Labels (first 14 categories)
# ---------------------------
disease_labels = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural Thickening",
    "Pneumonia", "Pneumothorax"
]

# ---------------------------
# 9. Process all images in the folder
# ---------------------------
for image_file in os.listdir(image_folder):
    if image_file.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_folder, image_file)
        print(f"\n🖼️ Processing: {image_file}")

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Retrieve Top-K CLIP Features (for now, simply take the first clip_k entries)
        clip_memory_tensor = clip_features[:args.clip_k]  # shape: (clip_k, feature_dim)
        clip_memory_tensor = torch.from_numpy(clip_memory_tensor).unsqueeze(0).float().to(device)
        
        # Debug: Print CLIP Memory details
        print("🔍 CLIP Memory Debugging:")
        print("   Shape:", clip_memory_tensor.shape)  # Expected: (1, 21, feature_dim)
        # print("   First 5 entries:", clip_memory_tensor[0, :5].tolist())

        # Generate Report & Disease Predictions
        captions, cls_preds, cls_preds_logits = model.generate(
            image_tensor, clip_memory_tensor, sample=False, num_beams=args.beam_size,
            max_length=args.gen_max_len, min_length=args.gen_min_len
        )

        # Post-process classification outputs:
        # cls_preds: shape (batch_size, 18) - we use first 14 for disease predictions.
        # Convert discrete predictions into token strings.
        # (This uses SCORES mapping: [BLA]=0, [POS]=1, [NEG]=2, [UNC]=3)
        predicted_token_classes = [SCORES[val] for val in cls_preds[0][:14]]
        
        # Also, convert cls_preds_logits (the softmax probabilities for the positive class)
        # to a list of probabilities. These logits should already be floats.
        disease_probs = cls_preds_logits[0].tolist()  # Expecting length 14
        
        # Debug: Check that we have 14 disease probabilities
        if len(disease_probs) != len(disease_labels):
            print("⚠️ Warning: Number of disease probabilities does not match number of labels!")
            print("   Probabilities length:", len(disease_probs))
            print("   Labels length:", len(disease_labels))
        
        # For display, combine disease name with the predicted token and probability.
        disease_output = {}
        for i, name in enumerate(disease_labels):
            disease_output[name] = {
                "predicted_class": predicted_token_classes[i] if i < len(predicted_token_classes) else "N/A",
                "probability": f"{disease_probs[i]:.4f}" if i < len(disease_probs) else "N/A"
            }

        # Store results for CSV/JSON
        result = {
            "Image": image_file,
            "Report": captions[0],
            "Disease_Classification": disease_output
        }
        output_results.append(result)

        # Print Results for the image
        print("🚀 **Generated Report:**", captions[0])
        print("📊 **Disease Classification:**")
        for disease, info in disease_output.items():
            print(f"   {disease}: {info['predicted_class']} (Prob: {info['probability']})")

# ---------------------------
# 10. Save Results to CSV
# ---------------------------
# Flatten output_results for CSV
csv_rows = []
for res in output_results:
    row = {"Image": res["Image"], "Report": res["Report"]}
    for disease, info in res["Disease_Classification"].items():
        row[f"{disease}_class"] = info["predicted_class"]
        row[f"{disease}_prob"] = info["probability"]
    csv_rows.append(row)

df = pd.DataFrame(csv_rows)
output_csv_path = "generated_reports.csv"
df.to_csv(output_csv_path, index=False)
print(f"\n✅ Reports saved to {output_csv_path}")
