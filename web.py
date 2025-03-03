import streamlit as st
import torch
import clip
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from models.blip import blip_decoder  # BLIP model for report generation
import json
import os

# ---------------------------
# 1. Setup Device and Paths
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_clip_path = 'data/mimic_cxr/clip-imp-pretrained_128_6_after_4.pt'  # Pre-trained CLIP weights
clip_text_features_path = "data/mimic_cxr/clip_text_features.json"  # CLIP text features
blip_model_path = "checkpoints/stanford/chexbert/model_promptmrg_20240305.pth"  # BLIP Model

# ---------------------------
# Load CLIP Model at Startup
# ---------------------------
@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    state_dict = torch.load(pretrained_clip_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, preprocess

# ---------------------------
# Load BLIP Model at Startup
# ---------------------------
@st.cache_resource
def load_blip_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])

    class Args:
        image_size = 224
        beam_size = 3
        gen_max_len = 150
        gen_min_len = 100
        clip_k = 21
        batch_size = 1

    args = Args()
    prompt = " ".join(["[BLA]"] * 18) + " "
    model = blip_decoder(args=args, tokenizer=tokenizer, image_size=args.image_size, prompt=prompt).to(device)
    state_dict = torch.load(blip_model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer

# Load models once
clip_model, clip_preprocess = load_clip_model()
blip_model, tokenizer = load_blip_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🩺 PromptMRG - Medical Report Generation")
st.write("Upload a chest X-ray image to generate a radiology report.")

uploaded_file = st.file_uploader(" Upload an X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=False, width=300)  # Reduced size

    # Preprocess Image for CLIP
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)

    # Measure CLIP Retrieval Time
    clip_start_time = time.time()

    # Encode Image Feature using CLIP
    with torch.no_grad():
        image_feature = clip_model.encode_image(image_tensor)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)  # Normalize
        image_feature = image_feature.to(torch.float32)

    # Load Precomputed CLIP Text Features
    with open(clip_text_features_path, "r") as f:
        all_text_features = np.array(json.load(f))

    all_text_features = torch.tensor(all_text_features, dtype=torch.float32).to(device)

    # Compute Cosine Similarity & Retrieve Top-K Matches
    similarity = torch.matmul(image_feature, all_text_features.T)
    top_k = 21
    topk_values, topk_indices = torch.topk(similarity, k=top_k, dim=-1)

    # Retrieve Top-K CLIP Features
    clip_memory = all_text_features[topk_indices.squeeze()].unsqueeze(0).float()

    # Measure CLIP Retrieval Time
    clip_elapsed_time = time.time() - clip_start_time  # Time taken for CLIP retrieval

    # Preprocess Image for BLIP
    blip_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image_blip = blip_transform(image).unsqueeze(0).to(device)

    # Measure BLIP Processing Time
    blip_start_time = time.time()

    # Generate Report & Disease Predictions
    with st.spinner("⏳ Generating Report..."):
        captions, cls_preds, cls_preds_logits = blip_model.generate(
            image_blip, clip_memory, sample=False, num_beams=3,
            max_length=150, min_length=100
        )

    blip_elapsed_time = time.time() - blip_start_time  # Time taken for BLIP processing

    # Define Disease Labels
    disease_labels = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural Thickening",
        "Pneumonia", "Pneumothorax"
    ]
    SCORES = ['[BLA]', '[POS]', '[NEG]', '[UNC]']
    predicted_token_classes = [SCORES[val] for val in cls_preds[0][:14]]
    disease_probs = cls_preds_logits[0].tolist()

    # Display Results
    st.subheader("📄 Generated Report")
    st.write(captions[0])

    st.subheader("🕒 Processing Time")
    st.write(f" CLIP Retrieval Time: **{clip_elapsed_time:.2f} seconds**")
    st.write(f" BLIP Report Generation Time: **{blip_elapsed_time:.2f} seconds**")
    st.write(f" Total Processing Time: **{clip_elapsed_time + blip_elapsed_time:.2f} seconds**")

    st.subheader("🩺 Disease Classification")
    disease_table = []
    for i, name in enumerate(disease_labels):
        disease_table.append([name, predicted_token_classes[i], f"{disease_probs[i]:.4f}"])
    st.table(disease_table)
