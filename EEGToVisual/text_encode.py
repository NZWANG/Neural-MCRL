import torch
import pickle
import open_clip
from torch import nn
import os
import gc

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_path = '/root/autodl-tmp/EEG2Vision/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.safetensors'
model_type = 'ViT-H-14'
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type,
    pretrained=model_path,
    precision='fp32',
    device=device
)

def load_texts(file_path):
    """加载文本描述"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            texts = pickle.load(f) 
        return texts
    else:
        print(f"Warning: {file_path} not found. No text descriptions loaded.")
        return []

def encode_texts(texts, batch_size=32):
    text_features_list = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        text_inputs = torch.cat([open_clip.tokenize(t) for t in batch_texts]).to(device)

        with torch.no_grad():
            text_features = vlmodel.encode_text(text_inputs)
        
        text_features = nn.functional.normalize(text_features, dim=-1).detach()
        text_features_list.append(text_features.cpu())

        del text_inputs
        gc.collect()
        torch.cuda.empty_cache()

    return torch.cat(text_features_list) 

def main():
    text_file_path = ''
    texts = load_texts(text_file_path)

    text_features = encode_texts(texts, batch_size=32)

    torch.save({'text_features': text_features}, 'text_features.pt')

    print(f"Text features shape: {text_features.shape}")
    print(f"Text features (first 5): {text_features[:5]}") 

if __name__ == "__main__":
    main()
