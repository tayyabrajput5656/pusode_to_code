import streamlit as st
import torch
import torch.nn as nn
import json
import math
import time

# Load vocabulary
with open("vocabulary.json", "r") as f:
    vocab = json.load(f)

# Streamlit UI Enhancements
st.set_page_config(page_title="Pseudocode to C++ Translator", page_icon="💡", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #1e1e2e;
            color: white;
        }
        .stTextInput, .stTextArea, .stButton {
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #ff4b4b;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Settings ⚙️")
st.sidebar.success(f"✅ Vocabulary loaded with {len(vocab)} tokens")

# Transformer Configuration
class Config:
    vocab_size = 12006  # Adjust based on vocabulary.json
    max_length = 100
    embed_dim = 256
    num_heads = 8
    num_layers = 2
    feedforward_dim = 512
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# Transformer Model
class Seq2SeqTransformer(nn.Module):
    def __init__(self, config):
        super(Seq2SeqTransformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.transformer = nn.Transformer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout
        )
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) * math.sqrt(config.embed_dim)
        tgt_emb = self.embedding(tgt) * math.sqrt(config.embed_dim)
        out = self.transformer(src_emb.permute(1, 0, 2), tgt_emb.permute(1, 0, 2))
        return self.fc_out(out.permute(1, 0, 2))

# Load Model
@st.cache_resource
def load_model(path):
    model = Seq2SeqTransformer(config).to(config.device)
    model.load_state_dict(torch.load(path, map_location=config.device))
    model.eval()
    return model

pseudo_to_cpp_model = load_model("pusodo_to_code.pth")

st.sidebar.success("✅ Model loaded successfully!")

# Translation Function
def translate(model, input_tokens, vocab, device, max_length=50):
    model.eval()
    input_ids = [vocab.get(token, vocab["<unk>"]) for token in input_tokens]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    output_ids = [vocab["<start>"]]
    for _ in range(max_length):
        output_tensor = torch.tensor(output_ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(input_tensor, output_tensor)
        next_token_id = predictions.argmax(dim=-1)[:, -1].item()
        output_ids.append(next_token_id)
        if next_token_id == vocab["<end>"]:
            break
    id_to_token = {idx: token for token, idx in vocab.items()}
    return " ".join([id_to_token.get(idx, "<unk>") for idx in output_ids[1:]])

# Streamlit UI
title_placeholder = st.empty()
title_placeholder.markdown("<h1 style='text-align: center; color: #ff4b4b;'>Pseudocode to C++ Translator 🚀</h1>", unsafe_allow_html=True)

st.write("\n")
st.info("Enter your pseudocode and get an instant C++ translation.")
user_input = st.text_area("Enter Pseudocode:", height=150)

if st.button("Translate ✨"):
    with st.spinner("Translating... 🛠️"):
        tokens = user_input.strip().split()
        translated_code = translate(pseudo_to_cpp_model, tokens, vocab, config.device)
        time.sleep(1.5)  # Simulating processing time for animation
    
    st.subheader("Generated Translation:")
    st.code(translated_code, language="cpp")
