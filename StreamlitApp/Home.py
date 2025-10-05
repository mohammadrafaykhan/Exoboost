import streamlit as st
from PIL import Image
import os
import base64

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="ExoPlorers | NASA Space Apps 2025",
    page_icon="🪐",
    layout="wide",
)

# -------------------------------
# Local Background Image (Base64 Method)
# -------------------------------
bg_image_path = "Files/bgimg.jpg"  # ensure this exists

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

if os.path.exists(bg_image_path):
    bin_str = get_base64_of_bin_file(bg_image_path)
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)
else:
    st.warning("⚠️ Background image not found. Please add 'Files/bgimg.jpg' in your folder.")

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            color: #ffffff;
        }
        .subtitle {
            text-align: center;
            font-size: 1.5em;
            color: #cfcfcf;
        }
        .team-section {
            text-align: center;
            margin-top: 3em;
        }
        .member {
            font-size: 1.2em;
            margin: 0.3em 0;
            color: #ffffff;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            color: #aaaaaa;
            margin-top: 4em;
        }

        /* Move all Streamlit content upward */
        div.block-container {
            padding-top: 1rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Main Content
# -------------------------------
st.markdown('<h1 class="main-title">🌌 ExoPlorers</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">NASA Space Apps Challenge 2025</h2>', unsafe_allow_html=True)

st.divider()

st.markdown(
    """
    <h3 style='text-align:center;'>🛰️ Challenge: <span style='color:#f4d03f;'>A World Away: Hunting for Exoplanets with AI</span></h3>
    <p style='text-align:center; max-width:800px; margin:auto; color:#e0e0e0;'>
    Our mission is to leverage artificial intelligence to enhance the detection and analysis of exoplanets — 
    worlds orbiting distant stars. Using advanced ML models and NASA’s open datasets, we aim to uncover 
    patterns hidden in cosmic data and contribute to humanity’s quest to find other Earth-like planets.
    </p>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Team Members Section
# -------------------------------
st.markdown('<div class="team-section">', unsafe_allow_html=True)
st.markdown("<h2>👩‍🚀 Team Members</h2>", unsafe_allow_html=True)

team_members = [
    "Zienab Esam",
    "Pratik Prakash Jadhav",
    "Mohammad Rafay Khan",
    "Alyaa Shahin",
    "Abdo Raslan",
    "Saif Elkady"
]

for member in team_members:
    st.markdown(f"<div class='member'>🪐 {member}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    """
    <div class="footer">
        Built with ❤️ for NASA Space Apps Challenge 2025 | Team ExoPlorers
    </div>
    """,
    unsafe_allow_html=True,
)
