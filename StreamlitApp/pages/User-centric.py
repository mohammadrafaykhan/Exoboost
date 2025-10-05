import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import base64
import random
import requests
import io

# Page styling
st.set_page_config(page_title="🪐 Exoplanet Explorer", page_icon="🪐", layout="centered")
# -------------------------------
# Local Background Image (Base64 Method)
# -------------------------------
bg_image_path = "Files/bgimg.jpg"  # ensure this exists
ex_image_path = "Files/extypes.jpg"

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

st.title("🏠 User")
st.write("Welcome to the user focused page of your Streamlit app!")


# Mapping: Pretty names -> Model feature names
NAME_MAP = {
    "Orbital Period (days)": "koi_period",
    "Orbital Period Error (+)": "koi_period_err1",
    "Time of Transit (BJD)": "koi_time0bk",
    "Transit Duration (hrs)": "koi_duration",
    "Transit Depth (ppm)": "koi_depth",
    "Planet Radius (Earth radii)": "koi_prad",
    "Equilibrium Temperature (K)": "koi_teq",
    "Insolation Flux (Earth flux)": "koi_insol",
    "Stellar Effective Temp (K)": "koi_steff",
    "Surface Gravity (log g)": "koi_slogg",
    "Stellar Radius Error (+)": "koi_srad_err1",
    "RA (deg)": "ra",
    "Dec (deg)": "dec",
    "Kepler Magnitude": "koi_kepmag",
    "Time of Transit Error (+)": "koi_time0bk_err1",
    "Impact Parameter": "koi_impact",
    "Impact Parameter Error (+)": "koi_impact_err1",
    "Impact Parameter Error (-)": "koi_impact_err2",
    "Transit Duration Error (+)": "koi_duration_err1",
    "Transit Depth Error (+)": "koi_depth_err1",
    "Planet Radius Error (+)": "koi_prad_err1",
    "Planet Radius Error (-)": "koi_prad_err2",
    "Insolation Flux Error (+)": "koi_insol_err1",
    "Transit Model SNR": "koi_model_snr",
    "Stellar Effective Temp Error (+)": "koi_steff_err1",
    "Stellar Effective Temp Error (-)": "koi_steff_err2",
    "Surface Gravity Error (+)": "koi_slogg_err1",
    "Surface Gravity Error (-)": "koi_slogg_err2",
    "Stellar Radius Error (-)": "koi_srad_err2",
    "Depth / Stellar Radius": "depth_to_srad",
    "Planet / Stellar Radius Ratio": "prad_to_srad_ratio",
    "Period / Impact Parameter": "period_to_impact",
    "Log(1 + Insolation Flux)": "log_insol",
    "Log(1 + Transit SNR)": "log_snr",
}

FEATURE_RANGES = {
    "Orbital Period (days)": (-0.389486, 3.884597),
    "Orbital Period Error (+)": (-0.249980, 3.912221),
    "Time of Transit (BJD)": (-0.509194, 3.841147),
    "Time of Transit Error (+)": (-0.651946, 3.741984),
    "Impact Parameter": (-0.630537, 3.631575),
    "Impact Parameter Error (+)": (-0.666667, 3.568627),
    "Impact Parameter Error (-)": (-3.567234, 0.7243362),
    "Transit Duration (hrs)": (-1.031911, 3.611988),
    "Transit Duration Error (+)": (-0.692735, 3.733467),
    "Transit Depth (ppm)": (-0.572557, 3.788191),
    "Transit Depth Error (+)": (-0.723214, 3.843750),
    "Planet Radius (Earth radii)": (-1.142857, 3.691814),
    "Planet Radius Error (+)": (-0.723404, 3.553191),
    "Planet Radius Error (-)": (-3.689655, 0.7241379),
    "Equilibrium Temperature (K)": (-1.433444, 3.570820),
    "Insolation Flux (Earth flux)": (-0.312844, 3.735998),
    "Insolation Flux Error (+)": (-0.261479, 3.751848),
    "Transit Model SNR": (-0.675000, 3.850000),
    "Stellar Effective Temp (K)": (-3.652231, 3.522310),
    "Stellar Effective Temp Error (+)": (-1.500000, 3.550000),
    "Stellar Effective Temp Error (-)": (-3.494382, 1.483146),
    "Surface Gravity (log g)": (-3.687500, 3.363971),
    "Surface Gravity Error (+)": (-0.693333, 3.658667),
    "Surface Gravity Error (-)": (-3.685185, 1.037037),
    "Stellar Radius Error (+)": (-0.941748, 3.543689),
    "Stellar Radius Error (-)": (-3.696721, 0.8032787),
    "RA (deg)": (-1.479197, 1.324565),
    "Dec (deg)": (-1.327027, 1.416869),
    "Kepler Magnitude": (-3.649838, 1.684677),
    "Depth / Stellar Radius": (-0.479075, 30.32855),
    "Planet / Stellar Radius Ratio": (-1.007994, 13.79850),
    "Period / Impact Parameter": (-0.239422, 8.018737e9),
    "Log(1 + Insolation Flux)": (-1.665119, 0.9453637),
    "Log(1 + Transit SNR)": (-1.799029, 1.569439),
}

# 7 Questions covering ALL 34 features
QUESTION_CLUSTERS = {
    "size": {
        "features": [
            "Planet Radius (Earth radii)",
            "Planet Radius Error (+)",
            "Planet Radius Error (-)",
            "Planet / Stellar Radius Ratio",
            "Depth / Stellar Radius"
        ],
        "question": "🪐 How big is your planet?",
        "options": {
            "Mercury": {"range": (0.0, 0.3), "desc": "tiny like Mercury"},
            "Mars": {"range": (0.3, 0.6), "desc": "small like Mars"},
            "Earth": {"range": (0.6, 1.4), "desc": "Earth-sized"},
            "Neptune": {"range": (1.4, 2.5), "desc": "big like Neptune"},
            "Jupiter": {"range": (2.5, 4.0), "desc": "huge like Jupiter"}
        }
    },
    "temperature": {
        "features": [
            "Equilibrium Temperature (K)",
            "Insolation Flux (Earth flux)",
            "Insolation Flux Error (+)",
            "Log(1 + Insolation Flux)"
        ],
        "question": "🌡️ How hot or cold is your planet?",
        "options": {
            "Pluto": {"range": (-1.5, -0.5), "desc": "freezing cold like Pluto"},
            "Neptune": {"range": (-0.5, 0.5), "desc": "super cold like Neptune"},
            "Earth": {"range": (0.5, 1.5), "desc": "just right like Earth"},
            "Venus": {"range": (1.5, 2.5), "desc": "very hot like Venus"},
            "Mercury": {"range": (2.5, 4.0), "desc": "scorching hot like Mercury"}
        }
    },
    "orbit": {
        "features": [
            "Orbital Period (days)",
            "Orbital Period Error (+)",
            "Period / Impact Parameter",
            "Time of Transit (BJD)",
            "Time of Transit Error (+)"
        ],
        "question": "🌍 How long is a year on your planet?",
        "options": {
            "Mercury": {"range": (-0.5, 0.5), "desc": "super quick like Mercury (88 days)"},
            "Venus": {"range": (0.5, 1.2), "desc": "pretty fast like Venus (225 days)"},
            "Earth": {"range": (1.2, 2.0), "desc": "like Earth (365 days)"},
            "Jupiter": {"range": (2.0, 3.0), "desc": "long like Jupiter (12 years)"},
            "Neptune": {"range": (3.0, 4.0), "desc": "super long like Neptune (165 years)"}
        }
    },
    "star_type": {
        "features": [
            "Stellar Effective Temp (K)",
            "Stellar Effective Temp Error (+)",
            "Stellar Effective Temp Error (-)",
            "Kepler Magnitude",
            "Stellar Radius Error (+)",
            "Stellar Radius Error (-)"
        ],
        "question": "⭐ What kind of star does your planet orbit?",
        "options": {
            "Red Dwarf": {"range": (-4.0, -1.0), "desc": "small cool red star"},
            "Orange Star": {"range": (-1.0, 0.5), "desc": "medium orange star"},
            "Sun-like": {"range": (0.5, 1.5), "desc": "yellow star like our Sun"},
            "White Star": {"range": (1.5, 2.5), "desc": "hot white star"},
            "Blue Giant": {"range": (2.5, 4.0), "desc": "massive blue giant star"}
        }
    },
    "gravity": {
        "features": [
            "Surface Gravity (log g)",
            "Surface Gravity Error (+)",
            "Surface Gravity Error (-)",
            "Impact Parameter",
            "Impact Parameter Error (+)",
            "Impact Parameter Error (-)"
        ],
        "question": "🎈 How strong is the gravity on your planet?",
        "options": {
            "Moon": {"range": (-4.0, -1.5), "desc": "super light like the Moon"},
            "Mars": {"range": (-1.5, 0.0), "desc": "light like Mars"},
            "Earth": {"range": (0.0, 1.5), "desc": "normal like Earth"},
            "Jupiter": {"range": (1.5, 2.5), "desc": "heavy like Jupiter"},
            "Super Heavy": {"range": (2.5, 4.0), "desc": "crushing gravity"}
        }
    },
    "atmosphere": {
        "features": [
            "Transit Duration (hrs)",
            "Transit Duration Error (+)",
            "Transit Depth (ppm)",
            "Transit Depth Error (+)",
            "Transit Model SNR",
            "Log(1 + Transit SNR)"
        ],
        "question": "💨 What's the atmosphere like on your planet?",
        "options": {
            "Mercury": {"range": (-1.5, 0.0), "desc": "no atmosphere like Mercury"},
            "Mars": {"range": (0.0, 1.0), "desc": "thin dusty air like Mars"},
            "Earth": {"range": (1.0, 2.0), "desc": "breathable air like Earth"},
            "Venus": {"range": (2.0, 3.0), "desc": "super thick clouds like Venus"},
            "Jupiter": {"range": (3.0, 4.0), "desc": "swirling gas storms like Jupiter"}
        }
    },
    "location": {
        "features": [
            "RA (deg)",
            "Dec (deg)"
        ],
        "question": "🔭 Where in the sky is your planet located?",
        "options": {
            "Northern Sky": {"range": (0.5, 1.5), "desc": "in the northern constellations"},
            "Southern Sky": {"range": (-1.5, -0.5), "desc": "in the southern constellations"},
            "Equatorial": {"range": (-0.5, 0.5), "desc": "near the celestial equator"},
            "Deep North": {"range": (1.0, 2.0), "desc": "far in the northern sky"},
            "Deep South": {"range": (-2.0, -1.0), "desc": "far in the southern sky"}
        }
    }
}

def generate_feature_values(answers):
    """Generate ALL 34 feature values based on user answers"""
    feature_values = {}
    
    for q_key, answer in answers.items():
        cluster = QUESTION_CLUSTERS[q_key]
        selected_range = cluster["options"][answer]["range"]
        
        # For each feature in this cluster, generate a value within the answer's range
        for feature_name in cluster["features"]:
            if feature_name in FEATURE_RANGES:
                min_val, max_val = FEATURE_RANGES[feature_name]
                
                # Map the answer range to the feature range
                range_min, range_max = selected_range
                # Normalize to 0-1 based on typical range (-2 to 4)
                norm_min = (range_min + 2) / 6
                norm_max = (range_max + 2) / 6
                
                # Clamp to 0-1
                norm_min = max(0.0, min(1.0, norm_min))
                norm_max = max(0.0, min(1.0, norm_max))
                
                # Apply to feature range
                feat_min = min_val + norm_min * (max_val - min_val)
                feat_max = min_val + norm_max * (max_val - min_val)
                
                # Generate random value in this range
                feature_values[feature_name] = random.uniform(feat_min, feat_max)
    
    return feature_values

def generate_image_prompt(answers, child_name, features):
    """Generate a text-to-image prompt based on answers AND feature values"""
    size_desc = QUESTION_CLUSTERS["size"]["options"][answers["size"]]["desc"]
    temp_desc = QUESTION_CLUSTERS["temperature"]["options"][answers["temperature"]]["desc"]
    orbit_desc = QUESTION_CLUSTERS["orbit"]["options"][answers["orbit"]]["desc"]
    star_desc = QUESTION_CLUSTERS["star_type"]["options"][answers["star_type"]]["desc"]
    gravity_desc = QUESTION_CLUSTERS["gravity"]["options"][answers["gravity"]]["desc"]
    atmos_desc = QUESTION_CLUSTERS["atmosphere"]["options"][answers["atmosphere"]]["desc"]
    location_desc = QUESTION_CLUSTERS["location"]["options"][answers["location"]]["desc"]
    
    # Get actual feature values for intensity
    planet_radius = features.get("Planet Radius (Earth radii)", 0)
    eq_temp = features.get("Equilibrium Temperature (K)", 0)
    insolation = features.get("Insolation Flux (Earth flux)", 0)
    transit_depth = features.get("Transit Depth (ppm)", 0)
    stellar_temp = features.get("Stellar Effective Temp (K)", 0)
    surface_gravity = features.get("Surface Gravity (log g)", 0)
    
    # Normalize values to 0-1 for intensity descriptions
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    
    radius_intensity = normalize(planet_radius, FEATURE_RANGES["Planet Radius (Earth radii)"][0], 
                                   FEATURE_RANGES["Planet Radius (Earth radii)"][1])
    temp_intensity = normalize(eq_temp, FEATURE_RANGES["Equilibrium Temperature (K)"][0], 
                                FEATURE_RANGES["Equilibrium Temperature (K)"][1])
    light_intensity = normalize(insolation, FEATURE_RANGES["Insolation Flux (Earth flux)"][0], 
                                 FEATURE_RANGES["Insolation Flux (Earth flux)"][1])
    atmos_intensity = normalize(transit_depth, FEATURE_RANGES["Transit Depth (ppm)"][0], 
                                 FEATURE_RANGES["Transit Depth (ppm)"][1])
    
    # Create detailed color palette based on temperature intensity
    if temp_intensity < 0.3:
        colors = f"predominantly icy blue and white with silver highlights, frosted crystalline appearance"
        surface = f"frozen tundra with towering ice formations, frost-covered plains, and frozen methane seas"
        glow = "faint blue-white glow from ice reflections"
    elif temp_intensity < 0.5:
        colors = f"pale blue, grey, and white with hints of brown, cold barren appearance"
        surface = f"cold rocky terrain with ice patches, frozen valleys, and sparse frost coverage"
        glow = "subtle dim lighting with cool tones"
    elif temp_intensity < 0.7:
        colors = f"deep blues, vibrant greens, and earthy browns, Earth-like appearance"
        surface = f"diverse terrain with possible liquid water oceans, continents, and varied geography"
        glow = "natural balanced lighting with warm and cool tones"
    elif temp_intensity < 0.85:
        colors = f"warm oranges, yellows, and reddish-brown, heated appearance"
        surface = f"hot rocky terrain with visible heat distortion, desert-like features, and warm atmosphere"
        glow = "warm orange-yellow glow with heat haze effects"
    else:
        colors = f"intense reds, bright oranges, and molten yellows, extreme heat appearance"
        surface = f"volcanic landscape with active lava rivers, magma pools, glowing fissures, and molten rock flows"
        glow = "intense red-orange glow from molten surface, heat radiation visible"
    
    # Create atmosphere description with intensity
    if atmos_intensity < 0.2:
        atmos = f"virtually no atmosphere, sharp shadows, crystal-clear view of cratered surface, stark and exposed"
    elif atmos_intensity < 0.4:
        atmos = f"thin, barely visible atmosphere with faint wisps, minimal cloud cover, transparent hazy layer"
    elif atmos_intensity < 0.6:
        atmos = f"moderate atmosphere with scattered clouds, visible weather patterns, Earth-like cloud formations"
    elif atmos_intensity < 0.8:
        atmos = f"thick, dense atmosphere with heavy cloud layers, obscured surface, Venus-like cloud coverage"
    else:
        atmos = f"extremely thick atmosphere with massive swirling storm systems, turbulent gas clouds, Jupiter-like bands and vortexes"
    
    # Star type visual with brightness intensity
    if "red" in star_desc.lower():
        star_visual = f"small red dwarf star with deep crimson glow, emitting dim reddish light (stellar temp: {stellar_temp:.1f})"
    elif "orange" in star_desc.lower():
        star_visual = f"medium-sized orange star with warm amber glow, casting orange-tinted light (stellar temp: {stellar_temp:.1f})"
    elif "sun" in star_desc.lower():
        star_visual = f"bright yellow sun-like star with golden radiance, similar to our Sun (stellar temp: {stellar_temp:.1f})"
    elif "white" in star_desc.lower():
        star_visual = f"brilliant white star with intense white-blue light, high luminosity (stellar temp: {stellar_temp:.1f})"
    else:
        star_visual = f"massive blue-white giant star with intense blue radiance, extremely luminous (stellar temp: {stellar_temp:.1f})"
    
    # Light intensity on planet
    if light_intensity < 0.3:
        lighting = "dimly lit, shadowy, twilight-like illumination, faint star in background"
    elif light_intensity < 0.7:
        lighting = "well-balanced lighting, Earth-like day illumination, comfortable brightness"
    else:
        lighting = "intensely bright, harsh lighting, overexposed sunny conditions, strong solar radiation"
    
    # Gravity effects on planet shape
    if surface_gravity < 0:
        gravity_effect = "slightly oblate shape due to low gravity, fluffy atmosphere extending far"
    elif surface_gravity < 1.5:
        gravity_effect = "near-perfect spherical shape, Earth-like proportions"
    else:
        gravity_effect = "slightly compressed shape due to high gravity, dense compact appearance"
    
    # Size scale description
    if radius_intensity < 0.2:
        size_detail = "tiny world, barely larger than a large moon"
    elif radius_intensity < 0.4:
        size_detail = "small rocky world, Mars-like scale"
    elif radius_intensity < 0.6:
        size_detail = "Earth-sized world, terrestrial planet scale"
    elif radius_intensity < 0.8:
        size_detail = "large world, Neptune-class size with substantial volume"
    else:
        size_detail = "massive gas giant, Jupiter-class behemoth dominating the view"
    
    prompt = f"""A stunning hyper-realistic space illustration of '{child_name}'s Exoplanet', a {size_detail} that is {size_desc}.

PLANET DETAILS:
- Surface: {surface}
- Colors: {colors}
- Atmosphere: {atmos}
- Visual Glow: {glow}
- Shape: {gravity_effect}
- Lighting: {lighting}
- Gravity: {gravity_desc} (surface gravity: {surface_gravity:.2f})
- Planet Radius: {planet_radius:.2f} Earth radii

STAR DETAILS:
- Star: {star_visual}
- Orbital Period: {orbit_desc}
- Position: {location_desc}

VISUAL STYLE:
- Photorealistic 4K space photography
- Cinematic lighting with accurate physics-based rendering
- Epic sense of scale showing relative sizes
- Rich color grading with {colors}
- Detailed textures showing surface features
- Atmospheric scattering effects
- Cosmic background with distant nebulae and stars
- NASA-quality scientific visualization
- Similar to Hubble Space Telescope imagery and modern exoplanet concept art

Technical details visible: planet illumination intensity {light_intensity:.2%}, atmospheric density {atmos_intensity:.2%}, temperature visualization at {temp_intensity:.2%} heat scale."""
    
    return prompt

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'intro'
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'child_name' not in st.session_state:
    st.session_state.child_name = ""



# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #4A90E2;
    }
    .stButton>button {
        width: 100%;
        background-color: #4A90E2;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# INTRO PAGE
if st.session_state.page == 'intro':
    st.title("🪐 Welcome to the Exoplanet Explorer! 🚀")
    
    child_name = st.text_input("What's your name, young explorer?", value=st.session_state.child_name)
    
    if child_name:
        st.session_state.child_name = child_name
        if st.button("Start My Space Adventure! 🌟"):
            st.session_state.page = 'explanation'
            st.rerun()

# EXPLANATION PAGE
elif st.session_state.page == 'explanation':
    st.title(f"🌟 Hi {st.session_state.child_name}! Let's Learn About Exoplanets! 🌟")
    
    st.markdown("""
    ### What is an Exoplanet? 🪐
    
    An **exoplanet** is a planet that lives around a different star, not our Sun! 
    
    Just like Earth goes around the Sun, exoplanets go around other stars far, far away in space! 🌌
    
    There are **thousands** of exoplanets out there! Some are:
    - 🔥 Super hot like a giant oven
    - ❄️ Freezing cold like a giant ice cube
    - 🪨 Rocky like Earth
    - ☁️ Made of gas like Jupiter
    - 💎 Maybe even covered in diamonds!
    
    Today, YOU will create your very own exoplanet! 🎨✨
    """)
    
    st.image(ex_image_path, 
         caption="Different types of exoplanets discovered by NASA!", use_column_width=True)
    
    if st.button("Let's Create My Planet! 🎨"):
        st.session_state.page = 'questions'
        st.rerun()

# QUESTIONS PAGE
elif st.session_state.page == 'questions':
    st.title(f"🎨 Design Your Exoplanet, {st.session_state.child_name}! 🎨")
    
    st.markdown("Answer these 7 questions to create your unique planet!")
    
    progress = len(st.session_state.answers) / len(QUESTION_CLUSTERS)
    st.progress(progress)
    
    # Display all questions
    for q_key, q_data in QUESTION_CLUSTERS.items():
        st.markdown(f"### {q_data['question']}")
        
        answer = st.radio(
            "Choose one:",
            options=list(q_data['options'].keys()),
            key=f"radio_{q_key}",
            index=list(q_data['options'].keys()).index(st.session_state.answers.get(q_key, list(q_data['options'].keys())[0]))
        )
        
        st.session_state.answers[q_key] = answer
        st.markdown("---")
    
    if len(st.session_state.answers) == len(QUESTION_CLUSTERS):
        if st.button("🚀 Create My Exoplanet! 🚀"):
            st.session_state.page = 'result'
            st.rerun()

# RESULT PAGE
elif st.session_state.page == 'result':
    st.title(f"🎉 Amazing! Here's {st.session_state.child_name}'s Exoplanet! 🎉")
    
    st.balloons()
    
    # Generate feature values
    features = generate_feature_values(st.session_state.answers)
    
    # Generate image prompt with features
    image_prompt = generate_image_prompt(st.session_state.answers, st.session_state.child_name, features)
    
    st.markdown("### 🖼️ Your Planet Description:")
    
    # Create a nice description
    st.info(f"""
    **{st.session_state.child_name}'s Exoplanet** is a magnificent world!
    
    🪐 **Size:** {QUESTION_CLUSTERS['size']['options'][st.session_state.answers['size']]['desc']}
    
    🌡️ **Temperature:** {QUESTION_CLUSTERS['temperature']['options'][st.session_state.answers['temperature']]['desc']}
    
    🌍 **Year Length:** {QUESTION_CLUSTERS['orbit']['options'][st.session_state.answers['orbit']]['desc']}
    
    ⭐ **Star Type:** {QUESTION_CLUSTERS['star_type']['options'][st.session_state.answers['star_type']]['desc']}
    
    🎈 **Gravity:** {QUESTION_CLUSTERS['gravity']['options'][st.session_state.answers['gravity']]['desc']}
    
    💨 **Atmosphere:** {QUESTION_CLUSTERS['atmosphere']['options'][st.session_state.answers['atmosphere']]['desc']}
    
    🔭 **Location:** {QUESTION_CLUSTERS['location']['options'][st.session_state.answers['location']]['desc']}
    """)
    
    # st.markdown("### 🎨 Fictional Visual for Your Planet:")
    # st.code(image_prompt, language="text")
    API_TOKEN = "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"  # <-- Put your API key here
    MODEL_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    OUTPUT_FILE = "generated_planet.png"

    if st.button("Generate Image"):
            with st.spinner("🎨 Generating image... this may take a few seconds"):
                headers = {"Authorization": f"Bearer {API_TOKEN}"}
                try:
                    response = requests.post(
                        MODEL_URL,
                        headers=headers,
                        json={"inputs": image_prompt},
                        timeout=120
                    )

                    if response.status_code == 200:
                        image = Image.open(io.BytesIO(response.content))
                        st.image(image, caption="Generated Image")

                        # Save and download
                        image.save(OUTPUT_FILE)
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format="PNG")
                        img_byte_arr.seek(0)
                        st.download_button(
                            label="📥 Download Image",
                            data=img_byte_arr,
                            file_name=OUTPUT_FILE,
                            mime="image/png"
                        )
                        st.success(f"✅ Image saved as {OUTPUT_FILE}")
                    elif response.status_code == 503:
                        st.warning("⏳ Model is loading... please try again in a few seconds.")
                    else:
                        st.error(f"❌ Error {response.status_code}: {response.text}")

                except Exception as e:
                    st.error(f"❌ Exception: {str(e)}")
        
    # st.markdown("### 📊 NASA KOI Dataset Features Generated:")
    # st.write(f"**Total Features Generated:** {len(features)} out of 34")
    
    # with st.expander("Click to see ALL your planet's scientific data! 🔬"):
    #     for feature_name, value in sorted(features.items()):
    #         st.write(f"**{feature_name}:** {value:.6f}")
    
    # st.success("🎨 Copy the prompt above and use it in an AI image generator like DALL-E, Midjourney, or Stable Diffusion to see your planet come to life!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Create Another Planet"):
            st.session_state.answers = {}
            st.session_state.page = 'questions'
            st.rerun()
    
    with col2:
        if st.button("🏠 Start Over"):
            st.session_state.answers = {}
            st.session_state.child_name = ""
            st.session_state.page = 'intro'
            st.rerun()

