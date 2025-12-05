import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from torch import nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import joblib
import streamlit_authenticator as stauth
from fpdf import FPDF, XPos, YPos 
import tempfile
import os
import plotly.express as px
import plotly.io as pio

# --- GRI CALCULATION IMPORTS & CONFIG ---
from skimage import io, color, filters
from scipy.signal import fftconvolve
import pickle
from sklearn.decomposition import IncrementalPCA

# --- Configuration for Gabor/PCA (MUST MATCH PCA TRAINING SCRIPT) ---
TARGET_WIDTH = 1024 # Keep this consistent
TARGET_HEIGHT = 768 # Keep this consistent
# IMPORTANT: Ensure this file is accessible in your deployment environment
pca_model_path = 'incremental_pca_model.pkl' 

# --- Fixed Gabor Parameters (MUST MATCH PCA TRAINING SCRIPT) ---
fixed_numRows = TARGET_HEIGHT
fixed_numCols = TARGET_WIDTH

fixed_wavelengthMin = 4 / np.sqrt(2)
fixed_wavelengthMax = np.hypot(fixed_numRows, fixed_numCols)
fixed_n = int(np.floor(np.log2(fixed_wavelengthMax / fixed_wavelengthMin)))
if fixed_n <= 1: fixed_n = 2
fixed_wavelength = 2**(np.arange(0, fixed_n - 1)) * fixed_wavelengthMin
fixed_deltaTheta = 45 # In degrees
fixed_orientation = np.arange(0, 180, fixed_deltaTheta) # In degrees


# --- Feature Extraction Function (must be IDENTICAL to PCA training) ---
def extract_features(image_array, gabor_bank, target_width, target_height, fixed_wavelength, fixed_orientation):
    """
    Extracts Gabor and spatial features from a single image.
    This function must be IDENTICAL to the one used in PCA training and GRI calculation.
    """
    # Ensure image is resized consistently
    if image_array.ndim == 2:
        Agray = image_array.astype(np.float64)
        if Agray.shape[0] != target_height or Agray.shape[1] != target_width:
             resized_img = cv2.resize(image_array, (target_width, target_height), interpolation=cv2.INTER_AREA)
             Agray = resized_img.astype(np.float64)
    else:
        resized_img = cv2.resize(image_array, (target_width, target_height), interpolation=cv2.INTER_AREA)
        Agray = color.rgb2gray(resized_img).astype(np.float64)

    numRows, numCols = Agray.shape

    gabormag = np.zeros((numRows, numCols, len(gabor_bank)), dtype=np.float64)
    for i, kernel in enumerate(gabor_bank):
        real_part = fftconvolve(Agray, np.real(kernel), mode='same')
        imag_part = fftconvolve(Agray, np.imag(kernel), mode='same')
        gabormag[:, :, i] = np.sqrt(real_part**2 + imag_part**2)

    K = 3 # Gaussian smoothing factor
    smoothed_gabormag = np.zeros_like(gabormag, dtype=np.float64)
    gabor_idx = 0
    
    for wl in fixed_wavelength:
        for orient_deg in fixed_orientation:
            if gabor_idx < len(gabor_bank):
                sigma_gauss = K * (0.5 * wl)
                if sigma_gauss <= 0: sigma_gauss = 0.1
                smoothed_gabormag[:, :, gabor_idx] = filters.gaussian(
                    gabormag[:, :, gabor_idx], sigma=sigma_gauss, preserve_range=True, channel_axis=None
                )
                gabor_idx += 1
            else:
                break

    X_coords = np.arange(numCols)
    Y_coords = np.arange(numRows)
    X_mesh, Y_mesh = np.meshgrid(X_coords, Y_coords)
    
    # Concatenate Gabor features and spatial coordinates (Gabor + X_coord + Y_coord)
    featureSet = np.concatenate((smoothed_gabormag, X_mesh[:, :, np.newaxis], Y_mesh[:, :, np.newaxis]), axis=2)
    X_flat = featureSet.reshape(numRows * numCols, -1)
    
    # Normalization (per feature column)
    std_devs = np.std(X_flat, axis=0)
    std_devs[std_devs == 0] = 1e-6 # Avoid division by zero
    
    Xnorm = X_flat / std_devs  
    
    return Xnorm

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide")

# --- PDF GENERATION CLASS AND FUNCTION ---
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        # Use new FPDF argument style
        self.cell(0, 10, 'Glaucoma Screening Report', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, new_x=XPos.RMARGIN, new_y=YPos.TOP, align='C')

def create_report_pdf(patient_info, original_img, overlay_img, metrics_df, metrics_fig, gri_value):
    pdf = PDF('P', 'mm', 'A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Patient Details ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Patient Details', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
    pdf.set_font('Helvetica', '', 11)
    for key, value in patient_info.items():
        pdf.cell(40, 8, f"{key}:", border=0, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(0, 8, str(value), border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    # --- GRI Result ---
    #pdf.set_font('Helvetica', 'B', 12)
    #pdf.set_fill_color(200, 220, 255) # Light blue background
    #gri_text = f'Glaucoma Risk Index (GRI): {gri_value:.3f}' if gri_value is not None else 'Glaucoma Risk Index (GRI): N/A'
    #pdf.cell(0, 10, gri_text, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L', fill=True)
    #pdf.ln(5)

    # --- Images ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Screening Images', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_orig, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_overlay:
        cv2.imwrite(tmp_orig.name, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(tmp_overlay.name, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        # Removed deprecated 'type' parameter
        pdf.image(tmp_orig.name, x=15, w=80) 
        pdf.image(tmp_overlay.name, x=110, w=80)
    
    pdf.set_font('Helvetica', 'I', 10)
    # Renamed 'txt' to 'text'
    pdf.text(x=45, y=pdf.get_y() + 65, text='Original Image') 
    pdf.text(x=135, y=pdf.get_y() + 65, text='Segmented Overlay')
    pdf.ln(75)

    # --- Analysis Results Table ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Analysis Results', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
    
    # UPDATED: More compact and clearer abbreviations for PDF headers
    pdf.set_font('Helvetica', 'B', 7.5) 
    
    headers_map = {
        "VCDR": "VCDR", "ACDR": "ACDR", "DDLS": "DDLS", 
        "INFERIOR_AREA": "InfArea", "DISC_AREA": "DiscArea", 
        "CUP_AREA": "CupArea", "RIM_AREA": "RimArea", 
        "DISC_VERTICAL_HEIGHT": "DiscVHgt", "CUP_VERTICAL_HEIGHT": "CupVHgt", 
        "DISC_HORIZONTAL_WIDTH": "DiscHWdt", "CUP_HORIZONTAL_WIDTH": "CupHWdt", 
        "GRI": "GRI", "Prediction": "Prediction", "Confidence": "Confidence"
    }
    
    headers_to_print = list(headers_map.keys())
    
    # Adjusted widths (in mm) for the 14 columns
    col_widths = [10, 10, 10, 12, 12, 12, 12, 14, 14, 14, 14, 10, 18, 18] 

    for i, header in enumerate(headers_to_print):
        width = col_widths[i]
        # Use abbreviated header, use new FPDF arguments
        pdf.cell(width, 10, headers_map[header], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')
    pdf.ln() # Move to the next line after the header row
    
    # Data rows - use smaller font for data
    pdf.set_font('Helvetica', '', 7.5) 
    for _, row in metrics_df.iterrows():
        for i, header in enumerate(headers_to_print):
            
            width = col_widths[i]

            is_float_metric = header in ["VCDR", "ACDR", "DDLS", "GRI"]
            is_conf_float = header == 'Confidence' and isinstance(row[header], float)
            is_area_metric = header in ["INFERIOR_AREA", "DISC_AREA", "CUP_AREA", "RIM_AREA", "DISC_VERTICAL_HEIGHT", "CUP_VERTICAL_HEIGHT", "DISC_HORIZONTAL_WIDTH", "CUP_HORIZONTAL_WIDTH"]
            
            if is_float_metric or is_conf_float:
                value_to_display = f"{row[header]:.3f}" if row[header] is not None else "N/A"
            elif is_area_metric:
                value_to_display = f"{int(row[header])}" if row[header] is not None else "N/A"
            else: 
                value_to_display = str(row[header])

            # Use new FPDF arguments for data cells
            pdf.cell(width, 10, value_to_display, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')
        pdf.ln() # Move to the next line after the data row
    pdf.ln(5) # Add space after the table

    # --- Bar Chart ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Glaucoma-Specific Metrics Chart', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
        if metrics_fig:
            metrics_fig.write_image(tmp_chart.name, scale=2)
            pdf.image(tmp_chart.name, w=180) 

    # FIX: Explicitly convert bytearray/bytes output to 'bytes' for Streamlit download_button
    return bytes(pdf.output(dest='S'))


# --- USER AUTHENTICATION CONFIG ---
config = {
    'credentials': {
        'usernames': {
            'testuser': {
                'email': 'test@user.com',
                'name': 'Test User',
                'password': '$2b$12$pMQfhnxFyeKAUJ6IYOBsC.LU/RRQELL9jrpfa3o6j3U39GnaQj4oy' # Hashed password for 'password123'
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'a_random_secret_key_for_this_app',
        'name': 'glaucoma_app_cookie'
    }
}
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- MODEL LOADING & CACHING ---
@st.cache_resource
def load_all_resources():
    # 1. Load SegFormer and Classifier
    processor = AutoImageProcessor.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
    model = SegformerForSemanticSegmentation.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
    model.eval()
    
    try:
        clf = joblib.load("random_forest_new.pkl")
    except FileNotFoundError:
        st.error('Classifier model file `random_forest_new.pkl` not found. RF prediction will fail.')
        clf = None

    # 2. Pre-calculate Gabor Bank
    gabor_bank = []
    for wl in fixed_wavelength:
        for orient_deg in fixed_orientation:
            orient_rad = np.deg2rad(orient_deg)
            sigma = 0.5 * wl
            if sigma < 1e-6: sigma = 1e-6
            try:
                kernel = filters.gabor_kernel(frequency=1/wl, theta=orient_rad, sigma_x=sigma, sigma_y=sigma)
                gabor_bank.append(kernel)
            except ValueError:
                pass
    
    # 3. Load PCA Model
    incremental_pca = None
    try:
        with open(pca_model_path, 'rb') as f:
            incremental_pca = pickle.load(f)
        if not isinstance(incremental_pca, IncrementalPCA):
            st.error("Loaded object is not an IncrementalPCA instance. GRI will not be calculated.")
            incremental_pca = None
    except FileNotFoundError:
        st.error(f"CRITICAL ERROR: PCA model not found at {pca_model_path}. GRI will not be calculated.")
    except Exception as e:
        st.error(f"CRITICAL ERROR: Failed to load PCA model: {e}. GRI will not be calculated.")
        
    return processor, model, clf, gabor_bank, incremental_pca

# --- Metric Calculation Functions ---
def calculate_ddls(disc_mask, cup_mask):
    y_disc, x_disc = np.where(disc_mask)
    y_cup, x_cup = np.where(cup_mask)
    if not (len(y_disc) > 0 and len(x_disc) > 0 and len(y_cup) > 0 and len(x_cup) > 0):
        return None
    disc_height = y_disc.max() - y_disc.min() + 1
    disc_width = x_disc.max() - x_disc.min() + 1
    cup_height = y_cup.max() - y_cup.min() + 1
    cup_width = x_cup.max() - x_cup.min() + 1
    DD = (disc_height + disc_width) / 2.0
    if DD == 0: return None
    vertical_rim_thickness = (disc_height - cup_height) / 2.0
    horizontal_rim_thickness = (disc_width - cup_width) / 2.0
    min_rim_width = max(0.0, min(vertical_rim_thickness, horizontal_rim_thickness))
    if min_rim_width <= 0: return 0.0
    ddls = min_rim_width / DD
    return round(ddls, 3)

def vertical_cdr(disc_mask, cup_mask):
    y_disc, y_cup = np.where(disc_mask)[0], np.where(cup_mask)[0]
    if not (len(y_disc) and len(y_cup)): return None
    disc_h = y_disc.max() - y_disc.min() + 1
    cup_h = y_cup.max() - y_cup.min() + 1
    if disc_h == 0: return None
    return round(cup_h / disc_h, 3)

def acdr_area(disc_mask, cup_mask):
    disc_sum = np.sum(disc_mask)
    if disc_sum == 0: return None
    return round(np.sum(cup_mask) / disc_sum, 3)

def cup_shape_index(disc_mask, cup_mask):
    disc_area = np.sum(disc_mask)
    cup_area = np.sum(cup_mask)
    rim_area = disc_area - cup_area
    if not (cup_area and rim_area > 0): return None
    return round(rim_area / cup_area, 3)

def area_ratio(disc_mask, cup_mask):
    rim_mask = disc_mask & (~cup_mask)
    ys, xs = np.where(disc_mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    
    inferior = np.sum(rim_mask[cy:, :])
    superior = np.sum(rim_mask[:cy, :])
    nasal = np.sum(rim_mask[:, cx:])
    temporal = np.sum(rim_mask[:, :cx])
    return int(inferior), int(superior), int(nasal), int(temporal)

def analyze_rim_geometry(disc_mask, cup_mask):
    rim_mask = disc_mask & (~cup_mask)
    rim_area = int(np.sum(rim_mask))
    return rim_area, None, None 

# --- Image Processing Function (UPDATED for GRI and 12-feature RF) ---
def process_image(image, processor, model, clf, gabor_bank, incremental_pca):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(rgb, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
    pred = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)
    disc_mask, cup_mask = (pred == 1), (pred == 2)
    overlay = rgb.copy()
    overlay[disc_mask] = [255, 255, 0] # Disc: Yellow
    overlay[cup_mask] = [255, 0, 0]   # Cup: Red
    
    # 1. Clinical Metrics Calculation
    vcdr = vertical_cdr(disc_mask, cup_mask)
    acdr = acdr_area(disc_mask, cup_mask)
    ddls = calculate_ddls(disc_mask, cup_mask) 
    inf, sup, nas, temp = area_ratio(disc_mask, cup_mask)
    rim_area, rim_min, rim_max = analyze_rim_geometry(disc_mask, cup_mask)
    disc_area = int(np.sum(disc_mask))
    cup_area = int(np.sum(cup_mask))
    y_disc = np.where(disc_mask)[0]
    y_cup = np.where(cup_mask)[0]
    disc_vert_height = int(y_disc.max() - y_disc.min() + 1) if len(y_disc) else 0
    cup_vert_height = int(y_cup.max() - y_cup.min() + 1) if len(y_cup) else 0
    x_disc = np.where(disc_mask)[1]
    x_cup = np.where(cup_mask)[1]
    disc_hori_width = int(x_disc.max() - x_disc.min() + 1) if len(x_disc) else 0
    cup_hori_width = int(x_cup.max() - x_cup.min() + 1) if len(x_cup) else 0
    
    # 2. Glaucoma Risk Index (GRI) Calculation
    GRI = None
    pc_means = {'PC1': None, 'PC2': None, 'PC3': None, 'PC4': None, 'PC5': None}
    
    if incremental_pca is not None and gabor_bank:
        try:
            rgb_for_gabor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Xnorm = extract_features(rgb_for_gabor, gabor_bank, TARGET_WIDTH, TARGET_HEIGHT, fixed_wavelength, fixed_orientation)
            
            principal_components = incremental_pca.transform(Xnorm)

            if principal_components.shape[1] >= 5:
                PC1 = principal_components[:, 0]
                PC2 = principal_components[:, 1]
                PC3 = principal_components[:, 2]
                PC4 = principal_components[:, 3]
                PC5 = principal_components[:, 4]

                pc_means = {
                    'PC1': np.mean(PC1), 'PC2': np.mean(PC2), 'PC3': np.mean(PC3), 
                    'PC4': np.mean(PC4), 'PC5': np.mean(PC5)
                }

                # GRI Calculation Formula
                GRI = 6.8375 - 1.1325 * pc_means['PC1'] + 1.6500 * pc_means['PC2'] + 2.7225 * pc_means['PC3'] + 0.6750 * pc_means['PC4'] + 0.6650 * pc_means['PC5']
            else:
                 GRI = None 
        except Exception:
            GRI = None

    # 3. Random Forest (RF) Prediction (12 Features: 11 Clinical + GRI)
    prediction_label, confidence = 'N/A', 'N/A'
    
    # List of 12 features needed for RF model (11 Clinical + GRI)
    clinical_features_for_rf = [vcdr, acdr, ddls, inf, disc_area, cup_area, rim_area, disc_vert_height, cup_vert_height, disc_hori_width, cup_hori_width, GRI]
    
    if clf is not None and all(x is not None for x in clinical_features_for_rf):
        feature_data = {
            'VCDR': [vcdr], 'ACDR': [acdr], 'DDLS': [ddls], 'INFERIOR_AREA':[inf], 
            'DISC_AREA':[disc_area], 'CUP_AREA':[cup_area], 'RIM_AREA':[rim_area],
            'DISC_VERTICAL_HEIGHT':[disc_vert_height], 'CUP_VERTICAL_HEIGHT':[cup_vert_height],
            'DISC_HORIZONTAL_WIDTH':[disc_hori_width], 'CUP_HORIZONTAL_WIDTH':[cup_hori_width],
            'GRI': [GRI] # <--- GRI is the 12th feature
        }
        features = pd.DataFrame(feature_data)
        
        # FIX: Convert DataFrame to NumPy array to avoid sklearn UserWarning on feature names
        features_array = features.to_numpy() 
        
        prob = clf.predict_proba(features_array)[0]
        prediction = np.argmax(prob)
        prediction_label = "Glaucoma" if prediction == 1 else "Normal"
        confidence = prob[prediction]
        
    return rgb, overlay, {
        # 11 Clinical Features
        "VCDR": vcdr, "ACDR": acdr,"DDLS": ddls, "INFERIOR_AREA":inf, "DISC_AREA":disc_area, "CUP_AREA":cup_area,"RIM_AREA":rim_area,"DISC_VERTICAL_HEIGHT":disc_vert_height,"CUP_VERTICAL_HEIGHT":cup_vert_height,'DISC_HORIZONTAL_WIDTH':disc_hori_width,'CUP_HORIZONTAL_WIDTH':cup_hori_width,
        # RF Prediction
        "Prediction": prediction_label, "Confidence": confidence,
        # PCA/GRI Results (GRI value is also one of the RF features)
        "GRI": GRI,
        **pc_means
    }

# --- LOGIN/ABOUT/CONTACTS PAGE LOGIC (Kept as is for brevity) ---
if not st.session_state.get("authentication_status"):
    if 'page_view' not in st.session_state:
        st.session_state.page_view = 'login'

    st.markdown("""
        <style>
            .block-container { padding: 2rem 5rem 2rem 5rem !important; }
            hr { margin-top: 0 !important; margin-bottom: 2rem !important; }
            div[data-testid="stForm"] label { font-size: 0px !important; }
            div[data-testid="stTextInput"] label[for*="Username"]::before {
                content: "Email Address/LoginID"; font-size: 1.1rem !important; font-weight: bold; color: #00008B;
            }
            div[data-testid="stTextInput"] label[for*="Password"]::before {
                content: "Password"; font-size: 1.1rem !important; font-weight: bold; color: #00008B;
            }
            input[type="text"], input[type="password"] {
                border: 2px solid #00008B !important; border-radius: 10px !important; height: 50px !important;
            }
            div[data-testid="stImage"] > img { display: block; margin-left: auto; margin-right: auto; }
        </style>
    """, unsafe_allow_html=True)

    nav1, nav2, nav3, _ = st.columns([0.15, 0.25, 0.3, 0.3])
    with nav1:
        button_type = "primary" if st.session_state.page_view == 'login' else "secondary"
        if st.button("Home Page", type=button_type, use_container_width=True):
            st.session_state.page_view = 'login'; st.rerun()
    with nav2:
        button_type = "primary" if st.session_state.page_view == 'about' else "secondary"
        if st.button("About the Project", type=button_type, use_container_width=True):
            st.session_state.page_view = 'about'; st.rerun()
    with nav3:
        button_type = "primary" if st.session_state.page_view == 'contacts' else "secondary"
        if st.button("Acknowledgement & Contacts", type=button_type, use_container_width=True):
            st.session_state.page_view = 'contacts'; st.rerun()

    st.write("") 

    if st.session_state.page_view == 'login':
        header_cols = st.columns([1, 2])
        with header_cols[0]:
            try: st.image("image.png")
            except FileNotFoundError: st.error("Logo file 'image.png' not found.")
        with header_cols[1]:
            st.markdown("<h1 style='text-align: center; color: #00008B;'>Glaucoma Screening from Retinal Fundus Images</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        body_cols = st.columns([1, 1])
        with body_cols[0]:
            authenticator.login()
            if st.session_state.get("authentication_status") is False:
                st.error('Username/password is incorrect')
        with body_cols[1]:
            st.markdown("<h3 style='text-align: center;'>Why Glaucoma is Serious – Some Facts</h3>", unsafe_allow_html=True)
            info_cols = st.columns([1, 2])
            with info_cols[0]:
                try: st.image("glaucoma.png")
                except FileNotFoundError: st.error("Diagram file 'glaucoma.png' not found.")
            with info_cols[1]:
                st.markdown("""
                **Blindness Can Be Prevented By Following Doctor/Ophthalmologist Instructions**
                - Glaucoma is like diabetes or hypertension, no pain or symptoms and it can’t be cured, but regular medication can keep it in control.
                - There about 12 Million people with Glaucoma in India. Only half of them aware of it.
                - For every person diagnosed to have Glaucoma, there is another person with undetected Glaucoma.
                - Many people don’t know they have Glaucoma, until they start to lose 50% of their eye sight, gradually however the doctor can detect and treat Glaucoma before most patients experience any symptoms.
                - Patients with glaucoma usually have less field of vision (total area of sight) when they have glaucoma and when they have lost all of the visual field, they are prone to blindness.
                - In Glaucoma all efforts are aimed to preserve the existing vision of a person.
                - Glaucoma is hereditary. All patients with Glaucoma should inform their family members to get screened for Glaucoma
                """)
        
         
        st.markdown("<br><br>", unsafe_allow_html=True)
        footer_cols = st.columns(3)
        
        # FIX: Ensure all images are placed within the respective columns with explicit widths
        # Column 0: Two logos side by side (Each takes ~half the column width)
        with footer_cols[0]:
            st.markdown("<h4 style='text-align: center;'>Funding Support</h4>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                try: st.image("hub.png", use_container_width=True)
                except FileNotFoundError: st.error("File 'hub.png' not found.")
            with col_b:
                try: st.image("money.png", use_container_width=True)
                except FileNotFoundError: st.error("File 'money.png' not found.")
                
        # Column 1: One medium-sized logo
        with footer_cols[1]:
            st.markdown("<h4 style='text-align: center;'>Project Development & Execution</h4>", unsafe_allow_html=True)
            try: st.image("mahindra university.png", use_container_width=True)
            except FileNotFoundError: st.error("File 'mahindra university.png' not found.")
            
        # Column 2: One circular logo (smaller than the max width of the column)
        with footer_cols[2]:
            st.markdown("<h4 style='text-align: center;'>Support for Data Collection</h4>", unsafe_allow_html=True)
            try: st.image("government of telangna.png")
            except FileNotFoundError: st.error("File 'logo_data_telangana.png' not found.")


    elif st.session_state.page_view == 'about':
        st.markdown("<h3 style='color: red;'>About the Project Page</h3>", unsafe_allow_html=True)
        st.markdown("""
        **GlauMitra AI** is an initiative driven by the need for advanced tools to enable early and accurate diagnosis of glaucoma in Indian patients, particularly in resource-limited settings, to ensure timely referral to ophthalmologists.
        - Early and accurate diagnosis of glaucoma is critical for effective treatment and the prevention of irreversible blindness. With this goal, we developed GlauMitra AI—an advanced artificial intelligence system designed to automatically detect early signs of glaucoma from retinal fundus images of Indian patients, specifically from Telangana State, using both conventional and handheld fundus cameras.
        - To build this system, we employed state-of-the-art AI and image processing techniques. Our model was trained on a self-curated dataset of approximately 15,000 fundus images—comprising both Glaucoma and Non-Glaucoma cases—collected from patients in the Nalgonda and Hyderabad districts. The dataset was split 80:20 for training and testing purposes, and the resulting system achieved an impressive 90% classification accuracy.
        - We envision **GlauMitra AI** as a valuable screening tool for early glaucoma detection, especially in resource-limited settings, enabling timely referrals to ophthalmologists and improving patient outcomes.
        """)
    
    elif st.session_state.page_view == 'contacts':
        st.markdown("<h3 style='color: red;'>Acknowledgement & Contacts Page</h3>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #00008B;'>Contact Information</h4>", unsafe_allow_html=True)
        st.markdown("Feel free to reach out to us with any inquiries or feedback.")
        
        contact_table = """
        <table style="width:100%; border-collapse: collapse;">
            <tr style="border: 1px solid #ddd;">
                <th style="padding: 8px; text-align: left; background-color: #f2f2f2; border: 1px solid #ddd;">Project Investigator and Co-Project Investigators</th>
                <th style="padding: 8px; text-align: left; background-color: #f2f2f2; border: 1px solid #ddd;">E-mail</th>
            </tr>
            <tr style="border: 1px solid #ddd;">
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">Dr. Bipin Singh (PI)<br>Assistant Professor, Centre for Life Sciences, Mahindra University,<br>Hyderabad, Telangana</td>
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">bipin.singh@mahindrauniversity.edu.in</td>
            </tr>
            <tr style="border: 1px solid #ddd;">
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">Dr. Santosh Thakur (Co-PI)<br>Assistant Professor, Centre for Life Sciences, Mahindra University,<br>Hyderabad, Telangana</td>
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">santosh.thakur@mahindrauniversity.edu.in</td>
            </tr>
            <tr style="border: 1px solid #ddd;">
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">Dr. Superna Mahendra (Collaborator)<br>Civil Surgeon Ophthalmologist, Government General Hospital,<br>Nalgonda, Telangana</td>
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">drsuperna95@gmail.com</td>
            </tr>
            <tr style="border: 1px solid #ddd;">
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">Mr. Mohit Bisaria<br>Senior Research Fellow, Centre for Life Sciences, Mahindra University,<br>Hyderabad, Telangana</td>
                <td rowspan-2 style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;"></td>
            </tr>
            <tr style="border: 1px solid #ddd;">
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">Mr. Sujal Shinde<br>BTech Final Year Student<br>Research Intern, Centre for Life Sciences, Mahindra University,<br>Hyderabad</td>
            </tr>
        </table>
        """
        st.markdown(contact_table, unsafe_allow_html=True)
        
        st.markdown("<br><h4 style='color: #00008B;'>Acknowledgements</h4>", unsafe_allow_html=True)
        
        st.markdown("""
        - Honourable Komatireddy Venkat Reddy, Minister of Roads, Buildings and Cinematography,Government of Telangana.
        - Mr. Jayesh Ranjan, Special Chief Secretary, Government of Telangana.
        - Mr. Bhavesh Mishra,IAS,Deputy Secretary IT Government of Telangana.
        - Smt Ila Tripathi,IAS District Collector and District Magistrate Nalgonda, Telangana.
        - DM&HO Dr.Putla Srinivas,DCH
        - Dr. G Ranjit Kumar,practicing optometrist & Visual Fields, Fellow in Optometry LV Prasad Eye Institute (LVPEI) 
        """)

# --- MAIN APPLICATION (Runs only after successful login) ---
elif st.session_state["authentication_status"]:

    # Load all resources (models, Gabor bank, PCA)
    processor, model, clf, gabor_bank, incremental_pca = load_all_resources()

    header_cols = st.columns([1.5, 3, 1])
    with header_cols[0]:
        st.markdown("<p style='color: red; font-size: 1.2rem; font-weight: bold;'>Prediction Page</p>", unsafe_allow_html=True)
        try: st.image(("image.png"), width=300)
        except FileNotFoundError: st.error("Logo 'image.png' not found.")
    with header_cols[1]:
        st.markdown("""
            <div style='background-color: #f0f2f6; border-radius: 10px; padding: 1rem; margin-top: 2rem; display: flex; align-items: center; justify-content: center; height: 100px;'>
                <h2 style='color: #0d6efd; text-align: center; font-weight: bold;'>Glaucoma Screening from Retinal Fundus Images</h2>
            </div>
        """, unsafe_allow_html=True)
    with header_cols[2]:
        st.markdown(f"<p style='text-align: right; margin-top: 2rem;'>Welcome <i>{st.session_state['name']}</i></p>", unsafe_allow_html=True)
        authenticator.logout('Logout', 'main')

    st.markdown("---")

    # Patient Details Input
    st.markdown("##### Patient Details")
    row1_cols = st.columns(2)
    with row1_cols[0]:
        patient_name = st.text_input("Patient Name")
    with row1_cols[1]:
        age = st.selectbox("Age", options=list(range(1, 101)))
    row2_cols = st.columns(2)
    with row2_cols[0]:
        gender = st.selectbox("Gender", options=["Male", "Female", "Others"])
    with row2_cols[1]:
        comorbidities = st.multiselect("Comorbidities", ["BP", "Diabetes","COPD","IHD","CVA","GLAUCOMA FAMILY HISTORY","TRAUMA"], placeholder="Select (optional)")

    st.markdown("---")
    st.markdown("##### Upload a retinal image")
    uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file and clf is not None:
        if not patient_name:
            st.warning("Please enter a Patient Name to generate a report.")
        else:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            with st.spinner("Analyzing image and calculating metrics..."):
                original, overlay, metrics = process_image(image, processor, model, clf, gabor_bank, incremental_pca)
            
            # --- Prepare DataFrame ---
            metrics_data_full = {
                "VCDR": metrics.get("VCDR"), "ACDR": metrics.get("ACDR"), "DDLS": metrics.get("DDLS"),
                "INFERIOR_AREA":metrics.get("INFERIOR_AREA"), "DISC_AREA":metrics.get("DISC_AREA"),
                "CUP_AREA":metrics.get("CUP_AREA"), "RIM_AREA":metrics.get("RIM_AREA"),
                "DISC_VERTICAL_HEIGHT":metrics.get("DISC_VERTICAL_HEIGHT"), 
                "CUP_VERTICAL_HEIGHT":metrics.get("CUP_VERTICAL_HEIGHT"),
                "DISC_HORIZONTAL_WIDTH":metrics.get("DISC_HORIZONTAL_WIDTH"),
                "CUP_HORIZONTAL_WIDTH":metrics.get("CUP_HORIZONTAL_WIDTH"),
                "GRI": metrics.get("GRI"), # GRI is the 12th feature/metric
                "PC1": metrics.get("PC1"), "PC2": metrics.get("PC2"), "PC3": metrics.get("PC3"),
                "PC4": metrics.get("PC4"), "PC5": metrics.get("PC5"),
                "Prediction": metrics.get("Prediction"), "Confidence": metrics.get("Confidence")
            }
            metrics_df = pd.DataFrame([metrics_data_full])
            
            # Drop PC columns for Streamlit display/PDF table, as requested
            cols_to_drop = [f"PC{i}" for i in range(1, 6)]
            metrics_df_display = metrics_df.drop(columns=cols_to_drop, errors='ignore')

            # --- Chart Generation (UPDATED for all 12 features) ---
            glaucoma_metrics_for_chart = {k: v for k, v in metrics_df_display.iloc[0].to_dict().items() 
                                         if k not in ["Prediction", "Confidence"] and v is not None}
            
            fig = None
            if glaucoma_metrics_for_chart:
                fig = go.Figure(data=[go.Bar(
                    x=list(glaucoma_metrics_for_chart.keys()), 
                    y=list(glaucoma_metrics_for_chart.values()), 
                    marker_color='teal', 
                    # Use appropriate formatting for text labels
                    text=[(f"{v:.3f}" if k in ["VCDR", "ACDR", "DDLS", "GRI"] else f"{int(v)}") 
                          for k, v in glaucoma_metrics_for_chart.items()], 
                    textposition='outside'
                )])
                fig.update_layout(
                    title="Glaucoma-Specific Metrics", 
                    height=600, 
                    yaxis=dict(title="Value (Ratio or Pixel Count)")
                )

            patient_info = {
                "Patient Name": patient_name, "Age": age, "Gender": gender,
                "Comorbidities": ", ".join(comorbidities) if comorbidities else "None"
            }
            
            # Download button
            pdf_bytes = create_report_pdf(patient_info, original, overlay, metrics_df_display, fig, metrics['GRI'])
            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_bytes,
                file_name=f"Glaucoma_Report_{patient_name.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )

            # Display Images
            col1, col2 = st.columns(2)
            with col1:
                st.image(original, caption="Original Image", use_container_width=True)
            with col2:
                st.image(overlay, caption="Segmented Optic Disc (Yellow) and Cup (Red)", use_container_width=True)
            
            # Display GRI Result (***REMOVED PART START***)
            
            # Display Clinical Prediction (***REMOVED PART END***)
                 
            st.markdown("---")
            st.markdown("##### Analysis Results")
            
            # Format display dataframe
            format_dict = {
                c: "{:.3f}" for c in ["VCDR", "ACDR", "DDLS", "GRI", "Confidence"]
            }
            for col in ["INFERIOR_AREA", "DISC_AREA", "CUP_AREA", "RIM_AREA", "DISC_VERTICAL_HEIGHT", "CUP_VERTICAL_HEIGHT", "DISC_HORIZONTAL_WIDTH", "CUP_HORIZONTAL_WIDTH"]:
                if col in metrics_df_display.columns:
                    format_dict[col] = "{:d}" 

            st.dataframe(metrics_df_display.style.format(format_dict), use_container_width=True)

            st.markdown(f"### Prediction: **{metrics['Prediction']}**")
            if isinstance(metrics['Confidence'], (float, np.floating)):
                 st.markdown(f"**Confidence:** {metrics['Confidence'] * 100:.1f}%")
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
