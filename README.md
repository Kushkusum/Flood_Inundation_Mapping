
# Flood Inundation Mapping Using SAR, Optical, Terrain and Temporal Data

This repository contains the complete implementation of a flood inundation mapping system using multi-source satellite data and deep learning.

---

## Frontend Dashboard Preview
(Add your image in /assets folder and rename accordingly)

![Dashboard](assets/frontend_landing_page.png)

---

## Project Objective

- Flood mapping using Sentinel-1 SAR and Sentinel-2
- Integration of DEM terrain features
- Use of temporal rainfall and soil moisture
- Model comparison and generalization (Assam 2023)
- Interactive Streamlit dashboard

---

## Dataset

### Sen1Floods11
Download from:
gs://sen1floods11/v1.1/

Countries used:
- India
- Pakistan
- Sri Lanka
- Cambodia (Mekong)
- Bolivia
- Colombia

---

## Input Data

- Sentinel-1 (VV, VH)
- Sentinel-2 (B2, B3, B4, B8, B11, B12)
- DEM: Elevation, Slope, TWI, HAND
- Temporal: CHIRPS, ERA5 (15 days)

---

## Model Variants

### V1 - U-Net Spatial
14-channel input

### V2 - U-Net + ConvLSTM

### V5 - U-Net + Temporal MLP (Best)

---

## Pipeline

1. Download dataset
2. Export DEM (GEE)
3. Export temporal data
4. Preprocess (.npy)
5. Train models
6. Evaluate
7. Assam generalization
8. Dashboard visualization

---

## Folder Structure

Flood_Inundation_Mapping/
├── notebooks/
├── src/
├── dashboard/
├── assets/
├── results/
└── README.md

---

## System Requirements

- RAM: 16GB+
- GPU: T4/P100 recommended
- Storage: 100GB

---

## Metrics

- IoU
- F1
- Precision
- Recall

---

## Important Notes

- Mekong = Cambodia mapping required
- Do NOT recompute normalization
- Use rasterio for reprojection
- Mask invalid pixels (-1)

---

## Results

### Validation
V1 IoU: 0.7360  
V5 IoU: 0.7367  

### Assam 2023
V1 IoU: 0.3049  
V5 IoU: 0.2934  

---

## Contributors

- Kusum B S
- N Nishita

---

## License

MIT
