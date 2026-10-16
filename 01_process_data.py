import os
import cv2
import numpy as np
import pandas as pd

def extract_foot_temperatures(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale to map intensity to temperature
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Mask out cold background (threshold out pixels close to ambient/black)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Approximate physical foot surface range: 26.0°C to 36.0°C
    temp_map = 26.0 + (gray.astype(np.float32) / 255.0) * 10.0
    
    # Isolate coordinates belonging to the foot
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Segment vertically into three primary regions of interest
    toes_region = temp_map[y : y + int(h * 0.33), x : x + w]
    toes_mask = mask[y : y + int(h * 0.33), x : x + w]
    
    midfoot_region = temp_map[y + int(h * 0.33) : y + int(h * 0.66), x : x + w]
    midfoot_mask = mask[y + int(h * 0.33) : y + int(h * 0.66), x : x + w]
    
    heel_region = temp_map[y + int(h * 0.66) : y + h, x : x + w]
    heel_mask = mask[y + int(h * 0.66) : y + h, x : x + w]
    
    # Calculate mean temperatures excluding black background
    toes_temp = toes_region[toes_mask > 0].mean() if np.any(toes_mask > 0) else 28.0
    mid_temp = midfoot_region[midfoot_mask > 0].mean() if np.any(midfoot_mask > 0) else 28.0
    heel_temp = heel_region[heel_mask > 0].mean() if np.any(heel_mask > 0) else 28.0
    
    return float(toes_temp), float(mid_temp), float(heel_temp)

def build_dataset(base_dir="."):
    records = []
    patient_id = 1
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                temps = extract_foot_temperatures(img_path)
                if temps is None:
                    continue
                
                toes_t, mid_t, heel_t = temps
                is_diabetic = 1 if "diabetic" in root.lower() else 0
                
                # Synthetic EHR alignment
                if is_diabetic:
                    age = int(np.random.normal(58, 8))
                    hba1c = float(np.clip(np.random.normal(8.6, 1.2), 6.5, 14.0))
                    duration_dm = int(np.clip(np.random.normal(12, 5), 1, 35))
                    # Ulcer risk label based on clinical threshold: elevated focal temp (>32.5°C) & high HbA1c
                    ulcer_risk = 1 if (heel_t > 32.2 or toes_t > 32.2) and hba1c > 8.0 else 0
                else:
                    age = int(np.random.normal(44, 10))
                    hba1c = float(np.clip(np.random.normal(5.4, 0.4), 4.5, 6.2))
                    duration_dm = 0
                    ulcer_risk = 0
                
                records.append({
                    "patient_id": f"PT_{patient_id:04d}",
                    "file_name": file,
                    "toes_temp": round(toes_t, 2),
                    "midfoot_temp": round(mid_t, 2),
                    "heel_temp": round(heel_t, 2),
                    "age": age,
                    "hba1c": round(hba1c, 1),
                    "diabetes_years": duration_dm,
                    "ulcer_risk": ulcer_risk
                })
                patient_id += 1

    df = pd.DataFrame(records)
    df.to_csv("digital_twin_patient_data.csv", index=False)
    print(f"Dataset generated with {len(df)} records -> digital_twin_patient_data.csv")

if __name__ == "__main__":
    build_dataset()