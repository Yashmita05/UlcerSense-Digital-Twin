# UlcerSense: Explainable Digital Twin for Proactive Diabetic Foot Ulcer (DFU) Risk Monitoring

## **1. Team Details**

- **Team Name:** Nexora
- **Team Lead:** Yashmita Bhushan

## **2. College Information**

- **College Name:** Indira Gandhi Delhi Technical University for Women (IGDTUW)

## **3. Project Title**

**UlcerSense: Explainable Digital Twin for Proactive Diabetic Foot Ulcer (DFU) Risk Monitoring**

## **4. Problem Statement**

Diabetic Foot Ulcers (DFUs) are serious complications associated with diabetes and can lead to chronic infections, tissue damage, and lower-limb amputations. Traditional healthcare approaches may identify problems only after visible skin breakdown or other symptoms appear, making early intervention challenging.

UlcerSense aims to explore a proactive monitoring approach by combining plantar thermal images with selected patient health indicators. The objective is to analyze localized temperature patterns and generate risk estimates that may help identify potentially concerning changes requiring further clinical evaluation.

## **5. Healthcare Use Case**

UlcerSense focuses on **proactive diabetic foot ulcer risk monitoring** using a Digital Twin concept.

The system combines plantar thermal image analysis with selected Electronic Health Record (EHR) indicators, including age, HbA1c, and diabetes duration. The foot is divided into three anatomical regions:

- **Toes**
- **Midfoot**
- **Heel**

The system extracts regional temperature features and combines them with patient indicators to generate a model-based risk estimate. Explainable AI helps identify which input features contributed to the prediction.

The dashboard is designed to display a three-zone foot heatmap, risk trends, feature contributions, and telemetry history. This approach aims to support monitoring and further clinical review rather than replace professional medical assessment.

## **6. Technical Stack**

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python 3.11 | Data processing and machine learning pipeline |
| Image Processing | OpenCV, NumPy | Thermal image processing and regional feature extraction |
| Data Processing | Pandas | Data aggregation and feature preparation |
| Machine Learning | XGBoost, XGBClassifier | Diabetic foot ulcer risk estimation |
| Explainable AI | SHAP, TreeExplainer | Explaining feature contributions to model outputs |
| Backend | FastAPI, Uvicorn | API endpoints and model-serving layer |
| Frontend | HTML5, Tailwind CSS | Dashboard structure and styling |
| Data Visualization | Chart.js | Telemetry and risk trend visualization |

## **7. AI/ML Model or Framework Details**

### **Dataset Used**

UlcerSense uses the **Thermography Images of Diabetic Foot** dataset available on Kaggle to support thermal image analysis and diabetic foot monitoring.

- **Dataset Name:** Thermography Images of Diabetic Foot
- **Dataset Source:** [Kaggle — Thermography Images of Diabetic Foot](https://www.kaggle.com/datasets/vuppalaadithyasairam/thermography-images-of-diabetic-foot?resource=download)
- **Purpose:** To support the development and evaluation of thermal image-processing and diabetic foot monitoring features.
- **Processing Tools:** OpenCV and NumPy.

The dataset is used as the image-data source for the project. Its labels and metadata determine which supervised learning tasks can be supported. Predicting future ulcer development requires appropriate outcome labels and a suitable evaluation methodology.

### **XGBoost Classifier**

UlcerSense uses an XGBoost classifier for its machine learning prediction component.

The intended input features include regional temperature measurements from the toes, midfoot, and heel, along with available patient indicators such as age, HbA1c, and diabetes duration.

The model processes these features to generate a risk estimate. Reliable risk probabilities depend on appropriate labelled data, model training, testing, and probability calibration.

### **SHAP Explainability**

SHAP (SHapley Additive exPlanations) with TreeExplainer is used to explain the contribution of individual input features to a model output.

This helps users inspect whether regional temperature measurements or patient indicators influenced a particular prediction. SHAP improves the interpretability of model outputs but does not establish medical causation.

### **OpenCV-Based Image Processing**

OpenCV and NumPy are used to process plantar thermal images, isolate the foot region, and extract features from three anatomical zones:

- Toes
- Midfoot
- Heel

The extracted regional features can be combined with available patient indicators for downstream analysis. Converting pixel intensities into actual physiological temperatures requires appropriate calibration or validated image metadata.

### **Backend and Dashboard**

FastAPI and Uvicorn provide the backend service layer, with the following endpoints:

- `/telemetry/stream-next` — provides the telemetry feed.
- `/analyze-image` — processes uploaded thermal images.
- `/predict` — generates model predictions.

The frontend uses HTML5, Tailwind CSS, and Chart.js to display the regional foot heatmap, risk trends, feature explanations, and telemetry history.

### **Unique Selling Points (USP)**

- **Localized Foot Analysis:** Divides the plantar region into toes, midfoot, and heel for focused analysis.
- **Explainable AI:** Uses SHAP to explain the contribution of individual features to model outputs.
- **Time-Aware Monitoring:** Organizes telemetry history and trends to compare observations over time.
- **Digital Twin Concept:** Combines thermal observations and patient indicators into a virtual patient monitoring profile.
- **Human-in-the-Loop Approach:** Supports further clinical review rather than relying entirely on automated decisions.

## **8. Demo Video**

The project demonstration video presents the UlcerSense concept, technical workflow, and prototype.

**Demo Video Link:** [Watch the UlcerSense Demo](https://drive.google.com/file/d/1ywFGI-JKkLXintlhWQGInx8t1tklY-B9/view?usp=sharing)

## **9. Open-Source License Details**

**Proposed License: MIT License**

The MIT License permits others to use, copy, modify, distribute, and sublicense the software, subject to its conditions.

To apply this license, a file named `LICENSE` should be added to the root directory of the GitHub repository. It should contain the complete MIT License text with the correct year and copyright holder.

The repository should only be described as MIT-licensed after the license file has been added.

## **10. Architecture Diagram (PDF/PPT Format)**

The UlcerSense system architecture diagram illustrates the workflow from thermal image input and preprocessing to feature integration, machine learning prediction, explainability, backend processing, and dashboard visualization.

**Architecture Diagram:**[View UlcerSense Architecture Diagram (PPTX)](UlcerSense_Architecture.pptx)

## **11. Presentation (PDF/PPT Format)**

The UlcerSense presentation covers the project overview, problem statement, healthcare use case, technical stack, AI/ML model, unique selling points, system architecture, and responsible-use considerations.

* **Project Presentation:** [View UlcerSense Presentation (PDF)](UlcerSense.pdf)


## **Responsible Use and Limitations**

UlcerSense is a proof-of-concept project and is not a clinically validated diagnostic tool. Model reliability depends on suitable labelled data, appropriate training, testing, and calibration. Clinical thresholds and alerts require validation and healthcare professional guidance.

Thermal image analysis and model predictions must not replace professional medical advice, diagnosis, or treatment. Patient health records and thermal images should be handled securely, using consented or appropriately de-identified data.

---

**Team Nexora | UlcerSense**

*An explainable, time-aware Digital Twin concept for proactive diabetic foot ulcer risk monitoring.*
