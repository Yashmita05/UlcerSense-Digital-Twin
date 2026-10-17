UlcerSense: Explainable Digital Twin for Proactive Diabetic Foot Ulcer (DFU) Risk Monitoring

1. Team Details

Team Name: Nexora

Team Lead: Yashmita Bhushan

2. College/Incubator Information

College Name: Indira Gandhi Delhi Technical University for Women (IGDTUW)

Incubator Name: N/A (update if applicable)

3. Project Title

UlcerSense: Explainable Digital Twin for Proactive Diabetic Foot Ulcer (DFU) Risk Monitoring

4. Problem Statement

Diabetic Foot Ulcers (DFUs) are serious complications associated with diabetes and can lead to chronic infections, tissue damage, and lower-limb amputations. Conventional care may identify problems only after visible skin breakdown or other symptoms appear, making early intervention challenging.

UlcerSense aims to explore a proactive monitoring approach by combining localized plantar temperature observations with selected patient health indicators. The objective is to identify potentially concerning patterns that may warrant further clinical evaluation.

5. Healthcare Use Case

UlcerSense focuses on proactive diabetic foot ulcer risk monitoring using a Digital Twin concept.

The system combines plantar thermal observations with selected Electronic Health Record (EHR) indicators, including age, HbA1c, and diabetes duration. The foot is divided into three anatomical regions:

Toes

Midfoot

Heel

The proposed system processes regional temperature features and patient indicators to generate a model-based risk estimate. Explainable AI helps identify which input features contributed to the model output.

The dashboard is designed to display regional temperature information, risk trends, feature contributions, and telemetry history. These insights are intended to support monitoring and further clinical review, not replace professional medical assessment.

6. Technical Stack

Component

Technology

Purpose

Programming Language

Python 3.11

Data processing and ML pipeline

Image Processing

OpenCV, NumPy

Image processing and regional feature extraction

Data Processing

Pandas

Data aggregation and feature preparation

Machine Learning

XGBoost, XGBClassifier

Model-based risk estimation

Explainable AI

SHAP, TreeExplainer

Feature contribution explanations

Backend

FastAPI, Uvicorn

API endpoints and model-serving layer

Frontend

HTML5, Tailwind CSS

Dashboard structure and styling

Data Visualization

Chart.js

Telemetry and risk trend charts

7. AI/ML Model or Framework Details

XGBoost Classifier

XGBoost is the proposed machine learning model for estimating diabetic foot ulcer risk. Its intended input features include regional temperatures from the toes, midfoot, and heel, along with patient indicators such as age, HbA1c, and diabetes duration.

The model processes these features to generate a risk estimate. Reliable risk probabilities require suitable labelled training data, model evaluation, and probability calibration.

SHAP Explainability

SHAP (SHapley Additive exPlanations) with TreeExplainer is intended to explain the contribution of individual input features to a model output. This helps users inspect which regional temperature measurements or patient indicators influenced a particular prediction.

SHAP explains model behaviour; it does not establish medical causation.

OpenCV-Based Image Processing

OpenCV and NumPy are intended to process plantar thermal images, isolate the plantar region, and extract features from the toes, midfoot, and heel.

Any conversion of pixel intensity into actual temperature must rely on appropriate calibration or validated image metadata.

Backend and Dashboard

FastAPI and Uvicorn provide the proposed backend service layer, with the following endpoints:

/telemetry/stream-next — telemetry feed

/analyze-image — image analysis

/predict — model prediction

The frontend uses HTML5, Tailwind CSS, and Chart.js to present the three-zone foot heatmap, risk trends, feature explanations, and telemetry history.

Unique Selling Points (USP)

Localized monitoring: Divides the foot into toes, midfoot, and heel regions.

Explainable AI: Uses SHAP to show feature contributions to model outputs.

Time-aware view: Organizes telemetry history and trends for comparing observations.

Digital Twin concept: Combines patient indicators and regional observations into a virtual monitoring profile.

Human-in-the-loop approach: Supports further review instead of fully automated clinical decisions.

8. Demo Video (Minimum 20 Minutes)

Demo Video Link:
https://drive.google.com/file/d/1ywFGI-JKkLXintlhWQGInx8t1tklY-B9/view?usp=sharing

Ensure the video is at least 20 minutes long and Google Drive access is set to Anyone with the link — Viewer.

9. Open-Source License Details

Proposed License: MIT License

The MIT License permits use, copying, modification, distribution, and sublicensing, subject to its conditions.

To apply this license, add a file named LICENSE to the root of the GitHub repository containing the complete MIT License text with the correct year and copyright holder.

Do not claim the repository is MIT-licensed until the license file has been added.

10. Architecture Diagram (PDF/PPT Format)

System Architecture

[ Plantar Thermal Image / Telemetry ]
                  |
                  v
       [ OpenCV Processing Pipeline ]
       - Image preprocessing
       - Plantar region isolation
       - Toes / Midfoot / Heel features
                  |
                  v
         [ Feature Integration ]
         - Regional temperature features
         - Age, HbA1c, diabetes duration
                  |
                  v
           [ XGBoost Classifier ]
           - Model-based risk estimate
                  |
                  v
            [ SHAP Explainer ]
            - Feature contribution analysis
                  |
                  v
            [ FastAPI Backend ]
            - /telemetry/stream-next
            - /analyze-image
            - /predict
                  |
                  v
        [ UlcerSense Dashboard ]
        - Three-zone foot heatmap
        - Risk trend charts
        - Feature explanations
        - Telemetry history

Architecture Diagram PDF: docs/UlcerSense_Architecture_Diagram.pdf

Upload the architecture diagram PDF to the repository's docs/ folder and ensure the path matches the file name.

11. Presentation (PDF/PPT Format)

The presentation covers the project overview, problem statement, healthcare use case, technical stack, AI/ML details, USP, architecture, and responsible-use considerations.

Presentation PDF: docs/UlcerSense_Nexora_Presentation.pdf

Presentation PowerPoint: docs/UlcerSense_Nexora_Presentation.pptx

Upload both presentation files to the repository's docs/ folder.

Important Note: Responsible Use

UlcerSense is a proof-of-concept project and is not a clinically validated diagnostic tool. Model reliability depends on suitable labelled data, training, testing, and calibration. Clinical thresholds and alerts require validation and healthcare professional guidance. The system must not replace professional medical advice, diagnosis, or treatment.
