# AI Enabled Visa Status Prediction and Processing Time Estimator

## Milestone 1: Data Collection & Preprocessing

### Objective
To create a clean and structured dataset for predicting visa processing time.

### Tasks Completed
- Created synthetic visa application dataset
- Handled missing decision dates
- Converted date formats
- Generated target variable (processing time in days)

### Tools Used
- Python
- Pandas
- VS Code
- Git & GitHub

### Output
Processed dataset stored in:
data/processed/visa_cleaned_data.csv

# AI-Visa-Processing-Time-Estimator
This project aims to reduce uncertainty faced by visa applicants by improving transparency and enhancing the overall applicant experience. It follows a complete AI workflow including data collection, preprocessing, feature engineering, model training, and evaluation, making it suitable for academic, research, and demonstration purposes.


## Milestone 2: Exploratory Data Analysis (EDA)

### Objective
To analyze the cleaned visa dataset and identify trends, patterns, and key factors affecting visa processing time.

### Tasks Completed
- Visualized distribution of visa processing time
- Compared processing times across visa types
- Analyzed processing times by applicant country
- Identified seasonal trends in visa applications
- Examined workload differences across processing centers
- Performed feature importance analysis using Random Forest

### Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code
- Git & GitHub

### Output
- Multiple visualization plots generated using src/eda.py
- Feature importance values printed to console
- Insights used to guide machine learning model selection

## Milestone 3: Predictive Modeling

### Objective
To develop and evaluate a machine learning model that predicts visa processing time based on historical application data.

### Model Selected
**Random Forest Regressor**

Random Forest was chosen over Linear Regression because:
- Visa processing time depends on non-linear factors such as country, visa type, and processing center
- It handles categorical features effectively after encoding
- It provides better accuracy and robustness for small to medium datasets

### Dataset Used
Processed dataset generated from Milestone 1 & 2:

### Features Used
- Applicant Country (encoded)
- Visa Type (encoded)
- Processing Center (encoded)
- Visa Status (encoded)
- Application Month

### Target Variable
- `processing_time_days`

### Model Training Process
- Categorical features were encoded using Label Encoding
- Data was split into training (80%) and testing (20%)
- Random Forest Regressor was trained with optimized parameters
- Model performance was evaluated using standard regression metrics

### Evaluation Metrics
- **MAE (Mean Absolute Error)** – Measures average prediction error in days
- **RMSE (Root Mean Squared Error)** – Penalizes larger errors
- **R² Score** – Indicates how well the model explains variance in processing time

### Results
The Random Forest model achieved:
- Low MAE and RMSE values
- High R² score indicating good predictive performance

### Output
- Trained model saved as:


## Milestone 4: Web App Development & Deployment

### Objective
To build and deploy a user-friendly web application that estimates visa processing time using a trained ML model.

### Implementation
- Developed an interactive web application using Streamlit
- Integrated the trained Random Forest prediction model
- Designed input forms for visa details
- Displayed estimated processing time range with confidence interval

### Deployment
- Application deployed using Streamlit Cloud
- Supports real-time prediction based on user input

### Outcome
- Improved transparency for visa applicants
- Easy-to-use AI-powered estimator


### DEPLOYMENT LINK
https://ai-visa-processing-time-estimator.streamlit.app/

