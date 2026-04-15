# 🚀 AI-Powered Predictive Maintenance System with IoT & Real-Time Monitoring

## 📌 Overview

This project implements an **AI-powered predictive maintenance system** using machine learning and IoT sensor data from the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) turbofan engine dataset. The system predicts machine failures before they occur, enabling proactive maintenance and minimizing costly downtime.

The project features:
- **Machine Learning Model**: Random Forest classifier for failure prediction
- **Real-Time Dashboard**: Interactive Streamlit dashboard with live simulation
- **Data Pipeline**: Complete data preprocessing and feature engineering
- **High Accuracy**: ~96% prediction accuracy with strong failure detection capability

---

## ❗ Problem Statement

Industries worldwide face the critical challenge of **unexpected machine failures**, resulting in:
- ⏱️ Unexpected downtime and loss of productivity
- 💰 Enormous financial losses
- ⚠️ Safety hazards for personnel
- 📉 Reduced operational efficiency

This project solves this by **predicting failures before they occur**, enabling:
- ✅ Preventive maintenance scheduling
- ✅ Optimized resource allocation
- ✅ Reduced maintenance costs
- ✅ Improved safety and reliability

---

## 🏭 Industry Relevance

This predictive maintenance system is applicable to:
- **Manufacturing**: Factory equipment, assembly lines
- **Automotive**: Engine systems, transmission units
- **Energy Systems**: Power generation turbines, grid infrastructure
- **Aviation**: Aircraft engines, hydraulic systems
- **Heavy Equipment**: Construction machinery, mining equipment
- **Healthcare**: Medical imaging equipment, surgical systems

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 90.9%, C++ 4.5%, Cython 3.8%, C 0.8% |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, Random Forest |
| **Visualization** | Matplotlib, Seaborn |
| **Web Dashboard** | Streamlit |
| **Dataset** | NASA CMAPSS Turbofan Engine Data |

---

## 📊 Dataset

**Source**: NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)

**Description**: 
- Real turbofan engine run-to-failure simulation data
- 21,000+ sensor readings from multiple engines
- Multiple condition monitoring sensors
- Target: Remaining Useful Life (RUL) prediction

**Data Characteristics**:
- Time-series format with degradation patterns
- Multiple operating conditions
- Realistic failure progression modeling

---

## 🏗️ Architecture
RAW DATA 
↓ 
├─→ Data Loading 
├─→ Exploratory Analysis 
↓ 
DATA PREPROCESSING 
↓ 
├─→ Handle Missing Values 
├─→ Remove Outliers 
├─→ Normalize Features 
↓ FEATURE ENGINEERING 
↓ 
├─→ Create New Features 
├─→ Select Relevant Features 
├─→ Scale All Features 
↓ MODEL TRAINING 
↓ 
├─→ Split Data (Train/Test) 
├─→ Train Random Forest 
├─→ Cross-Validation 
↓ MODEL EVALUATION 
↓ 
├─→ Calculate Metrics 
├─→ Generate Confusion Matrix 
├─→ Analyze Performance 
↓ PREDICTIONS & DASHBOARD 
↓ 
└─→ Real-Time Monitoring 
└─→ Failure Alerts 
└─→ Historical Analysis


**Pipeline Components**:
1. **Data Loading**: Import NASA CMAPSS dataset
2. **Preprocessing**: Handle missing values, normalize features
3. **Feature Engineering**: Create domain-relevant features
4. **Model Training**: Random Forest classifier
5. **Evaluation**: Comprehensive metrics and validation
6. **Dashboard**: Real-time monitoring interface

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- Virtual environment (recommended)

### Step-by-Step Setup

**1. Clone the repository:**
git clone https://github.com/Aniketsatpathy/AI-Predictive-Maintenance-IoT.git
cd AI-Predictive-Maintenance-IoT

2. Create a virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:

bash
pip install -r requirements.txt
▶️ Usage
Running the Complete Pipeline
1. Execute the data preprocessing and model training:

bash
python main.py
This will:

Load the NASA dataset
Preprocess and prepare the data
Train the Random Forest model
Generate evaluation metrics and plots
Save outputs to outputs/ directory
2. Launch the interactive dashboard:

bash
streamlit run app/app.py
Access the dashboard at http://localhost:8501

Dashboard Features
📊 Real-time sensor data visualization
🎯 Failure prediction results
📈 Model performance metrics
⚙️ Interactive parameter adjustment
📉 Historical trend analysis
📈 Performance Results

Model Metrics
Accuracy: ~96%
Precision: High accuracy in identifying failures
Recall: Strong failure detection capability
AUC-ROC: Excellent discrimination between classes

Evaluation Metrics

![Classification Report](https://github.com/Aniketsatpathy/AI-Predictive-Maintenance-IoT/blob/master/images/Classification_report.png?raw=true)

Confusion Matrix Analysis
This matrix shows:

True Negatives: Correctly identified normal operation
True Positives: Correctly identified failures
False Negatives: Missed failures (critical)
False Positives: False alarms
Prediction Distribution
Prediction Distribution

📸 Project Visualizations & Screenshots
1. Dataset Preview
Overview of the raw NASA CMAPSS sensor data structure and characteristics.

![Dataset Preview](https://github.com/Aniketsatpathy/AI-Predictive-Maintenance-IoT/blob/master/images/Dataset_preview.png?raw=true)

This visualization shows:

Raw sensor readings from turbofan engines
Multiple condition monitoring parameters
Time-series progression
Data distribution across operational modes
2. Preprocessing & Feature Engineering Output
Processed and engineered features ready for machine learning model training.

![Processed Data Preview](https://github.com/Aniketsatpathy/AI-Predictive-Maintenance-IoT/blob/master/images/preprocessed_data_preview.png?raw=true)

Key preprocessing steps:

Normalization and scaling
Missing value handling
Feature selection and engineering
Temporal feature creation
Target variable labeling
3. Model Classification Report
Detailed metrics showing model performance across all classes.

![Classification Report](https://github.com/Aniketsatpathy/AI-Predictive-Maintenance-IoT/blob/master/images/Classification_report.png?raw=true)

Metrics included:

Precision, Recall, F1-Score
Support (sample count per class)
Weighted averages
Class-specific performance
4. Failure Trend Analysis
Temporal patterns and trends in engine failures throughout the dataset.

![Failure Trends](https://github.com/Aniketsatpathy/AI-Predictive-Maintenance-IoT/blob/master/images/Failure_trends.png?raw=true)

Analysis provides:

Failure progression over time
Seasonal patterns in failures
Degradation curve visualization
Maintenance interval optimization
5. Interactive Dashboard Interface
User-friendly Streamlit dashboard for real-time monitoring and predictions.

![Dashboard UI](https://github.com/Aniketsatpathy/AI-Predictive-Maintenance-IoT/blob/master/images/Dashboard_UI.png?raw=true)

Dashboard capabilities:

Real-time sensor monitoring
Prediction display
Historical data visualization
Alert system for impending failures
Interactive parameter controls
Performance metrics tracking

6. Dashboard Live Demo
Video demonstration of the interactive dashboard in action.

📹 Video: Dashboard Demo

The demo showcases:

Live data streaming simulation
Real-time prediction updates
Dashboard responsiveness
Feature interactions
Alert triggering mechanism

🎯 Key Learning Outcomes
Through this project, you'll learn:

Time-Series Machine Learning

Sequential data analysis
Temporal patterns recognition
RUL (Remaining Useful Life) prediction techniques
Feature Engineering

Domain knowledge application
Statistical feature creation
Feature selection methods
Dimensionality reduction
Model Development & Evaluation

Random Forest classifier implementation
Cross-validation strategies
Hyperparameter tuning
Imbalanced dataset handling
Dashboard Development

Streamlit framework mastery
Real-time data visualization
Interactive UI design
Performance optimization
Industrial AI Applications

IoT sensor data processing
Predictive maintenance strategies
Cost-benefit analysis
Scalability considerations
Deployment best practices


📁 Project Structure
AI-Predictive-Maintenance-IoT/
├── data/
│   └── nasa_cmapss/          # NASA dataset files
├── src/
│   ├── preprocessing.py       # Data cleaning & preparation
│   ├── feature_engineering.py # Feature creation
│   ├── model.py              # Model training & evaluation
│   └── utils.py              # Utility functions
├── app/
│   └── app.py                # Streamlit dashboard
├── outputs/
│   ├── plots/                # Generated visualizations
│   │   ├── confusion_matrix.png
│   │   └── prediction_distribution.png
│   └── models/               # Trained model files
├── images/                   # Project documentation images
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

📅 Development Timeline
Day 1 — Project Initialization & Setup
Tasks Completed:

Repository created and configured
Folder structure established
Initial project documentation
Environment setup
Commit: Initial project setup

Day 2 — Data Integration
Tasks Completed:

NASA CMAPSS dataset integrated
Data loading module implemented
Dataset exploration and analysis
Data quality assessment
Commit: Added dataset and loading module

Output: Raw dataset ready for preprocessing

Day 3 — Data Preprocessing & Labeling
Tasks Completed:

Data cleaning implementation
Missing value handling
Outlier detection and treatment
Target variable (failure) labeling
Data normalization
Commit: Implemented preprocessing and labeling

Output: Clean, labeled dataset ready for modeling

Day 4 — Model Development
Tasks Completed:

Random Forest model implementation
Hyperparameter configuration
Model training on preprocessed data
Training/validation split
Initial performance evaluation
Commit: Trained Random Forest model

Output: Trained ML model achieving ~96% accuracy

Day 5 — Comprehensive Evaluation
Tasks Completed:

Classification metrics computation (Precision, Recall, F1-Score)
Confusion matrix generation
ROC-AUC analysis
Cross-validation evaluation
Visualization generation
Commit: Added evaluation metrics and plots

Output: Detailed performance analysis with visualizations

Generated Plots:

Confusion Matrix Analysis
Prediction Distribution
Classification Report
Feature Importance Charts

Day 6 — Dashboard Development
Tasks Completed:

Streamlit dashboard framework setup
Real-time data simulation module
Interactive UI components
Real-time prediction integration
Alert system implementation
Historical data visualization
Commit: Built dashboard UI with real-time simulation

Output: Fully functional interactive dashboard

Dashboard Features:

Real-time sensor monitoring
Live failure predictions
Performance metrics display
Historical trend analysis
Alert notifications

Day 7 — Documentation & Final Release
Tasks Completed:

Comprehensive README documentation
Architecture diagrams and explanations
Usage instructions and examples
Results summary and analysis
Image and screenshot collection
Final project optimization
Commit: Final project upload with documentation

Output: Production-ready project with complete documentation


🚀 Deployment & Scalability

Local Deployment
bash
streamlit run app/app.py --server.port=8501
Production Considerations
Docker containerization for consistency
Cloud platform integration (AWS, Azure, GCP)
Real-time data pipeline setup
Model versioning and management
API development for external integration
Monitoring and logging systems
Scalability Features
Modular code architecture
Batch processing capabilities
Database integration ready
API-ready infrastructure
Load balancing support

🤝 Contributing
We welcome contributions! To contribute:

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

📞 Contact & Support
Author: Aniketsatpathy

Project Repository: GitHub - AI-Predictive-Maintenance-IoT

For Questions or Issues:

Open an issue on GitHub
Check existing documentation
Review project wiki for detailed guides

🙏 Acknowledgments
NASA CMAPSS Dataset: For providing high-quality turbofan engine simulation data
Scikit-learn: Comprehensive machine learning library
Streamlit: Simple and powerful app framework
Open-source Community: For invaluable tools and resources

📚 References & Resources
Key Papers & Resources
NASA PHM Data Challenge
Scikit-learn Documentation
Streamlit Documentation
Random Forest Classifier Guide
Related Topics
Predictive Maintenance Strategies
Time-Series Forecasting
IoT Data Processing
Industrial AI Applications
Condition-Based Maintenance (CBM)
Last Updated: April 15, 2026

This project demonstrates the power of AI in solving real-world industrial challenges through data-driven decision making.
