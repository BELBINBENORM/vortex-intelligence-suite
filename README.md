# 🌀 Vortex Intelligence Suite
**Automated Feature Engineering & Diagnostic Intelligence for Data Science**

Vortex Intelligence is a lightweight Python engine designed to bridge the gap between raw data and model-ready features. It performs deep statistical analysis, detects data levels (Nominal to Ratio), and provides visual diagnostics in one line of code.

## 🚀 Key Features
* **Smart Data Leveling**: Automatically classifies features into Nominal, Ordinal, Interval, or Ratio.
* **Predictive Scoring**: Uses LightGBM Gain and AUC-ROC to identify "Strong Signals."
* **Bold Diagnostics**: High-visibility terminal summaries with perfectly aligned statistical reports.
* **Triple-Threat Visuals**: Generates 3-row diagnostic grids (Distribution, Outliers/Composition, and Target Relationship) automatically.
---
## 🛠️ Installation

```bash
git clone https://github.com/BELBINBENORM/vortex-intelligence-suite.git
cd vortex-intelligence-suite
pip install -r requirements.txt
```

## 📓 Notebook Installation

If you are using **Jupyter Notebook**, **JupyterLab**, or **Google Colab**, you can install the suite directly from GitHub with a single command:

```python
!pip install git+https://github.com/BELBINBENORM/vortex-intelligence-suite.git
```
---
## 🚀 Quick Start in Notebook

Once installed, you can import and run the suite anywhere in your notebook:

```python
from vortex_intelligence import VortexIntelligence

# Initialize with your X and y
vortex = VortexIntelligence(X, y, task='classification')

# Get the detailed report
report = vortex.get_report()

# Generate the 3-row visual diagnostics as well as a detailed report
report = vortex.get_visual_report()
```
---
## 📊 Intelligence Summary Output

The suite provides a rapid human-readable summary of your dataset's health:

<img width="747" height="211" alt="image" src="https://github.com/user-attachments/assets/470b065b-f27c-45f5-95c7-472a18e54d28" />

<img width="600" height="357" alt="image" src="https://github.com/user-attachments/assets/0c3b24ce-d43e-44e0-beb5-3c788db24b40" />

Will give for all features.

---
*Developed by **BELBIN BENO R M***
