# 🌪️ Vortex Intelligence Suite

**Vortex Intelligence Suite** is a high-speed, automated feature profiling and diagnostic tool designed for machine learning practitioners. [cite_start]It provides a comprehensive "intelligence report" by combining deep statistical metrics with Gradient Boosting feature importance to evaluate feature signals and noise[cite: 1, 12, 15].
---
## 🚀 Features

* [cite_start]**Target Diagnostics**: Automatically analyzes class imbalance for classification tasks and distribution properties like skew and kurtosis for regression[cite: 5, 6, 9, 11].
* [cite_start]**Statistical Profiling**: Calculates skewness, kurtosis, and outlier counts using the Interquartile Range (IQR) method[cite: 3, 7, 13, 14].
* [cite_start]**Signal Detection**: Categorizes features into **Strong Signal**, **Weak Signal**, or **Global Noise** based on LightGBM "Gain" importance and target correlation[cite: 8, 16, 17].
* [cite_start]**Hybrid Data Support**: Handles both numerical and categorical columns within a single report, including data type identification[cite: 3, 11, 12, 15].
* [cite_start]**Automated EDA**: Generates a grid-based visualization suite (3 plots per row) that automatically adapts between histograms and count plots based on data cardinality[cite: 18, 19, 20].
---
## 🛠️ Installation

```bash
git clone [https://github.com/BELBINBENORM/vortex-intelligence-suite.git](https://github.com/BELBINBENORM/vortex-intelligence-suite.git)
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

# Initialize the suite (task can be 'classification' or 'regression')
vortex = VortexIntelligence(X_train, y_train, task='classification')

# Generate the intelligence report and textual summary
report = vortex.get_report()

# Visualize the features and target distribution
vortex.plot_vortex_eda()
```
---
## 📊 Intelligence Summary Output

The suite provides a rapid human-readable summary of your dataset's health:

* [cite_start]**⚖️ Balance**: Class ratio and imbalance warnings for classification or skew/kurtosis for regression. [cite: 5, 6]
* [cite_start]**📂 Structure**: Counts of numerical and categorical columns found in the dataset. [cite: 3, 6]
* [cite_start]**🚩 Outliers**: Count of columns containing statistical outliers found in numerical columns. [cite: 3, 6]
* [cite_start]**📐 Shape**: Quantification of right/left skewness and heavy-tailed (Kurtosis) distributions. [cite: 3, 7]
* [cite_start]**✨ Null Values**: Detection and reporting of missing values across the dataset. [cite: 4, 7, 8]
* [cite_start]**📏 Data Range**: The minimum and maximum range of all numerical data. [cite: 4, 8]
* [cite_start]**🎯 Verdict**: Count of high-value features identified as "Strong Signals". [cite: 8, 16]

---
*Developed by **BELBIN BENO R M***
