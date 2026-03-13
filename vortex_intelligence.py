import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class VortexIntelligence:
    # ANSI Color Constants
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(self, X, y, task='classification', imbalance_threshold=4, skew_threshold=0.75, kurtosis_threshold=10.0, outlier_iqr_multiplier=3.0, feature_names=None):
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"feat_{i}" for i in range(X.shape[1])]
            self.X = pd.DataFrame(X, columns=feature_names).reset_index(drop=True)
        else:
            self.X = X.copy().reset_index(drop=True)
        self.y = pd.Series(y).reset_index(drop=True)
        self.task = task.lower()
        self.report = None
        self.imbalance_threshold = imbalance_threshold
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.cat_features = []
        self.cat_idx = []

    def _determine_data_level(self, col):
        unique_c = self.X[col].nunique()
        is_num = np.issubdtype(self.X[col].dtype, np.number)
        if not is_num or unique_c <= 5: return "Nominal"
        if is_num and np.array_equal(self.X[col], self.X[col].astype(int)) and unique_c <= 20: return "Ordinal"
        return "Ratio" if (is_num and self.X[col].min() >= 0) else "Interval"

    def _generate_text_summary(self):
        if self.report is None: return
        
        numeric_report = self.report[self.report['level'].isin(['Interval', 'Ratio'])]
        categorical_report = self.report[self.report['level'].isin(['Nominal', 'Ordinal'])]
        
        outlier_cols = numeric_report[numeric_report['outlier_count'] > 0].shape[0]
        skew_count = numeric_report[(numeric_report['skewness'] != "Categorical") & 
                                    ((numeric_report['skewness'] > self.skew_threshold) |
                                     (numeric_report['skewness'] < -self.skew_threshold))].shape[0]
        high_kurt = numeric_report[(numeric_report['kurtosis'] != "Categorical") & 
                                   (numeric_report['kurtosis'] > self.kurtosis_threshold)].shape[0]
        
        missing_count = self.report[self.report['null_ratio'] > 0].shape[0]
        levels = self.report['level'].value_counts().to_dict()
        strong_signals = self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0]

        # Safety logic for empty reports
        num_min = numeric_report['min'].min() if not numeric_report.empty else 0.0
        num_max = numeric_report['max'].max() if not numeric_report.empty else 0.0
        cat_min = categorical_report['min'].min() if not categorical_report.empty else 0.0
        cat_max = categorical_report['max'].max() if not categorical_report.empty else 0.0

        def b(text): return f"{self.BOLD}{text}{self.RESET}"
        pad = 25

        print(f"\n{b('--- VORTEX INTELLIGENCE SUMMARY ---')}\n")

        print(f"📋 {b('Dataset Scale'):<{pad}} : {self.CYAN}{b(f'{len(self.X):,} Rows')}")

        if self.task == 'classification':
            counts = self.y.value_counts()
            ratio = counts.max() / counts.min()
            bal_status = "High Imbalance" if ratio > self.imbalance_threshold else "Balanced / Manageable"
            bal_color = self.RED if ratio > self.imbalance_threshold else self.GREEN
            print(f"⚖️ {b('Balance'):<{pad}} : {bal_color}{b(bal_status)} {b(f'(Ratio {ratio:.2f}:1 | Thr: {self.imbalance_threshold}:1)')}")
            
            top_f = self.report.iloc[0]['feature_name']
            auc_val = self.report.iloc[0]['auc_roc']
            auc_color = self.GREEN if auc_val > 0.65 else (self.YELLOW if auc_val > 0.55 else self.RED)
            print(f"🛡️ {b('Predictive'):<{pad}} : {auc_color}{b(f'Top Feature [{top_f}] has AUC-ROC of {auc_val:.4f}')}")

        out_msg = f"Detected in {outlier_cols} columns" if outlier_cols > 0 else "No extreme outliers"
        print(f"🚩 {b('Outliers'):<{pad}} : {(self.RED if outlier_cols > 0 else self.GREEN)}{b(out_msg)} {b(f'(Limit: {self.outlier_iqr_multiplier}xIQR)')}")
        
        skew_msg = f"{skew_count} columns skewed" if skew_count > 0 else "Symmetric"
        print(f"📐 {b('Skewness'):<{pad}} : {(self.RED if skew_count > 0 else self.GREEN)}{b(skew_msg)} {b(f'(Thr: ±{self.skew_threshold})')}")
        
        kurt_msg = f"{high_kurt} columns with High Kurtosis" if high_kurt > 0 else "Healthy tails"
        print(f"🏔️ {b('Peaks'):<{pad}} : {(self.RED if high_kurt > 0 else self.GREEN)}{b(kurt_msg)} {b(f'(Thr: <{self.kurtosis_threshold})')}")
        
        null_msg = "Missing data detected" if missing_count > 0 else "Dataset is complete (100% density)"
        print(f"✨ {b('Null Values'):<{pad}} : {(self.RED if missing_count > 0 else self.GREEN)}{b(null_msg)}")
        
        num_count = levels.get('Ratio', 0) + levels.get('Interval', 0)
        cat_count = levels.get('Nominal', 0) + levels.get('Ordinal', 0)
        print(f"📊 {b('Composition'):<{pad}} : {self.CYAN}{b(f'{num_count} Numerical, {cat_count} Categorical')}")
        
        cat_range_color = self.CYAN if cat_min >= 0 else self.RED
        print(f"📉 {b('Categorical Range'):<{pad}} : {cat_range_color}{b(f'Min: {cat_min:.2f} | Max: {cat_max:.2f}')}")
        print(f"📈 {b('Numerical Range'):<{pad}} : {self.CYAN}{b(f'Min: {num_min:.2f} | Max: {num_max:.2f}')}")
        
        verdict_color = self.GREEN if strong_signals > (len(self.X.columns)/2) else self.YELLOW
        print(f"🎯 {b('Verdict'):<{pad}} : {verdict_color}{b(f'{strong_signals} Strong Signals.')}\n")

    def get_report(self):
        X_tmp = self.X.copy()
        self.cat_features, self.cat_idx = [], []
        for i, col in enumerate(self.X.columns):
            level = self._determine_data_level(col)
            if level in ["Nominal", "Ordinal"]:
                self.cat_features.append(col); self.cat_idx.append(i)
                X_tmp[col] = X_tmp[col].astype('category')
        model = LGBMClassifier(n_estimators=100, importance_type='gain', verbosity=-1) if self.task == 'classification' else LGBMRegressor(n_estimators=100, importance_type='gain', verbosity=-1)
        model.fit(X_tmp, self.y)
        gains = dict(zip(self.X.columns, model.feature_importances_))
        data_list = []
        for col in self.X.columns:
            level = self._determine_data_level(col); is_num = np.issubdtype(self.X[col].dtype, np.number)
            if level in ["Interval", "Ratio"]:
                skew_val, kurt_val = self.X[col].skew(), self.X[col].kurtosis()
                Q1, Q3 = self.X[col].quantile(0.25), self.X[col].quantile(0.75)
                out_count = ((self.X[col] < (Q1 - self.outlier_iqr_multiplier * (Q3-Q1))) | (self.X[col] > (Q3 + self.outlier_iqr_multiplier * (Q3-Q1)))).sum()
            else: skew_val = kurt_val = "Categorical"; out_count = -1
            auc = 0.5
            if self.task == 'classification':
                try: score = roc_auc_score(self.y, pd.to_numeric(self.X[col], errors='coerce').fillna(0)); auc = max(score, 1 - score)
                except: auc = 0.5
            data_list.append({'feature_name': col, 'level': level, 'min': self.X[col].min() if is_num else 0, 'max': self.X[col].max() if is_num else 0, 'skewness': skew_val, 'kurtosis': kurt_val, 'outlier_count': int(out_count), 'auc_roc': auc, 'null_ratio': self.X[col].isnull().mean(), 'vortex_action': "✅ STRONG SIGNAL" if (gains.get(col, 0) > 100 or auc > 0.65) else "⚠️ WEAK SIGNAL"})
        self.report = pd.DataFrame(data_list).sort_values('auc_roc', ascending=False).reset_index(drop=True)
        self._generate_text_summary()
        return self.report

    def get_visual_report(self, figsize=(18, 5)):
        if self.report is None: self.get_report()
        sns.set_style("whitegrid")
        for feature in self.report['feature_name']:
            u_count = self.X[feature].nunique()
            is_num = np.issubdtype(self.X[feature].dtype, np.number)
            is_cont = is_num and u_count > 10
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            fig.suptitle(f"FEATURE ANALYSIS: {feature.upper()}", fontsize=16, fontweight='bold', y=1.05)
            if is_cont:
                sns.histplot(self.X[feature], kde=True, ax=axes[0], color='skyblue')
                axes[0].set_title(f"Distribution (Skew: {self.X[feature].skew():.2f})")
                sns.boxplot(x=self.X[feature], ax=axes[1], color='salmon', fliersize=5)
                axes[1].set_title("Outlier Detection (Boxplot)")
            else:
                sns.countplot(data=self.X, x=feature, ax=axes[0], palette="viridis")
                axes[0].set_title("Frequency")
                self.X[feature].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=sns.color_palette("pastel"), startangle=90)
                axes[1].set_ylabel('')
                axes[1].set_title("Composition (%)")
            if self.task == 'classification':
                if not is_cont:
                    ct = pd.crosstab(self.X[feature], self.y, normalize='index')
                    sns.heatmap(ct, annot=True, cmap="YlGnBu", ax=axes[2], cbar=False, fmt='.2f')
                else:
                    sns.violinplot(x=self.y, y=self.X[feature], ax=axes[2], palette="muted", split=True)
            plt.tight_layout(); plt.show()
        return self.report
