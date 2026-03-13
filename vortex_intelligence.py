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
        # --- ARRAY HANDLING LOGIC ---
        if isinstance(X, np.ndarray):
            if feature_names is None:
                # Generate default names if none provided for the array
                feature_names = [f"feat_{i}" for i in range(X.shape[1])]
            self.X = pd.DataFrame(X, columns=feature_names).reset_index(drop=True)
        else:
            self.X = X.copy().reset_index(drop=True)
        
        self.y = pd.Series(y).reset_index(drop=True)
        self.task = task.lower() [cite: 2]
        self.report = None
        self.imbalance_threshold = imbalance_threshold [cite: 2]
        self.skew_threshold = skew_threshold [cite: 2]
        self.kurtosis_threshold = kurtosis_threshold [cite: 2]
        self.outlier_iqr_multiplier = outlier_iqr_multiplier [cite: 2]

    def _determine_data_level(self, col):
        unique_c = self.X[col].nunique()
        is_num = np.issubdtype(self.X[col].dtype, np.number)
        if not is_num or unique_c <= 5:
             return "Nominal" [cite: 3]
        if is_num and np.array_equal(self.X[col], self.X[col].astype(int)) and unique_c <= 20:
            return "Ordinal"
        if is_num:
            return "Ratio" if self.X[col].min() >= 0 else "Interval"
        return "Nominal"

    def _generate_text_summary(self):
        if self.report is None: return
        
        numeric_report = self.report[self.report['level'].isin(['Interval', 'Ratio'])] [cite: 4]
        outlier_cols = numeric_report[numeric_report['outlier_count'] > 0].shape[0] [cite: 4]
        skew_count = numeric_report[(numeric_report['skewness'] != "Nominal") & 
                                    ((numeric_report['skewness'] > self.skew_threshold) |
                                     (numeric_report['skewness'] < -self.skew_threshold))].shape[0] [cite: 5]
        high_kurt = numeric_report[(numeric_report['kurtosis'] != "Nominal") & 
                                   (numeric_report['kurtosis'] > self.kurtosis_threshold)].shape[0] [cite: 5]
        
        missing_count = self.report[self.report['null_ratio'] > 0].shape[0] [cite: 5]
        levels = self.report['level'].value_counts().to_dict() [cite: 5]
        strong_signals = self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0] [cite: 5]

        global_min = self.report['min'].min() [cite: 6]
        global_max = self.report['max'].max() [cite: 6]

        def b(text): return f"{self.BOLD}{text}{self.RESET}"

        print(f"\n   {b('--- VORTEX INTELLIGENCE SUMMARY ---')}\n")
        pad = 25

        if self.task == 'classification':
            counts = self.y.value_counts() [cite: 7]
            ratio = counts.max() / counts.min() [cite: 7]
            bal_status = "High Imbalance" if ratio > self.imbalance_threshold else "Balanced / Manageable" [cite: 7]
            bal_color = self.RED if ratio > self.imbalance_threshold else self.GREEN [cite: 7]
            print(f"⚖️ {b('Balance'):<{pad}} : {bal_color}{b(bal_status)} {b(f'(Ratio {ratio:.2f}:1 | Thr: {self.imbalance_threshold}:1)')}") [cite: 8]
            
            top_f = self.report.iloc[0]['feature_name'] [cite: 8]
            auc_val = self.report.iloc[0]['auc_roc'] [cite: 8]
            auc_color = self.GREEN if auc_val > 0.65 else self.YELLOW [cite: 8]
            print(f"🛡️ {b('Predictive'):<{pad}} : {auc_color}{b(f'Top Feature [{top_f}] has AUC-ROC of {auc_val:.4f}')} {b('(Goal: >0.65)')}") [cite: 8]

        print(f"🚩 {b('Outliers'):<{pad}} : {(self.RED if outlier_cols > 0 else self.GREEN)}{b(f'Detected in {outlier_cols} columns' if outlier_cols > 0 else 'It\'s fine. No extreme outliers')} {b(f'(Limit: {self.outlier_iqr_multiplier}xIQR)')}") [cite: 9]
        print(f"📐 {b('Skewness'):<{pad}} : {(self.RED if skew_count > 0 else self.GREEN)}{b(f'{skew_count} columns skewed' if skew_count > 0 else 'It\'s fine. Symmetric')} {b(f'(Thr: ±{self.skew_threshold})')}") [cite: 9]
        print(f"🏔️ {b('Peaks'):<{pad}} : {(self.RED if high_kurt > 0 else self.GREEN)}{b(f'{high_kurt} columns with High Kurtosis' if high_kurt > 0 else 'It\'s fine. Healthy tails')} {b(f'(Thr: <{self.kurtosis_threshold})')}") [cite: 10, 11]
        print(f"✨ {b('Null Values'):<{pad}} : {(self.RED if missing_count > 0 else self.GREEN)}{b('Missing data detected' if missing_count > 0 else 'It\'s fine. Dataset is complete (100% density)')}") [cite: 11]
        
        comp_text = f"{levels.get('Ratio', 0)} Ratio, {levels.get('Interval', 0)} Interval, {levels.get('Ordinal', 0)} Ordinal, {levels.get('Nominal', 0)} Nominal" [cite: 11]
        print(f"📊 {b('Composition'):<{pad}} : {self.CYAN}{b(comp_text)}") [cite: 12]
        print(f"🌐 {b('Data Range'):<{pad}} : {self.CYAN}{b(f'Min: {global_min:.2f} | Max: {global_max:.2f}')}") [cite: 12]
        print(f"🎯 {b('Verdict'):<{pad}} : {self.CYAN}{b(f'{strong_signals} Strong Signals.')}\n") [cite: 12]

    def get_report(self):
        X_tmp = self.X.copy()
        
        # Track categories for external use
        self.cat_features = [] # List of names
        self.cat_idx = []      # List of indices

        for i, col in enumerate(self.X.columns):
            level = self._determine_data_level(col) [cite: 14]
            # If it's not a continuous number (Interval/Ratio), it's a category 
            if level in ["Nominal", "Ordinal"]: [cite: 2, 3]
                self.cat_features.append(col)
                self.cat_idx.append(i)
                # Cast to category for internal LGBM importance calculation [cite: 13]
                X_tmp[col] = X_tmp[col].astype('category') [cite: 13]
        
        # --- Internal Importance Calculation ---
        model = LGBMClassifier(n_estimators=100, importance_type='gain', verbosity=-1) if self.task == 'classification' else \
                LGBMRegressor(n_estimators=100, importance_type='gain', verbosity=-1) [cite: 13]
        model.fit(X_tmp, self.y) [cite: 13]
        gains = dict(zip(self.X.columns, model.feature_importances_)) [cite: 13]

        data_list = []
        for col in self.X.columns:
            level = self._determine_data_level(col) [cite: 14]
            is_num = np.issubdtype(self.X[col].dtype, np.number)
            dtype_name = str(self.X[col].dtype) [cite: 14]
            
            if level in ["Interval", "Ratio"]:
                skew_val = self.X[col].skew() [cite: 14]
                kurt_val = self.X[col].kurtosis() [cite: 14]
                Q1, Q3 = self.X[col].quantile(0.25), self.X[col].quantile(0.75) [cite: 15]
                out_count = ((self.X[col] < (Q1 - self.outlier_iqr_multiplier * (Q3-Q1))) |
                             (self.X[col] > (Q3 + self.outlier_iqr_multiplier * (Q3-Q1)))).sum() [cite: 16]
            else:
                skew_val = "Nominal"
                kurt_val = "Nominal"
                out_count = 0

            auc = 0.5
            if self.task == 'classification': [cite: 17]
                try:
                    score = roc_auc_score(self.y, pd.to_numeric(self.X[col], errors='coerce').fillna(0))
                    auc = max(score, 1 - score)
                except: auc = 0.5
            
            corr = abs(self.X[col].corr(self.y)) if is_num else 0.0 [cite: 18]

            data_list.append({
                'feature_name': col, 
                'level': level,
                'dtype': dtype_name,
                'mean': self.X[col].mean() if is_num else 0, [cite: 19]
                'std': self.X[col].std() if is_num else 0, [cite: 19]
                'min': self.X[col].min() if is_num else 0, [cite: 19]
                'max': self.X[col].max() if is_num else 0, [cite: 19]
                'skewness': skew_val, 
                'kurtosis': kurt_val, [cite: 20]
                'outlier_count': int(out_count), [cite: 20]
                'auc_roc': auc, [cite: 20]
                'abs_target_corr': corr, [cite: 20]
                'null_ratio': self.X[col].isnull().mean(), [cite: 20]
                'unique_counts': int(self.X[col].nunique()), [cite: 20]
                'lgbm_gain': int(gains.get(col, 0)), [cite: 21]
                'vortex_action': "✅ STRONG SIGNAL" if (gains.get(col, 0) > 100 or auc > 0.65) else "⚠️ WEAK SIGNAL" [cite: 21]
            })

        self.report = pd.DataFrame(data_list).sort_values('lgbm_gain', ascending=False).reset_index(drop=True) [cite: 21]
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        self._generate_text_summary()
        return self.report

    def get_visual_report(self, figsize=(18, 5)):
        if self.report is None:
            self.get_report()
        else:
            self._generate_text_summary()

        sns.set_style("whitegrid") [cite: 22]
        
        for feature in self.report['feature_name']: [cite: 23]
            unique_count = self.X[feature].nunique() [cite: 23]
            is_num = np.issubdtype(self.X[feature].dtype, np.number) [cite: 23]
            is_continuous = is_num and unique_count > 10 [cite: 24]
            
            fig, axes = plt.subplots(1, 3, figsize=figsize) [cite: 24]
            fig.suptitle(f"FEATURE ANALYSIS: {feature.upper()}", fontsize=16, fontweight='bold', y=1.05) [cite: 24]

            if is_continuous:
                sns.histplot(self.X[feature], kde=True, ax=axes[0], color='skyblue') [cite: 25]
                axes[0].set_title(f"Distribution (Skew: {self.X[feature].skew():.2f})") [cite: 25]
            else:
                sns.countplot(data=self.X, x=feature, ax=axes[0], palette="viridis") [cite: 25]
                axes[0].set_title(f"Frequency (Categorical/Encoded)") [cite: 25]

            if is_continuous:
                sns.boxplot(x=self.X[feature], ax=axes[1], color='salmon', fliersize=5) [cite: 26]
                axes[1].set_title("Outlier Detection (Boxplot)") [cite: 26]
            else:
                self.X[feature].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=sns.color_palette("pastel"), startangle=90) [cite: 26, 27]
                axes[1].set_ylabel('') [cite: 27]
                axes[1].set_title("Composition (%)") [cite: 27]

            if self.task == 'classification': [cite: 28]
                if not is_continuous:
                    ct = pd.crosstab(self.X[feature], self.y, normalize='index') [cite: 28]
                    sns.heatmap(ct, annot=True, cmap="YlGnBu", ax=axes[2], cbar=False, fmt='.2f') [cite: 28]
                    axes[2].set_title("Target Correlation Heatmap") [cite: 28]
                else:
                    sns.violinplot(x=self.y, y=self.X[feature], ax=axes[2], palette="muted", split=True) [cite: 29]
                    axes[2].set_title("Relationship with Target") [cite: 29]
            else:
                if is_continuous:
                    sns.regplot(x=self.X[feature], y=self.y, ax=axes[2], scatter_kws={'alpha':0.3}, line_kws={'color':'red'}) [cite: 30]
                    axes[2].set_title("Regression Trend") [cite: 30]
                else:
                    sns.boxplot(x=self.X[feature], y=self.y, ax=axes[2], palette="magma") [cite: 31]
                    axes[2].set_title("Target Dist. per Category") [cite: 32]

            plt.tight_layout()
            plt.show()
        return self.report
