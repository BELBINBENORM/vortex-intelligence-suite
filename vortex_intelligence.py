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

    def __init__(self, X, y, task='classification',imbalance_threshold=4,skew_threshold=0.75,kurtosis_threshold=10.0,outlier_iqr_multiplier=3.0):
        self.X = X.copy().reset_index(drop=True)
        self.y = pd.Series(y).reset_index(drop=True)
        self.task = task.lower()
        self.report = None
        self.imbalance_threshold = imbalance_threshold
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier

    def _determine_data_level(self, col):
        unique_c = self.X[col].nunique()
        is_num = np.issubdtype(self.X[col].dtype, np.number)
        if not is_num or unique_c <= 5:
            return "Nominal"
        if is_num and np.array_equal(self.X[col], self.X[col].astype(int)) and unique_c <= 20:
            return "Ordinal"
        if is_num:
            return "Ratio" if self.X[col].min() >= 0 else "Interval"
        return "Nominal"

    def _generate_text_summary(self):
        if self.report is None: return
        
        # Filter for numeric metrics
        numeric_report = self.report[self.report['level'].isin(['Interval', 'Ratio'])]
        
        outlier_cols = numeric_report[numeric_report['outlier_count'] > 0].shape[0]
        skew_count = numeric_report[(numeric_report['skewness'] != "Nominal") & 
                                    ((numeric_report['skewness'] > self.skew_threshold) | 
                                     (numeric_report['skewness'] < -self.skew_threshold))].shape[0]
        high_kurt = numeric_report[(numeric_report['kurtosis'] != "Nominal") & 
                                   (numeric_report['kurtosis'] > self.kurtosis_threshold)].shape[0]
        
        missing_count = self.report[self.report['null_ratio'] > 0].shape[0]
        levels = self.report['level'].value_counts().to_dict()
        strong_signals = self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0]

        # Calculate Global Range for Verification
        global_min = self.report['min'].min()
        global_max = self.report['max'].max()

        def b(text): return f"{self.BOLD}{text}{self.RESET}"

        print(f"\n   {b('--- VORTEX INTELLIGENCE SUMMARY ---')}\n")
        pad = 25

        # Balance
        if self.task == 'classification':
            counts = self.y.value_counts()
            ratio = counts.max() / counts.min()
            bal_status = "High Imbalance" if ratio > self.imbalance_threshold else "Balanced / Manageable"
            bal_color = self.RED if ratio > self.imbalance_threshold else self.GREEN
            print(f"⚖️ {b('Balance'):<{pad}} : {bal_color}{b(bal_status)} {b(f'(Ratio {ratio:.2f}:1 | Thr: {self.imbalance_threshold}:1)')}")
            
            # Predictive
            top_f = self.report.iloc[0]['feature_name']
            auc_val = self.report.iloc[0]['auc_roc']
            auc_color = self.GREEN if auc_val > 0.65 else self.YELLOW
            print(f"🛡️ {b('Predictive'):<{pad}} : {auc_color}{b(f'Top Feature [{top_f}] has AUC-ROC of {auc_val:.4f}')} {b('(Goal: >0.65)')}")

        # Outliers
        print(f"🚩 {b('Outliers'):<{pad}} : {(self.RED if outlier_cols > 0 else self.GREEN)}{b(f'Detected in {outlier_cols} columns' if outlier_cols > 0 else 'It\'s fine. No extreme outliers')} {b(f'(Limit: {self.outlier_iqr_multiplier}xIQR)')}")
        
        # Skewness
        print(f"📐 {b('Skewness'):<{pad}} : {(self.RED if skew_count > 0 else self.GREEN)}{b(f'{skew_count} columns skewed' if skew_count > 0 else 'It\'s fine. Symmetric')} {b(f'(Thr: ±{self.skew_threshold})')}")
        
        # Peaks
        print(f"🏔️ {b('Peaks'):<{pad}} : {(self.RED if high_kurt > 0 else self.GREEN)}{b(f'{high_kurt} columns with High Kurtosis' if high_kurt > 0 else 'It\'s fine. Healthy tails')} {b(f'(Thr: <{self.kurtosis_threshold})')}")
        
        # Nulls
        print(f"✨ {b('Null Values'):<{pad}} : {(self.RED if missing_count > 0 else self.GREEN)}{b('Missing data detected' if missing_count > 0 else 'It\'s fine. Dataset is complete (100% density)')}")
        
        # Composition
        comp_text = f"{levels.get('Ratio', 0)} Ratio, {levels.get('Interval', 0)} Interval, {levels.get('Ordinal', 0)} Ordinal, {levels.get('Nominal', 0)} Nominal"
        print(f"📊 {b('Composition'):<{pad}} : {self.CYAN}{b(comp_text)}")
        
        # Range (NEW: Useful for verifying scaling/clipping)
        print(f"🌐 {b('Data Range'):<{pad}} : {self.CYAN}{b(f'Min: {global_min:.2f} | Max: {global_max:.2f}')}")

        # Verdict
        print(f"🎯 {b('Verdict'):<{pad}} : {self.CYAN}{b(f'{strong_signals} Strong Signals.')}\n")
    def get_report(self):
        X_tmp = self.X.copy()
        for c in X_tmp.select_dtypes(exclude=[np.number]).columns: 
            X_tmp[c] = X_tmp[c].astype('category')
        
        model = LGBMClassifier(n_estimators=100, importance_type='gain', verbosity=-1) if self.task == 'classification' else \
                LGBMRegressor(n_estimators=100, importance_type='gain', verbosity=-1)
        model.fit(X_tmp, self.y)
        gains = dict(zip(self.X.columns, model.feature_importances_))

        data_list = []
        for col in self.X.columns:
            level = self._determine_data_level(col)
            is_num = np.issubdtype(self.X[col].dtype, np.number)
            dtype_name = str(self.X[col].dtype)
            
            if level in ["Interval", "Ratio"]:
                skew = self.X[col].skew()
                kurt = self.X[col].kurtosis()
                Q1, Q3 = self.X[col].quantile(0.25), self.X[col].quantile(0.75)
                out_count = ((self.X[col] < (Q1 - self.outlier_iqr_multiplier * (Q3-Q1))) | 
                             (self.X[col] > (Q3 + self.outlier_iqr_multiplier * (Q3-Q1)))).sum()
            else:
                skew = "Nominal"
                kurt = "Nominal"
                out_count = 0

            auc = 0.5
            if self.task == 'classification':
                try:
                    score = roc_auc_score(self.y, pd.to_numeric(self.X[col], errors='coerce').fillna(0))
                    auc = max(score, 1 - score)
                except: auc = 0.5
            
            corr = abs(self.X[col].corr(self.y)) if is_num else 0.0

            data_list.append({
                'feature_name': col, 
                'level': level,
                'dtype': dtype_name,  # NEW COLUMN
                'mean': self.X[col].mean() if is_num else 0,
                'std': self.X[col].std() if is_num else 0,
                'min': self.X[col].min() if is_num else 0,
                'max': self.X[col].max() if is_num else 0,
                'skewness': skew, 
                'kurtosis': kurt,
                'outlier_count': int(out_count), 
                'auc_roc': auc,
                'abs_target_corr': corr, 
                'null_ratio': self.X[col].isnull().mean(),
                'unique_counts': int(self.X[col].nunique()), 
                'lgbm_gain': int(gains.get(col, 0)),
                'vortex_action': "✅ STRONG SIGNAL" if (gains.get(col, 0) > 100 or auc > 0.65) else "⚠️ WEAK SIGNAL"
            })

        self.report = pd.DataFrame(data_list).sort_values('lgbm_gain', ascending=False).reset_index(drop=True)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        self._generate_text_summary()
        return self.report

        
    def get_visual_report(self, figsize=(18, 5)):
        """Generates a 3-column diagnostic row for every feature with smart detection for encoded categories."""
        if self.report is None:
            self.get_report()
        else:
            self._generate_text_summary()

        sns.set_style("whitegrid")
        
        for feature in self.report['feature_name']:
            # Check if numeric AND has enough variety to be truly continuous
            unique_count = self.X[feature].nunique()
            is_num = np.issubdtype(self.X[feature].dtype, np.number)
            
            # CRITICAL FIX: Categorical/Encoded if unique values <= 10
            is_continuous = is_num and unique_count > 10
            
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            fig.suptitle(f"FEATURE ANALYSIS: {feature.upper()}", fontsize=16, fontweight='bold', y=1.05)

            # --- 1. DISTRIBUTION CHART ---
            if is_continuous:
                sns.histplot(self.X[feature], kde=True, ax=axes[0], color='skyblue')
                axes[0].set_title(f"Distribution (Skew: {self.X[feature].skew():.2f})")
            else:
                sns.countplot(data=self.X, x=feature, ax=axes[0], palette="viridis")
                axes[0].set_title(f"Frequency (Categorical/Encoded)")

            # --- 2. COMPOSITION / OUTLIER CHART ---
            if is_continuous:
                sns.boxplot(x=self.X[feature], ax=axes[1], color='salmon', fliersize=5)
                axes[1].set_title("Outlier Detection (Boxplot)")
            else:
                self.X[feature].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                                   colors=sns.color_palette("pastel"), startangle=90)
                axes[1].set_ylabel('')
                axes[1].set_title("Composition (%)")

            # --- 3. TARGET RELATIONSHIP ---
            if self.task == 'classification':
                # FIX: If it's encoded or categorical (NOT continuous), show Heatmap
                if not is_continuous:
                    ct = pd.crosstab(self.X[feature], self.y, normalize='index')
                    sns.heatmap(ct, annot=True, cmap="YlGnBu", ax=axes[2], cbar=False, fmt='.2f')
                    axes[2].set_title("Target Correlation Heatmap")
                else:
                    # Only show Violin for truly continuous numbers (like Tenure)
                    sns.violinplot(x=self.y, y=self.X[feature], ax=axes[2], palette="muted", split=True)
                    axes[2].set_title("Relationship with Target")
            else:
                # Regression Trend
                if is_continuous:
                    sns.regplot(x=self.X[feature], y=self.y, ax=axes[2], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
                    axes[2].set_title("Regression Trend")
                else:
                    sns.boxplot(x=self.X[feature], y=self.y, ax=axes[2], palette="magma")
                    axes[2].set_title("Target Dist. per Category")

            plt.tight_layout()
            plt.show()
        return self.report
