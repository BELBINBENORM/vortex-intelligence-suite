import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')


# THE CLEANER (Structural & Math Prep)
class ChurnCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, drop_id=True, clip_limit=4.0):
        self.drop_id = drop_id
        self.clip_limit = clip_limit
        self.scaler = RobustScaler()
        
        # Static Mapping Logic
        self.gender_map = {'Female': 1, 'Male': 0}
        self.contract_map = {'Month-To-Month': 0, 'One Year': 1, 'Two Year': 2}
        self.internet_map = {'No': 0, 'Dsl': 1, 'Fiber Optic': 2}
        self.payment_map = {
            'Mailed Check': 0, 'Electronic Check': 1, 
            'Bank Transfer (Automatic)': 2, 'Credit Card (Automatic)': 3
        }
        self.yes_no_cols = [
            'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

    def fit(self, X, y=None):
        X_mapped = self._structural_mapping(X)
        self.numeric_cols_ = X_mapped.select_dtypes(include=[np.number]).columns
        X_stabilized = self._apply_math_corrections(X_mapped)
        self.scaler.fit(X_stabilized[self.numeric_cols_])
        return self

    def _structural_mapping(self, X):
        X = X.copy()
        if self.drop_id and 'id' in X.columns:
            X.drop(columns=['id'], inplace=True)

        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(
                X['TotalCharges'].astype(str).replace(r'^\s*$', '0', regex=True), 
                errors='coerce'
            ).fillna(0)

        if 'gender' in X.columns:
            X['gender'] = X['gender'].astype(str).str.title().map(self.gender_map).fillna(0)
            
        for col in self.yes_no_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).str.contains('yes', case=False).astype(int)

        X['Contract'] = X['Contract'].astype(str).str.title().map(self.contract_map).fillna(0)
        X['InternetService'] = X['InternetService'].astype(str).str.title().map(self.internet_map).fillna(0)
        X['PaymentMethod'] = X['PaymentMethod'].astype(str).str.title().map(self.payment_map).fillna(0)
        return X

    def _apply_math_corrections(self, X):
        for col in ['TotalCharges', 'MonthlyCharges', 'tenure']:
            if col in X.columns:
                X[col] = np.log1p(X[col])
        return X

    def transform(self, X):
        X = self._structural_mapping(X)
        X = self._apply_math_corrections(X)
        X[self.numeric_cols_] = self.scaler.transform(X[self.numeric_cols_])
        X[self.numeric_cols_] = np.clip(X[self.numeric_cols_], -self.clip_limit, self.clip_limit)
        return X.astype(np.float32)


# THE INTELLIGENCE (Reporting & Audit)
class VortexIntelligence:
    def __init__(self, X, y, task='classification', 
                 imbalance_threshold=3.0, 
                 skew_threshold=0.5, 
                 kurtosis_threshold=3.0, 
                 outlier_iqr_multiplier=1.5):
        
        self.X = X.copy().reset_index(drop=True)
        self.y = pd.Series(y).reset_index(drop=True)
        self.task = task.lower()
        self.report = None
        
        self.imbalance_threshold = imbalance_threshold
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier

    def _generate_text_summary(self):
        if self.report is None: return

        num_stats = self.report[self.report['dtype'].str.contains('int|float|complex|bool')]
        outlier_cols = num_stats[num_stats['outlier_count'] > 0].shape[0]
        right_skew = num_stats[num_stats['skewness'] > self.skew_threshold].shape[0]
        left_skew = num_stats[num_stats['skewness'] < -self.skew_threshold].shape[0]
        high_kurt = num_stats[num_stats['kurtosis'] > self.kurtosis_threshold].shape[0]
        null_cols = self.report[self.report['null_ratio'] > 0].shape[0]
        
        num_data = self.X.select_dtypes(include=[np.number])

        print("\n📝 --- VORTEX INTELLIGENCE SUMMARY ---")
        
        if self.task == 'classification':
            counts = self.y.value_counts()
            ratio = counts.max() / counts.min()
            if ratio > self.imbalance_threshold:
                print(f"⚖️ Balance: High Imbalance detected (Ratio {ratio:.2f}:1).")
            else:
                print(f"⚖️ Balance: Classes are well-balanced (Ratio {ratio:.2f}:1).")
        
        if outlier_cols > 0:
            print(f"🚩 Outliers: Detected in {outlier_cols} columns ({self.outlier_iqr_multiplier}xIQR).")
        else:
            print("✨ Outliers: No outliers detected. Distribution is stable.")

        if (right_skew + left_skew) > 0:
            print(f"📐 Skewness: {right_skew} Right, {left_skew} Left detected (±{self.skew_threshold}).")
        else:
            print("✨ Skewness: All features are mathematically symmetrical.")

        if high_kurt > 0:
            print(f"🏔️ Peaks: High Kurtosis detected in {high_kurt} columns (>{self.kurtosis_threshold}).")
        else:
            print("✨ Peaks: No extreme Kurtosis found. Tails are healthy.")

        if null_cols > 0:
            print(f"☁️ Null Values: Detected in {null_cols} columns.")
        else:
            print("✨ Null Values: Dataset is complete (No missing data).")
            
        print(f"📏 Data Range: {num_data.min().min():.2f} to {num_data.max().max():.2f}")
        print(f"🎯 Feature Verdict: {self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0]} Strong Signals.")
        print("-" * 40)

    def _analyze_target(self):
        print(f"\n🎯 --- TARGET ANALYSIS ({self.task.upper()}) ---")
        if self.task == 'classification':
            counts = self.y.value_counts()
            perms = self.y.value_counts(normalize=True) * 100
            print(pd.DataFrame({'Count': counts, 'Percentage': perms.map('{:.2f}%'.format)}))
        else:
            print(f"Mean: {self.y.mean():.4f} | Skew: {self.y.skew():.4f}")

    def get_report(self):
        self._analyze_target()
        print("\n🧠 Scanning features for signal and noise...")

        stats = pd.DataFrame(index=self.X.columns)
        stats['dtype'] = self.X.dtypes.astype(str)
        
        desc = self.X.describe(include='all').T
        for col in ['mean', 'std', 'min', 'max']:
            if col in desc.columns: stats[col] = desc[col]

        num_cols = self.X.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            stats.loc[num_cols, 'skewness'] = self.X[num_cols].skew()
            stats.loc[num_cols, 'kurtosis'] = self.X[num_cols].kurtosis()
            Q1, Q3 = self.X[num_cols].quantile(0.25), self.X[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            stats.loc[num_cols, 'outlier_count'] = ((self.X[num_cols] < (Q1 - self.outlier_iqr_multiplier * IQR)) | 
                                                    (self.X[num_cols] > (Q3 + self.outlier_iqr_multiplier * IQR))).sum()
            stats.loc[num_cols, 'abs_target_corr'] = self.X[num_cols].corrwith(self.y).abs()

        stats['null_ratio'] = self.X.isnull().mean()
        stats['unique_counts'] = self.X.nunique()

        X_tmp = self.X.copy()
        for col in X_tmp.select_dtypes(exclude=[np.number]).columns:
            X_tmp[col] = X_tmp[col].astype('category')

        model = LGBMClassifier(n_estimators=100, importance_type='gain', verbosity=-1) if self.task == 'classification' else LGBMRegressor(n_estimators=100, importance_type='gain', verbosity=-1)
        model.fit(X_tmp, self.y)
        stats['lgbm_gain'] = model.feature_importances_

        def judge(row):
            if row['lgbm_gain'] > 100 or (pd.notnull(row['abs_target_corr']) and row['abs_target_corr'] > 0.05):
                return "✅ STRONG SIGNAL"
            if row['lgbm_gain'] == 0 and (pd.isnull(row['abs_target_corr']) or row['abs_target_corr'] < 0.005):
                return "🗑️ GLOBAL NOISE"
            return "⚠️ WEAK SIGNAL"

        stats['vortex_action'] = stats.apply(judge, axis=1)
        self.report = stats.sort_values(by='lgbm_gain', ascending=False)
        self._generate_text_summary()
        return self.report

    def plot_vortex_eda(self):
        if self.report is None: return print("Run get_report() first.")
        cols = self.X.columns.tolist()
        n_rows = (len(cols) + 2) // 3
        plt.figure(figsize=(18, 5 * n_rows))
        for i, col in enumerate(cols):
            plt.subplot(n_rows, 3, i + 1)
            if self.X[col].nunique() > 10 and np.issubdtype(self.X[col].dtype, np.number):
                sns.histplot(self.X[col], kde=True, color='teal')
            else:
                sns.countplot(x=self.X[col], palette='viridis')
            plt.title(f"{col}\nGain: {self.report.loc[col, 'lgbm_gain']:.1f}")
        plt.tight_layout()
        plt.show()
