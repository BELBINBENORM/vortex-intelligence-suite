import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class VortexIntelligence:
    # UI Constants
    GREEN, RED, YELLOW, CYAN = "\033[92m", "\033[91m", "\033[93m", "\033[96m"
    BOLD, RESET = "\033[1m", "\033[0m"

    def __init__(self, X, y, task='classification', 
                 imbalance_threshold=4, skew_threshold=0.75, 
                 kurtosis_threshold=10.0, outlier_iqr_multiplier=3.0,
                 redundancy_threshold=0.90, cardinality_threshold=100,
                 leakage_threshold=0.95, feature_names=None):
        
        # Hard-cast to DataFrame for consistency
        if isinstance(X, np.ndarray):
            f_names = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
            self.X = pd.DataFrame(X, columns=f_names).reset_index(drop=True)
        else:
            self.X = X.copy().reset_index(drop=True)
            
        self.y = pd.Series(y).reset_index(drop=True)
        self.task = task.lower()
        self.report = None
        
        # Threshold Parameters
        self.imbalance_threshold = imbalance_threshold
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.redundancy_threshold = redundancy_threshold
        self.cardinality_threshold = cardinality_threshold
        self.leakage_threshold = leakage_threshold
        self.cat_features = []

    def _determine_data_level(self, col):
        unique_c = self.X[col].nunique()
        is_num = np.issubdtype(self.X[col].dtype, np.number)
        if not is_num or unique_c <= 5: return "Nominal"
        if is_num and np.array_equal(self.X[col], self.X[col].astype(int)) and unique_c <= 20: return "Ordinal"
        return "Ratio" if (is_num and self.X[col].min() >= 0) else "Interval"

    def _generate_text_summary(self):
        if self.report is None: return
        
        # Segregate for summary stats
        num_rep = self.report[self.report['level'].isin(['Interval', 'Ratio'])]
        cat_rep = self.report[self.report['level'].isin(['Nominal', 'Ordinal'])]
        
        # 1. Redundancy (Pearson)
        corr_matrix = self.X.select_dtypes(include=[np.number]).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        redundant_pairs = [column for column in upper.columns if any(upper[column] > self.redundancy_threshold)]
        
        # 2. Stability & Cardinality
        low_variance = [col for col in self.X.columns if self.X[col].nunique() <= 1]
        high_card = [col for col in self.cat_features if (self.X[col].nunique() > self.cardinality_threshold)]
        
        # 3. Leakage & Power
        p_metric = 'auc_roc' if self.task == 'classification' else 'spearman_corr'
        leakage_features = self.report[self.report[p_metric] > self.leakage_threshold]['feature_name'].tolist()

        # 4. Math Counts
        outlier_cols = num_rep[num_rep['outlier_count'] > 0].shape[0]
        # Skip string labels for skew/kurt check
        skew_count = num_rep[(num_rep['skewness'] != "Categorical") & (pd.to_numeric(num_rep['skewness']).abs() > self.skew_threshold)].shape[0]
        high_kurt = num_rep[(num_rep['kurtosis'] != "Categorical") & (pd.to_numeric(num_rep['kurtosis']) > self.kurtosis_threshold)].shape[0]
        strong_signals = self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0]

        def b(text): return f"{self.BOLD}{text}{self.RESET}"
        pad = 25
        print(f"\n{b('--- VORTEX INTELLIGENCE SUMMARY ---')}\n")

        # Items 1-3: Scale & Health
        print(f"📋 {b('1. Dataset Scale'):<{pad}} : {self.CYAN}{b(f'{len(self.X):,} Rows x {self.X.shape[1]} Features')}")
        print(f"📊 {b('2. Composition'):<{pad}} : {self.CYAN}{b(f'{num_rep.shape[0]} Numerical, {cat_rep.shape[0]} Categorical')}")
        print(f"✨ {b('3. Null Values'):<{pad}} : {(self.RED if self.report['null_ratio'].sum() > 0 else self.GREEN)}{b('Complete' if self.report['null_ratio'].sum() == 0 else 'Missing detected')}")
        
        # Item 4: Balance (Classification only)
        if self.task == 'classification':
            ratio = self.y.value_counts().max() / self.y.value_counts().min()
            print(f"⚖️ {b('4. Balance'):<{pad}} : {(self.RED if ratio > self.imbalance_threshold else self.GREEN)}{b(f'Ratio {ratio:.2f}:1')}")

        # Items 5-7: Quality
        print(f"🖇️ {b('5. Redundancy'):<{pad}} : {(self.YELLOW if redundant_pairs else self.GREEN)}{b(f'{len(redundant_pairs)} duplicates' if redundant_pairs else 'Clean')} (>{self.redundancy_threshold})")
        print(f"🧊 {b('6. Stability'):<{pad}} : {(self.RED if low_variance else self.GREEN)}{b(f'{len(low_variance)} constant cols' if low_variance else 'Healthy variance')}")
        print(f"🗂️ {b('7. Cardinality'):<{pad}} : {(self.YELLOW if high_card else self.GREEN)}{b(f'High complexity in {len(high_card)} cols' if high_card else 'Well-grouped')} (>{self.cardinality_threshold})")

        # Items 8-10: Distributions
        print(f"🚩 {b('8. Outliers'):<{pad}} : {(self.RED if outlier_cols > 0 else self.GREEN)}{b(outlier_cols if outlier_cols > 0 else 'None')} ({self.outlier_iqr_multiplier}xIQR)")
        print(f"📐 {b('9. Skewness'):<{pad}} : {(self.RED if skew_count > 0 else self.GREEN)}{b(skew_count)} (Thr: ±{self.skew_threshold})")
        print(f"🏔️ {b('10. Peaks (Kurtosis)'):<{pad}} : {(self.RED if high_kurt > 0 else self.GREEN)}{b(high_kurt)} (Thr: <{self.kurtosis_threshold})")

        # Items 11-12: Hardened Ranges (Explicit cast to float)
        c_min = cat_rep['min'].astype(float).min() if not cat_rep.empty else 0.0
        c_max = cat_rep['max'].astype(float).max() if not cat_rep.empty else 0.0
        n_min = num_rep['min'].astype(float).min() if not num_rep.empty else 0.0
        n_max = num_rep['max'].astype(float).max() if not num_rep.empty else 0.0

        print(f"📉 {b('11. Categorical Range'):<{pad}} : {(self.CYAN if c_min >= 0 else self.RED)}{b(f'Min: {c_min:.2f} | Max: {c_max:.2f}')}")
        print(f"📈 {b('12. Numerical Range'):<{pad}} : {self.CYAN}{b(f'Min: {n_min:.2f} | Max: {n_max:.2f}')}")

        # Items 13-15: Power & Verdict
        top_f, top_v = self.report.iloc[0]['feature_name'], self.report.iloc[0][p_metric]
        print(f"🛡️ {b('13. Predictive'):<{pad}} : {self.GREEN}{b(f'[{top_f}] {p_metric.upper()}: {top_v:.4f}')}")
        print(f"🌊 {b('14. Leakage Alert'):<{pad}} : {(self.RED if leakage_features else self.GREEN)}{b(f'SUSPICIOUS: {leakage_features}' if leakage_features else 'None')} (>{self.leakage_threshold})")
        
        v_color = self.RED if (leakage_features or low_variance or c_min < 0) else self.GREEN
        print(f"🎯 {b('15. Final Verdict'):<{pad}} : {v_color}{b(f'{strong_signals} Strong Signals. Check report before training.')}\n")

    def get_report(self):
        X_tmp = self.X.copy()
        self.cat_features = []
        corr_matrix = self.X.select_dtypes(include=[np.number]).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        for col in self.X.columns:
            if self._determine_data_level(col) in ["Nominal", "Ordinal"]:
                self.cat_features.append(col)
                X_tmp[col] = X_tmp[col].astype('category')
        
        model = LGBMClassifier(n_estimators=100, verbosity=-1) if self.task == 'classification' else LGBMRegressor(n_estimators=100, verbosity=-1)
        model.fit(X_tmp, self.y)
        gains = dict(zip(self.X.columns, model.feature_importances_))
        
        data_list = []
        for col in self.X.columns:
            level = self._determine_data_level(col)
            is_num = np.issubdtype(self.X[col].dtype, np.number)
            is_cat = level in ["Nominal", "Ordinal"]
            
            # Task-Specific Metric
            if self.task == 'classification':
                try: 
                    score = roc_auc_score(self.y, pd.to_numeric(self.X[col], errors='coerce').fillna(0))
                    power_val = max(score, 1 - score)
                except: power_val = 0.5
                metric_name = 'auc_roc'
            else:
                power_val = self.X[col].corr(self.y, method='spearman')
                power_val = abs(power_val) if not np.isnan(power_val) else 0.0
                metric_name = 'spearman_corr'

            # Descriptive Stats Profile
            if is_num:
                desc = self.X[col].describe()
                skew_v, kurt_v = (self.X[col].skew(), self.X[col].kurtosis()) if not is_cat else ("Categorical", "Categorical")
                iqr = desc['75%'] - desc['25%']
                out_c = ((self.X[col] < (desc['25%'] - self.outlier_iqr_multiplier*iqr)) | 
                         (self.X[col] > (desc['75%'] + self.outlier_iqr_multiplier*iqr))).sum() if not is_cat else -1
            else:
                desc = {'mean':0,'std':0,'min':0,'25%':0,'50%':0,'75%':0,'max':0}
                skew_v = kurt_v = "Categorical"; out_c = -1

            twins = upper.index[upper[col] > self.redundancy_threshold].tolist()

            data_list.append({
                'feature_name': col, 'level': level, 'importance_gain': gains.get(col, 0),
                metric_name: power_val, 'vortex_action': "✅ STRONG SIGNAL" if (gains.get(col, 0) > 100 or power_val > 0.65) else "⚠️ WEAK SIGNAL",
                'duplicate_of': ", ".join(twins) if twins else "None", 'is_leakage': power_val > self.leakage_threshold,
                'is_constant': self.X[col].nunique() <= 1, 'cardinality': self.X[col].nunique(), 'null_ratio': self.X[col].isnull().mean(),
                'mean': desc['mean'], 'std_dev': desc['std'], 'min': desc['min'], 'p25': desc['25%'], 'p50': desc['50%'], 'p75': desc['75%'], 'max': desc['max'],
                'skewness': skew_v, 'kurtosis': kurt_v, 'outlier_count': int(out_c)
            })
            
        self.report = pd.DataFrame(data_list).sort_values(metric_name, ascending=False).reset_index(drop=True)
        self._generate_text_summary()
        return self.report

    def get_visual_report(self, figsize=(18, 5)):
        if self.report is None: self.get_report()
        sns.set_style("whitegrid")
        for feature in self.report['feature_name']:
            u_count = self.X[feature].nunique()
            is_num = np.issubdtype(self.X[feature].dtype, np.number)
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            fig.suptitle(f"FEATURE: {feature.upper()}", fontsize=16, fontweight='bold', y=1.05)
            
            # 1. Dist
            if is_num and u_count > 10: sns.histplot(self.X[feature], kde=True, ax=axes[0], color='skyblue')
            else: sns.countplot(data=self.X, x=feature, ax=axes[0], palette="viridis")
            
            # 2. Composition/Spread
            if is_num and u_count > 10: sns.boxplot(x=self.X[feature], ax=axes[1], color='salmon')
            else: self.X[feature].value_counts().head(10).plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
            
            # 3. Target Relation
            if self.task == 'classification': sns.violinplot(x=self.y, y=self.X[feature], ax=axes[2], split=True)
            else: sns.regplot(x=self.X[feature], y=self.y, ax=axes[2], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
            
            plt.tight_layout(); plt.show()
