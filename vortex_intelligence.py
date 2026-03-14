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
        
        num_rep = self.report[self.report['level'].isin(['Interval', 'Ratio'])] 
        cat_rep = self.report[self.report['level'].isin(['Nominal', 'Ordinal'])]
        
        redundant_count = self.report[self.report['reason'].str.contains("Redundant", na=False)].shape[0]
        const_count = self.report[self.report['is_constant'] == True].shape[0]
        card_stress = self.report[(self.report['level'].isin(['Nominal', 'Ordinal'])) & 
                                  (self.report['cardinality'] > self.cardinality_threshold)].shape[0]
        leak_count = self.report[self.report['is_leakage'] == True].shape[0]
        out_total = num_rep['outlier_count'].sum() if not num_rep.empty else 0
        skew_count = num_rep[(num_rep['skewness'] != "Categorical") & 
                             (pd.to_numeric(num_rep['skewness']).abs() > self.skew_threshold)].shape[0]
        kurt_count = num_rep[(num_rep['kurtosis'] != "Categorical") & 
                             (pd.to_numeric(num_rep['kurtosis']) > self.kurtosis_threshold)].shape[0]
        strong_signals = self.report[self.report['vortex_action'] == '🚀 STRONG SIGNAL'].shape[0]

        def b(text): return f"{self.BOLD}{text}{self.RESET}"
        pad = 25
        print(f"\n{b('--- VORTEX INTELLIGENCE SUMMARY ---')}\n")
        
        # Scale & Health 
        scale_txt = "Robust Scale" if len(self.X) > 10000 else "Limited Scale" if len(self.X) > 100 else "Micro Scale"
        print(f"📋 {b('Dataset Scale'):<{pad}} : {self.CYAN}{b(f'{scale_txt} ({len(self.X):,} Rows x {self.X.shape[1]} Features)')}")
        comp_txt = "Mixed Feature Types" if num_rep.shape[0] > 0 and cat_rep.shape[0] > 0 else "Uniform Type"
        print(f"📊 {b('Composition'):<{pad}} : {self.CYAN}{b(f'{comp_txt} ({num_rep.shape[0]} Numerical, {cat_rep.shape[0]} Categorical)')}")
        
        null_sum = self.report['null_ratio'].sum()
        null_c, null_txt = (self.GREEN, "Pristine (No missing data detected)") if null_sum == 0 else (self.YELLOW, "Warning (Sparse missing data detected)")
        if null_sum > 0.5: null_c, null_txt = (self.RED, "Critical (High volume of missing data)")
        print(f"✨ {b('Null Values'):<{pad}} : {null_c}{b(null_txt)}")

        # Balance (Only for Classification)
        if self.task == 'classification':
            counts = self.y.value_counts()
            ratio = counts.max() / counts.min()
            bal_c = self.GREEN if ratio <= 1.5 else self.YELLOW if ratio <= self.imbalance_threshold else self.RED
            print(f"⚖️ {b('Balance'):<{pad}} : {bal_c}{b(f'Ratio {ratio:.2f}:1')}")

        # Redundancy
        red_c = self.GREEN if redundant_count == 0 else self.YELLOW if redundant_count <= (len(self.X.columns) // 2) else self.RED
        print(f"🖇️ {b('Redundancy'):<{pad}} : {red_c}{b(f'Clean' if redundant_count == 0 else f'{redundant_count} Twins Detected')}")

        # Stability, Cardinality, Outliers
        print(f"🧊 {b('Stability'):<{pad}} : {self.GREEN if const_count == 0 else self.RED}{b('Healthy' if const_count == 0 else f'{const_count} Dead Columns')}")
        print(f"🗂️ {b('Cardinality'):<{pad}} : {self.GREEN if card_stress == 0 else self.YELLOW}{b('Optimized' if card_stress == 0 else f'{card_stress} High Complexity')}")
        print(f"🚩 {b('Outliers'):<{pad}} : {self.GREEN if out_total == 0 else self.YELLOW}{b('None' if out_total == 0 else f'{int(out_total)} detected')}")
        
        # Distribution
        print(f"📐 {b('Skewness'):<{pad}} : {self.GREEN if skew_count == 0 else self.YELLOW}{b('Symmetric' if skew_count == 0 else f'{skew_count} Distorted')}")
        print(f"🏔️ {b('Kurtosis'):<{pad}} : {self.GREEN if kurt_count == 0 else self.YELLOW}{b('Normal Peaks' if kurt_count == 0 else f'{kurt_count} Sharp Peaks')}")

        # --- DYNAMIC PREDICTIVE METRIC LOGIC ---
        if self.task == 'classification':
            p_metric, m_label, m_threshold = 'auc_roc', 'AUC', 0.65
        else:
            p_metric, m_label, m_threshold = 'spearman_corr', 'Spearman', 0.45

        top_f = self.report.iloc[0]['feature_name'] if not self.report.empty else "N/A"
        top_v = self.report.iloc[0][p_metric] if not self.report.empty else 0.0
        
        pred_c = self.GREEN if top_v > m_threshold else (self.YELLOW if top_v > (m_threshold - 0.1) else self.RED)
        pred_txt = f"Strong Signal [{top_f}] {m_label}: {top_v:.4f}" if top_v > m_threshold else (f"Moderate Signal {m_label}: {top_v:.4f}" if top_v > (m_threshold - 0.1) else f"No Signal Found ({m_label}: {top_v:.4f})")
        
        print(f"🛡️ {b('Predictive'):<{pad}} : {pred_c}{b(pred_txt)}")
        
        # Leakage Alert
        leak_c, leak_txt = (self.GREEN, "Safe") if leak_count == 0 else (self.YELLOW, "Warning (Potential Leakage)")
        if leak_count > 0 and top_v > 0.98: leak_c, leak_txt = (self.RED, "Danger (Direct Leakage)")
        print(f"🌊 {b('Leakage Alert'):<{pad}} : {leak_c}{b(leak_txt)}")
        
        # Final Verdict
        v_color = self.RED if (leak_count > 0 or const_count > 0 or strong_signals == 0) else self.GREEN
        v_msg = f"{strong_signals} Strong Signals. Ready." if v_color == self.GREEN else f"Review {leak_count + const_count} blockers before training."
        if strong_signals == 0: v_msg = "0 Strong Signals. Data quality too low."
        
        print(f"🎯 {b('Final Verdict'):<{pad}} : {v_color}{b(v_msg)}\n")
        
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
            is_const = self.X[col].nunique() <= 1
            
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

            if is_num: 
                desc = self.X[col].describe()
                skew_v, kurt_v = (self.X[col].skew(), self.X[col].kurtosis()) if not is_cat else ("Categorical", "Categorical")
                iqr = desc['75%'] - desc['25%']
                out_c = ((self.X[col] < (desc['25%'] - self.outlier_iqr_multiplier*iqr)) | (self.X[col] > (desc['75%'] + self.outlier_iqr_multiplier*iqr))).sum() if not is_cat else -1 
            else:
                desc = {'mean':0,'std':0,'min':0,'25%':0,'50%':0,'75%':0,'max':0}
                skew_v = kurt_v = "Categorical"; out_c = -1 

            # --- REDUNDANCY LOGIC WITH DATAFRAME STORAGE ---
            # Check if this column is a twin of another column
            twins = upper.index[upper[col] > self.redundancy_threshold].tolist() if col in upper.columns else []
            # Also check if this column is the one being dropped because it's a twin of something else
            is_redundant = any(upper[col] > self.redundancy_threshold) if col in upper.columns else False
            
            is_leakage = power_val > self.leakage_threshold
            reason = "Healthy"
            
            if is_leakage: action, reason = "💀 DANGER (DROP)", "Data Leakage (Too high correlation)" 
            elif is_const: action, reason = "💀 DANGER (DROP)", "Constant Column (No variance)"
            elif twins: 
                action, reason = "💀 DANGER (DROP)", f"Redundant (Twin of {', '.join(twins)})"
            elif (gains.get(col, 0) > 100 or power_val > 0.65): action, reason = "🚀 STRONG SIGNAL", "High Predictive Power" 
            else: action, reason = "⚠️ WEAK/NOISY", "Low Predictive Impact"

            data_list.append({ 
                'feature_name': col, 
                'vortex_action': action, 
                'reason': reason, 
                'level': level, 
                'importance_gain': gains.get(col, 0), 
                metric_name: power_val,
                'redundant_with': ", ".join(twins) if twins else "None",
                'is_leakage': is_leakage, 
                'is_constant': is_const, 
                'cardinality': self.X[col].nunique(), 
                'null_ratio': self.X[col].isnull().mean(),
                'mean': desc['mean'], 'std_dev': desc['std'], 'min': desc['min'], 
                'p25': desc['25%'], 'p50': desc['50%'], 'p75': desc['75%'], 'max': desc['max'], 
                'skewness': skew_v, 'kurtosis': kurt_v, 'outlier_count': int(out_c)
            })
            
        self.report = pd.DataFrame(data_list).sort_values(metric_name, ascending=False).reset_index(drop=True)
        self._generate_text_summary()
        return self.report

    def get_visual_report(self, figsize=(18, 5)): 
        if self.report is None: 
            self.get_report() 
        else: 
            self._generate_text_summary()
        sns.set_style("whitegrid")
        for feature in self.report['feature_name']:
            level = self.report.loc[self.report['feature_name'] == feature, 'level'].values[0] 
            is_categorical = level in ["Nominal", "Ordinal"]
            u_count = self.X[feature].nunique()
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            fig.suptitle(f"FEATURE: {feature.upper()} ({level})", fontsize=16, fontweight='bold', y=1.05)
            if not is_categorical and u_count > 10: 
                sns.histplot(self.X[feature], kde=True, ax=axes[0], color='skyblue')
            else:
                sns.countplot(data=self.X, x=feature, ax=axes[0], palette="viridis")
            if not is_categorical and u_count > 10: 
                sns.boxplot(x=self.X[feature], ax=axes[1], color='salmon')
            else: 
                self.X[feature].value_counts().head(10).plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
            if self.task == 'classification': 
                if is_categorical:
                    ctab = pd.crosstab(self.X[feature], self.y, normalize='index')
                    sns.heatmap(ctab, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[2], cbar=False) 
                else:
                    sns.violinplot(x=self.y, y=self.X[feature], ax=axes[2], palette="muted")
            else: 
                sns.regplot(x=self.X[feature], y=self.y, ax=axes[2], scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
            plt.tight_layout(); plt.show()
        return self.report
