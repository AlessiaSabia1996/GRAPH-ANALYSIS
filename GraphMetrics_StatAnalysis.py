#!/usr/bin/env python
# coding: utf-8


# In[1]:


import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting, image
import numpy as np
import networkx as nx
import os
import pandas as pd
import re
import seaborn as sns
from scipy import stats
from scipy.stats import kstest
from scipy.stats import ttest_ind
from scipy.stats import boxcox
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.formula.api import mixedlm
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from sklearn.utils import resample
import nibabel as nib


# In[3]:


def concat_df(patient, control, metric, csv_path):
    
    patient['Group'] = 'Patients'
    control['Group'] = 'Controls'
    scaler =  MinMaxScaler()

    control[metric] = scaler.fit_transform(control[[metric]])
    patient[metric] = scaler.fit_transform(patient[[metric]])
    combined_df = pd.concat([patient, control], ignore_index=True)
    combined_df.to_csv(f"{csv_path}+/+{metric}.csv", index=False )
    return combined_df


# In[4]:


def test_stat(metric, df):
    patients = df[df['Group'] == 'Patients']
    controls = df[df['Group'] == 'Controls']
    
    shapiro_patients = stats.shapiro(patients[metric])
    shapiro_controls = stats.shapiro(controls[metric])

    print("Normality Test (Shapiro-Wilk): H0 data are normally distributed")
    if shapiro_patients.pvalue > 0.05:
        print(f"Patients are normally distributed: W-statistic = {shapiro_patients.statistic}, p-value = {shapiro_patients.pvalue}")
    else:
        print("Pazienti NOT normally distributed")
    if shapiro_controls.pvalue > 0.05:
        print(f"Controls are normally distributed: W-statistic = {shapiro_controls.statistic}, p-value = {shapiro_controls.pvalue}")
    else:
        print("Controlli NOT normally distributed")
        
    levene_stat, levene_p = stats.levene(
        patients[metric],
        controls[metric]
    )

    print("\nOmogeneity of variances test:")
    print(f"Statistic = {levene_stat}, p-value = {levene_p}")

    t_stat, p_value_t = stats.ttest_ind(
        patients[metric],
        controls[metric], alternative="greater"
    )
    print("\nTest T for indipendent samples. H0 average controls > average patients:")
    print(f"t-statistic = {t_stat}, p-value = {p_value_t}")
    t_stat2, p_value_t2 = stats.ttest_ind(
        patients[metric],
        controls[metric]
    )
    print("\nTest T for indipendent samples. H0 no difference in the distributions:")
    print(f"t-statistic = {t_stat2}, p-value = {p_value_t2}")

    u_stat, p_value_mw = stats.mannwhitneyu(
        patients[metric],
        controls[metric]
    )

    print("\nMann-Whitney U test, in case samples are not normally distributed:")
    print(f"U-statistic = {u_stat}, p-value = {p_value_mw}")
    


# In[ ]:


def anova(df):
    model = ols('Density ~ C(Group) + Age + C(Sex)', data=df).fit()
    
    anova_table = sm.stats.anova_lm(model, typ=3, robust='hc3')
    
    p_value = anova_table.loc['C(Group)', 'PR(>F)']
    
    results_df = pd.DataFrame({
        'Region': ['Global'],  
        'p_value': [p_value]
    })
    
    # In case of multiple test apply Benjamini-Hochberg correction 
    #results_df['p_value_corrected'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
    
    significant_results = results_df[results_df['p_value'] < 0.05]
    
    print(significant_results)
    print(anova_table)


# In[ ]:


def anova_local_metrics(df, metric):
    p_values = []  
    nodes = df['Node'].unique()
    
    print(len(nodes))
    results = []
    for node in nodes:
        node_data = df[df['Node'] == node]
    
        model = ols(f'{metric} ~ C(Group) + Age + C(Sex)', data=node_data).fit()
    
        anova_table = sm.stats.anova_lm(model, typ=3, robust='hc3')  # Usa il tipo 1 per il tuo caso
    
        p_value = anova_table.loc['C(Group)', 'PR(>F)']
    
        results.append({
            'Region': node,  
            'p_value': p_value
        })
    
    results_df = pd.DataFrame(results)
    
    results_df['p_value_corrected'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
    
    significant_results = results_df[results_df['p_value_corrected'] < 0.05]
    
    print(significant_results)
    print(anova_table)


# In[ ]:


# GRAPH DENSITY
patients_df_gd = pd.read_csv(csv_path)
controls_df_gd = pd.read_csv(csv_path)

combined_density = concat_df(patients_df_gd, controls_df_gd, "Density")
test_stat("Density", combined_density)


# In[ ]:


stats.probplot(combined_density['Density'], dist="norm", plot=plt)
plt.show()


# In[ ]:


anova(combined_density)


# In[ ]:


plt.figure(figsize=(18, 6)) 

plt.subplot(1, 2, 1)
sns.histplot(controls_df_gd['Density'], kde=True)
plt.title('Histogram of density distribution in Controls')

plt.subplot(1, 2, 2)
sns.histplot(patients_df_gd['Density'], kde=True)
plt.title('Histogram of density distribution in Patients')


# In[ ]:


print(combined_density.groupby('Group').describe())

sns.boxplot(x='Group', y='Density', data=combined_density)
plt.title('Comparison of Graph Density: MS vs Controls')
plt.show()


# In[ ]:


#Transitivity
patients_cc = pd.read_csv(csv_path)
controls_cc = pd.read_csv(csv_path)

combined_cc = concat_df(patients_cc, controls_cc, "Transitivity")
test_stat("Transitivity", combined_cc)


# In[ ]:


stats.probplot(combined_cc['Transitivity'], dist="norm", plot=plt)
plt.show()


# In[ ]:


anova(combined_cc)


# In[ ]:


plt.figure(figsize=(18, 6)) 

plt.subplot(1, 2, 1)
sns.histplot(controls_cc['Transitivity'], kde=True)
plt.title('Histogram of transitivity distribution in Controls')

plt.subplot(1, 2, 2)
sns.histplot(patients_cc['Transitivity'], kde=True)
plt.title('Histogram of transitivity distribution in Patients')


# In[ ]:


print(combined_cc.groupby('Group').describe())

sns.boxplot(x='Group', y='Transitivity', data=combined_cc)
plt.title('Comparison of Transitivity: MS vs Controls'')
plt.show()


# In[ ]:


#AVERAGE SHORTEST PATH LENGHT
patients_spl = pd.read_csv(csv_path)
controls_spl = pd.read_csv(csv_path)

combined_spl = concat_df(patients_spl, controls_spl, "SPL")
test_stat("SPL", combined_spl)


# In[ ]:


stats.probplot(combined_spl['SPL'], dist="norm", plot=plt)
plt.show()


# In[ ]:


anova(combined_spl)


# In[ ]:


print(combined_spl.groupby('Group').describe())

sns.boxplot(x='Group', y='SPL', data=combined_spl)
plt.title('Comparison of SPL: MS vs Controls')
plt.show()


# In[ ]:


plt.figure(figsize=(18, 6))  

plt.subplot(1, 2, 1)
sns.histplot(controls_spl['SPL'], kde=True)
plt.title('Histogram of SPL distribution in Controls')

plt.subplot(1, 2, 2)
sns.histplot(patients_spl['SPL'], kde=True)
plt.title('Histogram of SPL distribution in Patients')


# In[ ]:


patients_df_ge = pd.read_csv(csv_path)
controls_df_ge = pd.read_csv(csv_path)

combined_efficiency = concat_df(patients_df_ge, controls_df_ge, "Efficiency")
test_stat("Efficiency", combined_efficiency)


# In[ ]:


stats.probplot(combined_efficiency['Efficiency'], dist="norm", plot=plt)
plt.show()


# In[ ]:


anova(combined_efficiency)


# In[ ]:


plt.figure(figsize=(18, 6))  
plt.subplot(1, 2, 1)
sns.histplot(controls_df_ge['Efficiency'], kde=True)
plt.title('Histogram of Efficiency distribution in Controls')

plt.subplot(1, 2, 2)
sns.histplot(patients_df_ge['Efficiency'], kde=True)
plt.title('Histogram of Efficiency distribution in Patients')


# In[ ]:


print(combined_efficiency.groupby('Group')['Efficiency'].describe())
sns.boxplot(x='Group', y='Efficiency', data=combined_efficiency)
plt.title('Comparison of Global Efficiency: MS vs Controls')
plt.show()


# In[ ]:


patients_df_MOD = pd.read_csv(csv_path)
controls_df_MOD = pd.read_csv(csv_path)

combined_modularities = concat_df(patients_df_MOD, controls_df_MOD, "Partition")
test_stat("Partition", combined_modularities)


# In[ ]:


stats.probplot(combined_modularities['Partition'], dist="norm", plot=plt)
plt.show()


# In[ ]:


plt.figure(figsize=(18, 6))  

plt.subplot(1, 2, 1)
sns.histplot(controls_df_MOD['Modularity'], kde=True)
plt.title('Histogram of Modularity distribution in Controls')

plt.subplot(1, 2, 2)
sns.histplot(patients_df_MOD['Modularity'], kde=True)
plt.title('Histogram of Modularity distribution in Patients')


# In[ ]:


anova(combined_modularities)


# In[ ]:


print(combined_modularities.groupby('Group').describe())
sns.boxplot(x='Group', y='Modularity', data=combined_modularities)
plt.title('Comparison of Modularity: MS vs Controls')
plt.show()


# In[ ]:


patients_clust = pd.read_csv(csv_path)
controls_clust = pd.read_csv(csv_path)

clust_aver = concat_df(patients_clust, controls_clust, "ClusteringAvg")

test_stat("ClusteringAvg", clust_aver)



# In[ ]:


stats.probplot(clust_aver['ClusteringAvg'], dist="norm", plot=plt)
plt.show()


# In[ ]:


anova(clust_aver)


# In[ ]:


print(clust_aver.groupby('Group').describe())
sns.boxplot(x='Group', y='ClusteringAvg', data=clust_aver)
plt.title('Comparison of Clustering: MS vs Controls')
plt.show()


# In[ ]:


# NODE DEGREE CENTRALITY
patients_df = pd.read_csv(csv_path)
controls_df = pd.read_csv(csv_path)

nodes_degree = concat_df(patients_df, controls_df, "DegreeCentrality")

print(nodes_degree.groupby('Group')['DegreeCentrality'].describe())

ks_stat, ks_p_value = kstest(nodes_degree['DegreeCentrality'], 'norm')
print(f'Test di Kolmogorov-Smirnov: stat = {ks_stat}, p-value = {ks_p_value}')


plt.figure(figsize=(18, 6))  
plt.subplot(1, 2, 1)
sns.histplot(controls_df['DegreeCentrality'], kde=True)
plt.title('Histogram of Node Degree distribution in Controls')

plt.subplot(1, 2, 2)
sns.histplot(patients_df['DegreeCentrality'], kde=True)
plt.title('Histogram of Node Degree distribution in Patients')


# In[ ]:


stats.probplot(nodes_degree['DegreeCentrality'], dist="norm", plot=plt)
plt.show()


# In[ ]:


patients_degree = pd.merge(patients_df, patients_df_gd[['Patient', 'Age', 'Sex']], on='Patient', how='left')
controls_degree = pd.merge(controls_df, controls_df_gd[['Patient', 'Age', 'Sex']], on='Patient', how='left')
degree_covariates = pd.concat([patients_degree, controls_degree], ignore_index=True)
degree_covariates.to_csv(outpath, index=False )

sns.boxplot(x='Group', y='DegreeCentrality', data=degree_covariates)
plt.title('Comparison of Degree Centrality: MS vs Controls')
plt.show()


# In[ ]:


anova_local_metrics(df =degree_covariates, metric='DegreeCentrality')


# In[5]:


patients_df_BET = pd.read_csv(csv_path)
controls_df_BET = pd.read_csv(csv_path)
combined_betweenness = concat_df(patients_df_BET, controls_df_BET, "Betweenness")


# In[ ]:


stats.probplot(combined_betweenness['Betweenness'], dist="norm", plot=plt)
plt.show()


# In[ ]:


print(combined_betweenness.groupby('Group')['Betweenness'].describe())

ks_stat, ks_p_value = kstest(combined_betweenness['Betweenness'], 'norm')
print(f'Test di Kolmogorov-Smirnov: stat = {ks_stat}, p-value = {ks_p_value}')


plt.figure(figsize=(18, 6)) 

plt.subplot(1, 2, 1)
sns.histplot(controls_df_BET['Betweenness'], kde=True)
plt.title('Histogram of Betweenness Centrality distribution in Controls')

plt.subplot(1, 2, 2)
sns.histplot(patients_df_BET['Betweenness'], kde=True)
plt.title('Histogram of Betweenness Centrality distribution in Patients')


# In[ ]:


from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson

#NOT NORMAL DISTRIBUTED: CANNOT USE ANOVA
patients_bet = pd.merge(patients_df_BET, patients_df_gd[['Patient', 'Age', 'Sex']], on='Patient', how='left')
controls_bet = pd.merge(controls_df_BET, controls_df_gd[['Patient', 'Age', 'Sex']], on='Patient', how='left')

bet_covariates = pd.concat([patients_bet, controls_bet], ignore_index=True)
bet_covariates.to_csv(out_csv, index=False )

model = glm("Betweenness ~ Group + Age + Sex", 
            data=bet_covariates, 
            family=Poisson()).fit()

print(model.summary())

sns.boxplot(x='Group', y='Betweenness', data=bet_covariates)
plt.title('Comparison of Betweenness: MS vs Controls')
plt.show()


# In[47]:


patients_df = pd.read_csv(csv_path)
controls_df = pd.read_csv(csv_path)

combined_clustering = concat_df(patients_df, controls_df, "Clustering")



# In[ ]:


stats.probplot(combined_clustering['Clustering'], dist="norm", plot=plt)
plt.show()


# In[ ]:


print(combined_clustering.groupby('Group').describe())

ks_stat, ks_p_value = kstest(combined_clustering['Clustering'], 'norm')
print(f'Test di Kolmogorov-Smirnov: stat = {ks_stat}, p-value = {ks_p_value}')

plt.figure(figsize=(18, 6))  
plt.subplot(1, 2, 1)  
sns.histplot(patients_df['Clustering'], kde=True)
plt.title('Histogram of Clustering distribution in Patients')

plt.subplot(1, 2, 2)  
sns.histplot(controls_df['Clustering'], kde=True)
plt.title('Histogram of Clustering distribution in Controls')


# In[ ]:


patients_clust = pd.merge(patients_df, patients_df_gd[['Patient', 'Age', 'Sex']], on='Patient', how='left')
controls_clust = pd.merge(controls_df, controls_df_gd[['Patient', 'Age', 'Sex']], on='Patient', how='left')
clust_covariates = pd.concat([patients_clust, controls_clust], ignore_index=True)
clust_covariates.to_csv(out_csv, index=False )

anova_local_metrics(df=clust_covariates, metric='Clustering')


# In[52]:


patients_df_VUL = pd.read_csv(csv_path)
controls_df_VUL = pd.read_csv(csv_path)

combined_vulnerability = concat_df(patients_df_VUL, controls_df_VUL, "Vulnerability")


# In[ ]:


stats.probplot(combined_vulnerability['Vulnerability'], dist="norm", plot=plt)
plt.show()


# In[ ]:


print(combined_vulnerability.groupby('Group')['Vulnerability'].describe())

ks_stat, ks_p_value = kstest(combined_vulnerability['Vulnerability'], 'norm')
print(f'Test di Kolmogorov-Smirnov: stat = {ks_stat}, p-value = {ks_p_value}')


plt.figure(figsize=(18, 6))  
plt.subplot(1, 2, 1)
sns.histplot(controls_df_VUL['Vulnerability'], kde=True)
plt.title('Histogram of Vulnerability distribution in Controls')

plt.subplot(1, 2, 2)
sns.histplot(patients_df_VUL['Vulnerability'], kde=True)
plt.title('Histogram of Vulnerability distribution in Patients')


# In[ ]:


patients_vul = pd.merge(patients_df_VUL, patients_df_gd[['Patient', 'Age', 'Sex']], on='Patient', how='left')
controls_vul = pd.merge(controls_df_VUL, controls_df_gd[['Patient', 'Age', 'Sex']], on='Patient', how='left')

vul_covariates = pd.concat([patients_vul, controls_vul], ignore_index=True)
vul_covariates.to_csv(out_csv, index=False )


anova_local_metrics(df=vul_covariates, metric='Vulnerability')

