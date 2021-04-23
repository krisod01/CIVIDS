import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import matplotlib.gridspec as gridspec


df=pd.read_csv("./Models/Anova.csv")
print(df.head(5))
print(len(df))

ax = sns.boxplot(x='Mode', y='F1-score', data=df, color='#dae8fc')
ax = sns.swarmplot(x="Mode", y="F1-score", data=df, color='#7d0013')
plt.show()

print()
print("One-way repeated measure ANOVA:")
res = pg.rm_anova(dv='F1-score', within='Mode', subject='triplet_nr', data=df, detailed=True)
print(res)
res.to_csv(r'./Models/anova-res.csv')


print()
print("Post-hoc test:")
post_hocs = pg.pairwise_ttests(dv='F1-score', within='Mode', subject='triplet_nr', padjust='fdr_bh', data=df)
print(post_hocs)
post_hocs.to_csv(r'./Models/post_hoc.csv')

post_hocs = pg.pairwise_ttests(dv='F1-score', within='Mode', subject='triplet_nr', padjust='fdr_bh', tail='one-sided',data=df)
post_hocs.to_csv(r'./Models/post_hoc2.csv')


print()
print("Assumption checks: ")
print(f"Sphericity: {pg.sphericity(data=df, dv='F1-score', subject='triplet_nr', within='Mode')[-1]}")
#print(f"Sphericity: {pg.sphericity(data=df, dv='F1-score', subject='triplet_nr', within='Mode', method='jns')[-1]}")

print()
print("Normality (Shapiro-Wilk):")
print(pg.normality(data=df, dv='F1-score', group='Mode'))

from statsmodels.stats.anova import AnovaRM
res=AnovaRM(data=df, depvar='F1-score', within=['Mode'], subject='triplet_nr')
print(res.fit())


import scipy.stats as stats

fscores_baseline = []
fscores_collaborative = []
fscores_collaborative2 = []

with open("./Models/f1-scores.csv", "r") as file:
    for line in file:
        s = line.split(",")
        if(int(s[0]) == 1):
            fscores_baseline.append(float(s[1]))
        elif(int(s[0]) == 2):
            fscores_collaborative.append(float(s[1]))
        elif(int(s[0]) == 3):
            fscores_collaborative2.append(float(s[1]))

w, pvalue = stats.bartlett(fscores_baseline, fscores_collaborative, fscores_collaborative2)
print(w, pvalue)

stat, p = stats.levene(fscores_baseline, fscores_collaborative, fscores_collaborative2)
print(stat, p)


import numpy as np
  
# First quartile (Q1)
Q1 = np.percentile(fscores_baseline, 25, interpolation = 'midpoint')
  
# Third quartile (Q3)


# Interquartile range (IQR)

print()
IQR = stats.iqr(fscores_baseline, interpolation = 'midpoint')
Q3 = np.percentile(fscores_baseline, 75, interpolation = 'midpoint')
print(f"Q3:{Q3}")
print(f"IQR:{IQR}")
print(f"Q3+1.5*IQR:{IQR*1.5+Q3}")
print(f"Max: {max(fscores_baseline)}")

print()
IQR = stats.iqr(fscores_collaborative, interpolation = 'midpoint')
Q3 = np.percentile(fscores_collaborative, 75, interpolation = 'midpoint')
print(f"Q3:{Q3}")
print(f"IQR:{IQR}")
print(f"Q3+1.5*IQR:{IQR*1.5+Q3}")
print(f"Max: {max(fscores_collaborative)}")

print()
IQR = stats.iqr(fscores_collaborative2, interpolation = 'midpoint')
Q3 = np.percentile(fscores_collaborative2, 75, interpolation = 'midpoint')
print(f"Q3:{Q3}")
print(f"IQR:{IQR}")
print(f"Q3+1.5*IQR:{IQR*1.5+Q3}")
print(f"Max: {max(fscores_collaborative2)}")




fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#fig.suptitle('Q-Q plots for Normal distribution')

pg.qqplot(fscores_baseline, ax=ax1)
ax1.set_title("Baseline", fontweight='bold', fontsize=24)

pg.qqplot(fscores_collaborative, ax=ax2)
ax2.set_title("Collaborative 1", fontweight='bold', fontsize=24)

pg.qqplot(fscores_collaborative2,ax=ax3)
ax3.set_title("Collaborative 2", fontweight='bold', fontsize=24)

plt.show()


# histograms
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
plt.ylabel('Frequency')
plt.xlabel("F1-scores")

ax1.hist(fscores_baseline, bins=20, histtype='bar', ec='k')
ax2.hist(fscores_collaborative, bins=20, histtype='bar', ec='k')
ax3.hist(fscores_collaborative2, bins=20, histtype='bar', ec='k')

plt.show()