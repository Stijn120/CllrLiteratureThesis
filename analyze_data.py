import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import scipy.stats as st
from matplotlib.cbook import boxplot_stats

# Create year x-axis
years = [int(x) for x in np.linspace(2006, 2022, 17)]

# Read Literature data into dataframe
df = pd.read_excel('Literature Overview.xlsx')

# Create df with first occurrence of every title
df_as = df[df['System Type'] != 'Human Experts']
df_unique_titles = df.groupby('Title').first()
df_unique_titles_as = df_unique_titles[df_unique_titles['System Type'] != 'Human Experts']
df_unique_titles_he = df_unique_titles[df_unique_titles['System Type'] == 'Human Experts']

# Print general interesting numbers
print("Number of publications on (semi-)automated LR systems:", len(df_unique_titles_as))
print("Number of publications on (semi-)automated LR systems that reported a Cllr:", len(df_unique_titles_as[df_unique_titles_as["Cllr Reported"] == "Yes"]))
print("Number of publications on (semi-)automated LR systems that reported a Cllr and a Cllr min:", len(df_unique_titles_as.dropna(subset=["Cllr min"])[df_unique_titles_as["Cllr Reported"] == "Yes"]))
print("Number of publications on (semi-)automated LR systems for which a Cllr could be calculated:", len(df_unique_titles_as[df_unique_titles_as["Search Category"] == "Cllr could be Calculated"]))
print("Number of publications on (semi-)automated LR systems that compares performance to human experts using Cllr:", len(df_unique_titles_as[df_unique_titles_as["Search Category"] == "Compares Performance to Experts using Cllr"]))
print("Number of publications on human experts:", len(df_unique_titles_he))
print("Number of publications on human expers for which a Cllr could be calculated:", len(df_unique_titles_he[df_unique_titles_he["Search Category"] == "Cllr could be Calculated"]))
print("Number of Cllr values in all publications on (semi-)automated LR systems:", len(df_as['Cllr'].dropna()))
print("Number of Cllr values in all publications on (semi-)automated LR systems that were calculated:", len(df_as[df_as['Search Category'] == "Cllr could be Calculated"]['Cllr'].dropna()))
print("Number of forensically relevant Cllr values in all publications on (semi-)automated LR systems:", len(df_as[df_as["Taken into account for Range"] == True].dropna(subset=["Cllr"])))

# Calculate percentage of papers reporting a Cllr per year and plot
cllr_reported = pd.get_dummies(df_unique_titles['Cllr Reported'])
cllr_reported['Forensic Area'] = df_unique_titles['Forensic Area']
cllr_reported['Forensic Analysis'] = df_unique_titles['Forensic Analysis']
cllr_reported['Year'] = df_unique_titles['Year']
# plt.figure(figsize=(15, 10))
# sns.lineplot(cllr_reported.groupby('Year').mean()['Yes'])
# plt.title('Percentage of publications reporting a Cllr per Year', fontsize=20)
# plt.xticks(years, fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Year', fontsize=20)
# plt.ylabel('% of publications reporting Cllr', fontsize=20)
# plt.show()
# #
# # # Calculate absolute number of papers per year reporting Cllr and plot
# plt.figure(figsize=(15, 10))
# sns.lineplot(df_unique_titles[df_unique_titles['Cllr Reported'] == 'Yes'].groupby('Year')['Authors'].count())
# plt.title('Number of publications reporting a Cllr per Year', fontsize=20)
# plt.xticks(years, fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Year', fontsize=20)
# plt.ylabel('Number of publications reporting Cllr', fontsize=20)
# plt.show()

# Barplot Proportion (Choose own categorical variable)
CAT_VARIABLE = 'Forensic Area'
ORDER_BY = df_unique_titles_as.replace(['Yes', 'No'], [1, 0]).groupby(CAT_VARIABLE).count().sort_values('Cllr Reported', ascending=False).index     # Order by bar heights
# ORDER_BY = years    # Order by year

plt.figure(figsize=(10, 7))
sns.set_color_codes('muted')
ax1 = sns.barplot(df_unique_titles_as.groupby(CAT_VARIABLE).count(),
            x=df_unique_titles_as.groupby(CAT_VARIABLE).count().index,
            y='Authors',
            color='white',
            edgecolor="lightblue", hatch=r"/",
            label='All',
            order=ORDER_BY)
ax2 = sns.barplot(df_unique_titles_as[df_unique_titles_as['Cllr Reported'] == 'Yes'].groupby(CAT_VARIABLE).count(),
            x=df_unique_titles_as[df_unique_titles_as['Cllr Reported'] == 'Yes'].groupby(CAT_VARIABLE).count().index,
            y='Authors',
            color="b",
            label='Proportion Reporting Cllr',
            order=ORDER_BY)
proportions = [str(x) + '%' for x in (cllr_reported.groupby(CAT_VARIABLE).mean().reindex(ORDER_BY) * 100).round(1)["Yes"]]
ax1.bar_label(ax1.containers[0], labels=proportions)
sns.despine(left=True, bottom=True)
plt.legend(loc='upper left')
# plt.title('Number of Publications on (semi-)Automated LR Systems per Forensic Area')
plt.title('Number of Publications on (semi-)Automated LR Systems per Year')
plt.xticks(fontsize=10, rotation=90)
plt.ylabel('Number of Publications')
plt.tight_layout()
plt.show()

# Plot proportion of cllr in every area
# plt.figure(figsize=(10, 5))
# sns.set_color_codes('muted')
# ax = sns.barplot(cllr_reported.groupby('Forensic Area').mean().round(3) * 100,
#             x=cllr_reported.groupby('Forensic Area').mean().index,
#             y="Yes",
#             color='b',
#             order=ORDER_BY)
# ax.bar_label(ax.containers[0])
# sns.despine(left=True, bottom=True)
# plt.xticks(fontsize=8, rotation=90)
# plt.ylabel('Proportion of Publications Reporting Cllr (%)')
# plt.tight_layout()
# plt.show()

# sns.histplot(df_unique_titles[df_unique_titles['Forensic Area'] == 'Forensic Biology'], x='Year', hue='Forensic Area', bins=years)
# plt.show()

# Plot publications not using Cllr per forensic area
# sns.barplot(df_unique_titles_as[df_unique_titles_as['Cllr Reported'] == 'No'][df_unique_titles_as["Year"].isin([2019, 2020, 2021, 2022])].groupby(CAT_VARIABLE).count(),
#             x=df_unique_titles_as[df_unique_titles_as['Cllr Reported'] == 'No'][df_unique_titles_as["Year"].isin([2019, 2020, 2021, 2022])].groupby(CAT_VARIABLE).count().index,
#             y='Authors',
#             color="b",
#             label='Proportion Not Reporting Cllr',
#             order=df_unique_titles_as[df_unique_titles_as['Cllr Reported'] == 'No'][df_unique_titles_as["Year"].isin([2019, 2020, 2021, 2022])].groupby(CAT_VARIABLE).count().sort_values('Authors', ascending=False).index)
# sns.despine(left=True, bottom=True)
# plt.legend()
# plt.title('Publications of (semi-)automated Likelihood Ratio Methods not Reporting Cllrs')
# plt.xticks(fontsize=8, rotation=90)
# plt.ylabel('Number of Publications')
# plt.tight_layout()
# plt.show()

# Calculate basic percentage statistics
# papers_reporting_cllr = len(df_unique_titles_as[df_unique_titles_as['Cllr Reported'] == 'Yes']) / len(df_unique_titles_as)
# papers_not_reporting_cllr = 1 - papers_reporting_cllr
# colors = sns.color_palette('muted')[:2]
# plt.pie([papers_not_reporting_cllr, papers_reporting_cllr], autopct='%1.1f%%', colors=colors, labels=['Cllr Not Reported', 'Cllr Reported'])
# # plt.title(textwrap.fill("Proportion of Publications on (Semi-)Automated LR Systems Reporting a Cllr Value", 50))
# plt.show()

# Plot search category distribution
# plt.figure(figsize=(10, 10))
# search_category_df = df_unique_titles_as[df_unique_titles_as["Cllr Reported"] == "Yes"].groupby('Search Category').count()["Authors"] / sum(df_unique_titles_as[df_unique_titles_as["Cllr Reported"] == "Yes"].groupby('Search Category').count()["Authors"])
# colors = sns.color_palette('muted')[:len(search_category_df)]
# plt.pie(search_category_df.values, autopct='%1.1f%%', colors=colors, labels=[textwrap.fill(text, 25) for text in search_category_df.index], textprops={'fontsize': 20})
# plt.tight_layout()
# # plt.legend([textwrap.fill(text, 25) for text in search_category_df.index], fontsize=20, loc="upper right")
# # plt.title("Proportion of Publications fallin in every Search Criterium")
# plt.show()

# plt.figure(figsize=(10, 10))
# search_category_df = df_unique_titles_as[df_unique_titles_as["Cllr Reported"] == "No"].groupby('Search Category').count()["Authors"] / sum(df_unique_titles_as[df_unique_titles_as["Cllr Reported"] == "No"].groupby('Search Category').count()["Authors"])
# colors = sns.color_palette('muted')[:len(search_category_df)]
# plt.pie(search_category_df.values, autopct='%1.1f%%', colors=colors, labels=[textwrap.fill(text, 25) for text in search_category_df.index], textprops={'fontsize': 20})
# plt.tight_layout()
# # plt.legend([textwrap.fill(text, 25) for text in search_category_df.index], fontsize=20, loc="upper right")
# # plt.title("Proportion of Publications fallin in every Search Criterium")
# plt.show()

# Look at score-based vs feature-based methods
# sns.boxplot(df, x='Forensic Area', y='Cllr', hue='Model Type')
# plt.xticks(rotation=90)
# plt.show()

# Number of Cllr reports per forensic area
# plt.figure(figsize=(20, 10))
# sns.barplot(df.groupby('Forensic Area').count(), x=df.groupby('Forensic Area').count().index, y='Authors', color='b', order=df.groupby('Forensic Area').count().sort_values('Authors', ascending=False).index)
# plt.title('Publications reporting Cllr per Forensic Area', fontsize=20)
# plt.xticks(fontsize=10, rotation=50)
# plt.yticks(fontsize=20)
# plt.xlabel('Forensic Expertise', fontsize=30)
# plt.ylabel('Number of Publications', fontsize=30)
# plt.show()
#
# Boxplot of Cllrs per forensic expertise area
# df_relevant_cllrs = df_as[df_as["Taken into account for Range"] == True].groupby('Dataset').min()
# plt.figure(figsize=(15, 10))
# sns.set_color_codes('muted')
# label_counts = df_relevant_cllrs.dropna(subset=['Cllr']).groupby('Forensic Area').count().sort_values('Cllr', ascending=False)['Cllr']
# cllr_count_order = label_counts.index
# ax = sns.boxplot(df_relevant_cllrs.dropna(subset=['Cllr']), x='Forensic Area', y='Cllr', order=cllr_count_order, color='lightblue')
# sns.despine(left=True, bottom=True)
# plt.title('Cllrs per Forensic Area', fontsize=20)
# plt.xticks(fontsize=20, rotation=90)
# ax.set_xticklabels(labels=[f"{y} ({str(x)})" for x,y in zip(label_counts, label_counts.index)])
# plt.yticks(fontsize=20)
# plt.ylim(0, 1.5)
# plt.xlabel('Forensic Expertise', fontsize=20)
# plt.ylabel('Cllr', fontsize=20)
# plt.tight_layout()
# for (x, area) in zip(ax.get_xticks(), cllr_count_order):
#     cllrs = df_relevant_cllrs[df_relevant_cllrs['Forensic Area'] == area].dropna(subset=['Cllr'])['Cllr'].values
#     if len(cllrs) > 2:
#         ci = st.t.interval(confidence=0.95, df=len(cllrs)-1, loc=np.mean(cllrs), scale=st.sem(cllrs))
#         if ci[0] < 0:
#             ci = (0, ci[1])
#         print(f"95% confidence interval {area}: {ci}")
#         plt.plot((x, x), ci, 'ro-', label='95% Confidence Interval' if x==0 else "")
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(10, 10))
# ax = sns.boxplot(data=df_min_three_area, x='Forensic Area', y='Cllr', order=cllr_count_order)
# sns.despine(left=True, bottom=True)
# label_counts = df_min_three_area.dropna(subset=['Cllr']).groupby('Forensic Area').count().sort_values('Cllr', ascending=False)['Cllr']
# plt.title('Smallest 3 Cllrs per Forensic Area', fontsize=20)
# plt.xticks(fontsize=20, rotation=90)
# ax.set_xticklabels(labels=[f"{y} ({str(x)})" for x,y in zip(label_counts, label_counts.index)])
# plt.yticks(fontsize=20)
# plt.xlabel('Forensic Expertise', fontsize=20)
# plt.ylabel('Cllr', fontsize=20)
# plt.tight_layout()
# plt.show()

# Plot boxplot of Cllrs in expertise area
plt.figure(figsize=(15, 10))
FORENSIC_AREA = 'Forensic Anthropology and Taphonomy'
area_df = df_as[(df_as['Forensic Area'] == FORENSIC_AREA) & (df_as['Taken into account for Range'] == True)].groupby('Dataset').min()
sns.set_color_codes('muted')
label_counts = area_df.dropna(subset=['Cllr']).groupby('Forensic Analysis').count().sort_values('Cllr', ascending=False)['Cllr']
ax = sns.boxplot(area_df.dropna(subset=['Cllr']), x='Forensic Analysis', y='Cllr', order=label_counts.index, color='lightblue')
ax.set_xticklabels(labels=[f"{y} ({str(x)})" for x,y in zip(label_counts, label_counts.index)])
# plt.title('Cllrs per Analysis')
sns.despine(left=True, bottom=True)
for (x, analysis) in zip(ax.get_xticks(), label_counts.index):
    cllrs = area_df[area_df['Forensic Analysis'] == analysis].dropna(subset=['Cllr'])['Cllr'].values
    outliers = boxplot_stats(cllrs).pop(0)['fliers']
    # cllrs = np.setdiff1d(cllrs, outliers) # Remove outliers
    if len(cllrs) > 2:
        ci = st.t.interval(confidence=0.95, df=len(cllrs)-1, loc=np.mean(cllrs), scale=st.sem(cllrs))
        if ci[0] < 0:
            ci = (0, ci[1])
        print(f"95% confidence interval {analysis}: {ci}")
        plt.plot((x, x), ci, 'ro-', label='95% Confidence Interval' if x==0 else "")
plt.legend(fontsize=15)
plt.xticks(fontsize=15, rotation=50)
plt.xlabel('Forensic Analysis', fontsize=15)
plt.ylabel("Cllr", fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.show()

# Barplot Proportion within forensic area (Choose own categorical variable)
# CAT_VARIABLE = 'Forensic Analysis'
# ORDER_BY = area_df.replace(['Yes', 'No'], [1, 0]).groupby(CAT_VARIABLE).count().sort_values('Cllr Reported', ascending=False).index  # Order by bar heights
# ORDER_BY = years    # Order by year

# plt.figure(figsize=(10, 7))
# sns.set_color_codes('muted')
# ax1 = sns.barplot(area_df.groupby(CAT_VARIABLE).count(),
#             x=area_df.groupby(CAT_VARIABLE).count().index,
#             y='Authors',
#             color='white',
#             edgecolor="lightblue", hatch=r"/",
#             label='All',
#             order=ORDER_BY)
# ax2 = sns.barplot(area_df[area_df['Cllr Reported'] == 'Yes'].groupby(CAT_VARIABLE).count(),
#             x=area_df[area_df['Cllr Reported'] == 'Yes'].groupby(CAT_VARIABLE).count().index,
#             y='Authors',
#             color="b",
#             label='Proportion Reporting Cllr',
#             order=ORDER_BY)
# proportions = [str(x) + '%' for x in (cllr_reported.groupby(CAT_VARIABLE).mean().reindex(ORDER_BY) * 100).round(1)["Yes"]]
# ax1.bar_label(ax1.containers[0], labels=proportions)
# sns.despine(left=True, bottom=True)
# plt.legend(loc='upper right')
# plt.title(f"Number of Publications on (semi-)Automated LR Systems per Forensic Analysis within {FORENSIC_AREA}")
# # plt.title(f"Number of Publications on (semi-)Automated LR Systems per Year within {FORENSIC_AREA}")
# plt.xticks(fontsize=10, rotation=0)
# label_counts = area_df.groupby(CAT_VARIABLE).count().sort_values('Authors', ascending=False)['Authors']
# ax1.set_xticklabels(labels=[f"{y} ({str(x)})" for x,y in zip(label_counts, label_counts.index)])
# plt.ylabel('Number of Publications')
# plt.tight_layout()
# plt.show()

# Plot boxplot of difference between human experts and automated systems
# analysis_df = df[df['Forensic Analysis'] == 'Glass Individualisation']
# sns.set_color_codes('muted')
# label_counts = analysis_df.dropna(subset=['Cllr']).groupby('System Type').count().sort_values('Cllr', ascending=False)['Cllr']
# ax = sns.boxplot(analysis_df[(analysis_df['Taken into account for Range'] == True) | (analysis_df['System Type'] == 'Human Experts')],
#             x='System Type', y='Cllr', color='lightblue')
# ax.set_xticklabels(labels=[f"{y} ({str(x)})" for x,y in zip(label_counts, label_counts.index)])
# plt.title('Cllrs of (semi-)Automated Systems vs Human Experts')
# sns.despine(left=True, bottom=True)
# plt.show()

# Get lowest Cllrs per dataset of certain analysis
min_cllr_per_dataset = df_as[(df_as['Forensic Analysis'] == 'Speaker Recognition') & (df_as['Taken into account for Range'] == True)].groupby('Dataset')['Cllr'].min().sort_values()
print(min_cllr_per_dataset)
