# -*- coding: utf-8 -*-
"""
Key to results data:

Div = League Division
Date = Match Date (dd/mm/yy)
HomeTeam = Home Team
AwayTeam = Away Team
FTHG and HG = Full Time Home Team Goals
FTAG and AG = Full Time Away Team Goals
FTR and Res = Full Time Result (H=Home Win, D=Draw, A=Away Win)
HTHG = Half Time Home Team Goals
HTAG = Half Time Away Team Goals
HTR = Half Time Result (H=Home Win, D=Draw, A=Away Win)

Match Statistics (where available)
Attendance = Crowd Attendance
Referee = Match Referee
HS = Home Team Shots
AS = Away Team Shots
HST = Home Team Shots on Target
AST = Away Team Shots on Target
HHW = Home Team Hit Woodwork
AHW = Away Team Hit Woodwork
HC = Home Team Corners
AC = Away Team Corners
HF = Home Team Fouls Committed
AF = Away Team Fouls Committed
HFKC = Home Team Free Kicks Conceded
AFKC = Away Team Free Kicks Conceded
HO = Home Team Offsides
AO = Away Team Offsides
HY = Home Team Yellow Cards
AY = Away Team Yellow Cards
HR = Home Team Red Cards
AR = Away Team Red Cards
HBP = Home Team Bookings Points (10 = yellow, 25 = red)
ABP = Away Team Bookings Points (10 = yellow, 25 = red)
"""
import numpy as np
import pandas as pd

path = 'E:\\ML\\Datasets'

data = pd.read_csv(f'{path}\\SP1.csv')

real_madrid = data[(data['HomeTeam'] == 'Real Madrid') | (data['AwayTeam'] == 'Real Madrid')].reset_index(drop=True)
real_madrid = real_madrid.drop('Div', axis=1)
cols = list(real_madrid.columns)
cols = cols[:21]
real_madrid = real_madrid[cols]

#Converting table
for i in range(real_madrid.shape[0]):
    if real_madrid.loc[i, 'HomeTeam'] == 'Real Madrid':     
        real_madrid.loc[i, 'where'] = 'home'  
    else:      
        real_madrid.loc[i, 'where'] = 'away'
        
for i in range(real_madrid.shape[0]):
    if real_madrid.loc[i, 'where'] == 'away' and real_madrid.loc[i,'FTR'] == 'A':
        real_madrid.loc[i, 'result'] = 1
    elif real_madrid.loc[i, 'where'] == 'home' and real_madrid.loc[i,'FTR'] == 'H':
        real_madrid.loc[i, 'result'] = 1
    elif real_madrid.loc[i,'FTR'] == 'D':
        real_madrid.loc[i, 'result'] = 2
    else:
        real_madrid.loc[i, 'result'] = 0

for i in range(real_madrid.shape[0]):
    if real_madrid.loc[i, 'where'] == 'away':
        real_madrid.loc[i, 'vs'] = real_madrid.loc[i, 'HomeTeam']
    else:
        real_madrid.loc[i, 'vs'] = real_madrid.loc[i, 'AwayTeam']

real_madrid = real_madrid.drop(['Date', 'HomeTeam', 'AwayTeam'], axis=1)
real_madrid = real_madrid.drop(['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG'], axis=1)

for i in range(real_madrid.shape[0]):
    if real_madrid.loc[i, 'where'] == 'away' and real_madrid.loc[i,'HTR'] == 'A':
        real_madrid.loc[i, 'ht_result'] = 1
    elif real_madrid.loc[i, 'where'] == 'home' and real_madrid.loc[i,'HTR'] == 'H':
        real_madrid.loc[i, 'ht_result'] = 1
    elif real_madrid.loc[i,'HTR'] == 'D':
        real_madrid.loc[i, 'ht_result'] = 2
    else:
        real_madrid.loc[i, 'ht_result'] = 0
real_madrid = real_madrid.drop(['HTR'], axis=1)

for i in range(real_madrid.shape[0]):
    if real_madrid.loc[i, 'where'] == 'away':
        real_madrid.loc[i, 'rm_shots'] = real_madrid.loc[i, 'AS']
        real_madrid.loc[i, 'rm_shots_ratio'] = real_madrid.loc[i, 'AST']/real_madrid.loc[i, 'AS']
        real_madrid.loc[i, 'vs_shots'] = real_madrid.loc[i, 'HS']
        real_madrid.loc[i, 'vs_shots_ratio'] = real_madrid.loc[i, 'HST']/real_madrid.loc[i, 'HS']
    elif real_madrid.loc[i, 'where'] == 'home':
        real_madrid.loc[i, 'rm_shots'] = real_madrid.loc[i, 'HS']
        real_madrid.loc[i, 'rm_shots_ratio'] = real_madrid.loc[i, 'HST']/real_madrid.loc[i, 'HS']
        real_madrid.loc[i, 'vs_shots'] = real_madrid.loc[i, 'AS']
        real_madrid.loc[i, 'vs_shots_ratio'] = real_madrid.loc[i, 'AST']/real_madrid.loc[i, 'AS']

real_madrid = real_madrid.drop(['HS', 'AS', 'AST', 'HST'], axis=1)
for i in range(real_madrid.shape[0]):
    real_madrid.loc[i, 'total_fouls'] = real_madrid.loc[i, 'HF'] + real_madrid.loc[i, 'AF']
    if real_madrid.loc[i, 'where'] == 'home':
        real_madrid.loc[i, 'pct_fouls'] = real_madrid.loc[i, 'HF']/real_madrid.loc[i, 'total_fouls']
    else:
        real_madrid.loc[i, 'pct_fouls'] = real_madrid.loc[i, 'AF']/real_madrid.loc[i, 'total_fouls']
real_madrid = real_madrid.drop(['HF', 'AF'], axis=1)
for i in range(real_madrid.shape[0]):
    if real_madrid.loc[i, 'where'] == 'home':
        real_madrid.loc[i, 'rm_corners'] = real_madrid.loc[i, 'HC']
        real_madrid.loc[i, 'vs_corners'] = real_madrid.loc[i, 'AC']
    else:
        real_madrid.loc[i, 'rm_corners'] = real_madrid.loc[i, 'AC']
        real_madrid.loc[i, 'vs_corners'] = real_madrid.loc[i, 'HC']
real_madrid = real_madrid.drop(['HC', 'AC'], axis=1)

for i in range(real_madrid.shape[0]):
    real_madrid.loc[i, 'total_yc'] = real_madrid.loc[i, 'HY'] + real_madrid.loc[i, 'AY']
    if real_madrid.loc[i, 'where'] == 'home':
        real_madrid.loc[i, 'pct_yc'] = real_madrid.loc[i, 'HY']/real_madrid.loc[i, 'total_yc']
        real_madrid.loc[i, 'rm_rc'] = real_madrid.loc[i, 'HR']
        real_madrid.loc[i, 'vs_rc'] = real_madrid.loc[i, 'AR']
    else:
        real_madrid.loc[i, 'pct_yc'] = real_madrid.loc[i, 'AY']/real_madrid.loc[i, 'total_yc']
        real_madrid.loc[i, 'rm_rc'] = real_madrid.loc[i, 'AR']
        real_madrid.loc[i, 'vs_rc'] = real_madrid.loc[i, 'HR']
real_madrid = real_madrid.drop(['HR', 'AR', 'HY', 'AY'], axis=1)
#'where' = 1 home, 0 away
# 1 - stands for win
# 0 - stands for lose
# 2 -stands for draw
real_madrid['where'] = np.where(real_madrid['where']=='home', 1, 0)
real_madrid = real_madrid.drop(['away'], axis=1)
teams = real_madrid['vs']
real_madrid = real_madrid.drop('vs', axis=1)
real_madrid.corr()
targets = real_madrid['result']
real_madrid_clf = real_madrid.drop('result', axis=1)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(real_madrid_clf, targets)

from sklearn.decomposition import PCA
pca = PCA(random_state=0)
pca.fit(real_madrid_clf)

print(f'First two explain: {np.sum(pca.explained_variance_ratio_[:2])}')
print(f'First four explain: {np.sum(pca.explained_variance_ratio_[:4])}')

pca = PCA(random_state=0, n_components=2).fit(real_madrid_clf)
reduced_rm = pd.DataFrame(pca.transform(real_madrid_clf), columns=['PC1', 'PC2'])
pca_samples = pca.transform(real_madrid_clf.loc[[4, 20, 28], :])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for c in range(2, 6):
    clusterer = KMeans(n_clusters=c ,random_state=0)
    clusterer.fit(reduced_rm)
    preds = clusterer.predict(reduced_rm)
    score = silhouette_score(reduced_rm, labels=preds)
    print(f'Silhoutte score for {c}: {score}')
clusterer = KMeans(n_clusters=3 ,random_state=0)
clusterer.fit(reduced_rm)
preds = clusterer.predict(reduced_rm)
centers = clusterer.cluster_centers_
centers_real = pca.inverse_transform(centers)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
def cluster_results(reduced_data, preds, centers, pca_samples):
	'''
	Visualizes the PCA-reduced cluster data in two dimensions
	Adds cues for cluster centers and student-selected sample data
	'''

	predictions = pd.DataFrame(preds, columns = ['Cluster'])
	plot_data = pd.concat([predictions, reduced_data], axis = 1)

	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Color map
	cmap = cm.get_cmap('gist_rainbow')

	# Color the points based on assigned cluster
	for i, cluster in plot_data.groupby('Cluster'):   
	    cluster.plot(ax = ax, kind = 'scatter', x = 'PC1', y = 'PC2', \
	                 color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i), s=30);

	# Plot centers with indicators
	for i, c in enumerate(centers):
	    ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \
	               alpha = 1, linewidth = 2, marker = 'o', s=200);
	    ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=100);
    # Plot transformed sample points 
	ax.scatter(x = pca_samples[:,0], y = pca_samples[:,1], \
	           s = 150, linewidth = 4, color = 'black', marker = 'x');
	# Set plot title
	ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross");
    
cluster_results(reduced_rm, preds, centers, pca_samples)


def pca_results(good_data, pca):
	'''
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
	Visualizes the PCA results
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	fig, ax = plt.subplots(figsize = (14,8))

	# Plot the feature weights as a function of the components
	components.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dimensions, rotation=0)


	# Display the explained variance ratios
	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)

pca_results(real_madrid_clf, pca)