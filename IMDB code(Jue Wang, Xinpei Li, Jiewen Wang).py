
# coding: utf-8

#%%


import pandas as pd
import numpy as np
import networkx as nx
'''
Import the actor dataset and name it actor_1

We don't use this dataset for direct analysis, only for constructing the dictionary of 
actorID to Names.
'''
actor_1 = pd.read_csv('name.basics.tsv', '\t')



#Create a dictionary for {actorID: birth Year}
actor_1 = actor_1[actor_1['birthYear'] != '\N']

actor_1['birthYear'] = actor_1['birthYear'].map(lambda x: int(x))

actor_birth = {}
for index, row in actor_1.iterrows():
    actor_birth[row['nconst']] = row['birthYear']

'''
Import the cast dataset and name it all_cast.

We drop entries that duplicate actorID and character, so to prevent a large number
of TV show actor entries in the dataset with multiple episodes per show.
'''


cast = pd.read_csv('title.principals.tsv', '\t')

cast = cast.dropna()

cast_actor = cast[cast['category']=="actor"]
cast_actress = cast[cast['category']=="actress"]
all_cast = pd.concat([cast_actor, cast_actress], axis = 0)


all_cast = all_cast.drop_duplicates(subset = ['nconst', 'characters'], keep = False)

#Initiate the dictionaries that we will use later

edge_movienum = {}

actor_nummovies = {}

movie_actorlist = {}
'''
Partition the all_cast dataset into groups with unique actorID and unique movieID to 
prevent looing through the whole dataset.
'''
actor_group = all_cast.groupby('nconst')
cast_group = all_cast.groupby('tconst')

#%%

    
#%%

# Creating dictionary of actor to the number of movies they featured in

for i, j in actor_group:
    actor_nummovies[i] = len(j)
# Creating dictionary of movie to its cast list
for i, j in cast_group:
    if i in movie_actorlist:
        movie_actorlist[i].extend(j['nconst'])
    else:
        movie_actorlist[i] = []
        movie_actorlist[i].extend(j['nconst'])
#%%

for m in movie_actorlist.items():
    actor_list = m[1]
    # if actor # is less than 2 then get rid of this row for convenience
    if len(actor_list)-1 > 1:
        index = len(actor_list)-1
        for i in range(1,index):
            for j in range(i+1,index+1):
                key1 = (actor_list[i],actor_list[j])
                key2 = (actor_list[j],actor_list[i])
                if key1 in edge_movienum:
                    edge_movienum[key1] += 1
                    edge_movienum[key2] += 1
                else:
                    edge_movienum[key1] = 1
                    edge_movienum[key2] = 1
#%%

edge_movienum_weight = {}
for keypair in edge_movienum.items():
    # keypair[1] = the frequency of pair<actor1, actor2>
    # keypair[0] = <actor1, actor2>
    weight = float(keypair[1])/int(actor_nummovies[str(keypair[0][0])]); #can change
    edge_movienum_weight[keypair[0]] = weight
#%%
    
# subset the weight dictionary using the criteria that weight must fall inside [0.7, 1.0]
new_weight = {}
for pair,weight in edge_movienum_weight.iteritems():
    if weight > 0.7 and weight < 1.0:
        new_weight[pair] = weight
    
nummovies = {}
for pair, weight in new_weight.iteritems():
    if pair[0] in nummovies:
        nummovies[pair[0]] += 1
    else:
        nummovies[pair[0]] = 1
    if pair[1] in nummovies:
        nummovies[pair[1]] += 1
    else:
        nummovies[pair[1]] = 1

# further subset the dictionary; the total number of times the pair has occured in the dictionary should be greater than 9
weight_plot = {}
for pair,weight in new_weight.iteritems():
    if nummovies[pair[0]] + nummovies[pair[1]]>9:
        weight_plot[pair] = weight

#%%
# the cast that exists in the subsetted version
selected_cast = []
for item in weight_plot.items():
    x, y = item[0]
    if x in selected_cast:
        pass
    else:
        selected_cast.append(x)
    if y in selected_cast:
        pass
    else:
        selected_cast.append(y)
'''
Import the actor dataset again
get a {nconst:name} dictionary
'''
actor_2 = pd.read_csv('name.basics.tsv', '\t')
labels = {}
for i in selected_cast:
    labels[i] = actor_2[actor_2['nconst'] == i]['primaryName'].tolist()[0]

# weight dictionary with actual names
new_weight_plot = {}
for pair, w in weight_plot.iteritems():
    new_weight_plot[(labels[pair[0]], labels[pair[1]])] = w

new_weight_plot

# the list with only actual name
new_selected_cast = []
for i in selected_cast:
    new_selected_cast.append(labels[i])

# creating the graph for plotting and pagerank
G = nx.DiGraph()
G.add_nodes_from(new_selected_cast)
for pair, w in new_weight_plot.items():
    x, y = pair
    G.add_edge(x, y, weight = w)

# pagerank
pr = nx.pagerank(G,alpha = 0.85)
sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse = True)
sorted_pr
import matplotlib.pyplot as plt
# darker the color => higher pagerank
plt.figure(figsize=(18,18))
nx.draw_circular(G,font_size = 20, with_labels = True, node_color=range(len(new_selected_cast)), node_size = 700, cmap=plt.cm.Blues)
plt.show
plt.savefig("plot.png", dpi=1000)



len(new_selected_cast)



#%%

# Plotting 3D graph using plotly
import igraph as ig
import plotly 
plotly.tools.set_credentials_file(username='daniel123wang', api_key='keYqKMg2gxNeg44Nh5iU')

actor_to_num = {}
for i in range(len(new_selected_cast)):
    actor_to_num[new_selected_cast[i]] = i

Edges_old = new_weight_plot.keys()
Edges = [(actor_to_num[i[0]], actor_to_num[i[1]])for i in Edges_old]
Edges
N = 32
labels = new_selected_cast

G = ig.Graph(Edges, directed = False)
layt=G.layout('kk', dim=3)
layt[5]
Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
Yn=[layt[k][1] for k in range(N)]# y-coordinates
Zn=[layt[k][2] for k in range(N)]# z-coordinates
Xe=[]
Ye=[]
Ze=[]
for e in Edges:
    Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
    Ye+=[layt[e[0]][1],layt[e[1]][1], None]
    Ze+=[layt[e[0]][2],layt[e[1]][2], None]
import plotly.plotly as py
import plotly.graph_objs as go
trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=1),
               hoverinfo='none'
               )
trace2=go.Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='markers',
               name='actors',
               marker=dict(symbol='dot',
                             size=6,
                             colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=0.5)
                             ),
               text = labels,
               hoverinfo='text'
               )
axis=dict(showbackground=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

layout = go.Layout(
         title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
         width=1000,
         height=1000,
         showlegend=False,
         scene=dict(
             xaxis=dict(axis),
             yaxis=dict(axis),
             zaxis=dict(axis),
        ),
     margin=dict(
        t=100
    ),
    hovermode='closest',
    annotations=[
           dict(
           showarrow=False,
            text="Data source: <a href='http://bost.ocks.org/mike/miserables/miserables.json'>[1] miserables.json</a>",
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
            size=14
            )
            )
        ],    )

data=[trace1, trace2]
fig=go.Figure(data=data, layout=layout)

py.iplot(fig)

#%%

# creating dictionary of {movieID: actual movie name}
movie_name = pd.read_csv('title.basics.tsv', '\t')

movie_name_dict = {}
for index, row in movie_name.iterrows():
    movie_name_dict[row['tconst']] = row['primaryTitle']

movie_name_dict

#%%
# Cleaning the movie dataset
movie_name = movie_name[movie_name['startYear'] != '\N']
movie_name['startYear'] = movie_name['startYear'].map(lambda x: int(x))
# movie_year {movie: startyear}
movie_year = {}
for index, row in movie_name.iterrows():
    movie_year[row['tconst']] = row['startYear']

movie_year
#%%

# dictionary of ratings of movie
rating = pd.read_csv("title.ratings.tsv", '\t')

movie_rating = {}

for index, row in rating.iterrows():
    movie_rating[row['tconst']] = row['averageRating']
    

# dictionary of the average age of cast at the time of making the movie
movie_avg_age = {}
for i, j in movie_actorlist.iteritems():
    if i in movie_year:
        total_age = 0
        n = 0
        for k in j:
            if k in actor_birth:
                total_age += (movie_year[i] - actor_birth[k])
            else:
                n += 1
        if len(j) != n:
            movie_avg_age[i] = total_age / (len(j) - n)

# Dictionary of number of votes
num_votes = {}
for index, row in rating.iterrows():
    num_votes[row['tconst']] = row['numVotes']
    

# Create a smaller subset of movies to get the training set to do cluster from
movie_subset = rating[(rating['averageRating'] > 7.0) & (rating['numVotes'] > 10000.0)]
len(movie_subset)
train_data = np.zeros([4156, 4])

movie_subset = movie_subset.reset_index(drop=True)

for i in range(4156):
    a = movie_subset.loc[i]['averageRating']
    b = movie_subset.loc[i]['numVotes']
    if movie_subset.loc[i]['tconst'] in movie_avg_age:
        c = movie_avg_age[movie_subset.loc[i]['tconst']]
    else:
        c = 0
    if movie_subset.loc[i]['tconst'] in movie_actorlist:
        d = actor_nummovies[movie_actorlist[movie_subset.loc[i]['tconst']][0]]
    else:
        d = 0

    train_data[i,:] = [a, b, c, d]

# K-means
from sklearn.cluster import KMeans
kmeans = KMeans(3)
kmeans.fit(train_data)
a = kmeans.labels_
seq_0 = []
seq_1 = []
seq_2 = []
for i in range(len(a)):
    if a[i] == 0:
        seq_0.append(i)
    if a[i] == 1:
        seq_1.append(i)
    if a[i] == 2:
        seq_2.append(i)

group_1 = movie_subset.loc[seq_0] 
group_2 = movie_subset.loc[seq_1]
group_3 = movie_subset.loc[seq_2]


# get the list of names of movies for the 3 clusters
group_1_name = []
for index, row in group_1.iterrows():
    group_1_name.append(movie_name_dict[row['tconst']])

group_2_name = []
for index, row in group_2.iterrows():
    group_2_name.append(movie_name_dict[row['tconst']])

group_3_name = []
for index, row in group_3.iterrows():
    group_3_name.append(movie_name_dict[row['tconst']])
    
group_1_name

# get the average ratings of the three clusters
ratings_1 = 0.0
for index, row in group_1.iterrows():
    rate = movie_rating[row['tconst']]
    ratings_1 = ratings_1 + rate
avg_rating_1 = ratings_1 / len(group_1)
avg_rating_1

ratings_2 = 0.0
for index, row in group_2.iterrows():
    rate = movie_rating[row['tconst']]
    ratings_2 = ratings_2 + rate
avg_rating_2 = ratings_2 / len(group_2)
avg_rating_2

ratings_3 = 0.0
for index, row in group_3.iterrows():
    rate = movie_rating[row['tconst']]
    ratings_3 = ratings_3 + rate
avg_rating_3 = ratings_3 / len(group_3)
avg_rating_3

group_3_name


# group_2 has the highest average rating
group_2_name
# compare the average age of the three clusters
age_2 = 0
n = 0
for index, row in group_2.iterrows():
    if row['tconst'] in movie_avg_age:
        age_2 += movie_avg_age[row['tconst']]
    else:
        n += 1
avg_age_2 = age_2 / (len(group_2) - 1)
avg_age_2

age_3 = 0
n = 0
for index, row in group_1.iterrows():
    if row['tconst'] in movie_avg_age:
        age_3 += movie_avg_age[row['tconst']]
    else:
        n += 1
avg_age_3 = age_3 / (len(group_3) - 1)
avg_age_3

# Average number of movies the leading cast has been in for the three clusters
num_movies = 0
n = 0
for index, row in group_1.iterrows():
    if row['tconst'] in movie_actorlist:
        num_movies += actor_nummovies[movie_actorlist[row['tconst']][0]]
    else:
        n += 1
    
avg_num_movies_1 = num_movies / (len(group_1) - n)
avg_num_movies_1

#%%

# Using plotly to create a pandas heatmap on the three clusters
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
grouped_1 = train_data[seq_0] 
grouped_2 = train_data[seq_1] 
grouped_3 = train_data[seq_2]
Y = np.repeat([0, 1, 2],[517, 91, 3548])
len(Y)
all_data = np.concatenate((grouped_1, grouped_2, grouped_3), axis=0)
all_data = pd.DataFrame(all_data)
all_data['Y'] = Y
all_data[1] = all_data[1] / 10000
grouped = all_data.groupby(['Y'], sort=True)

# compute sums for every column in every group
sums = grouped.sum()
sums

sums

data = [go.Heatmap( z=sums.values.tolist(), 
                   y=['1st cluster', '2nd cluster', '3rd cluster'],
                   x=['Ratings',
                      '#Votes',
                      'Cast Average Age',
                      'Lead Actor experience'
                     ],
                   colorscale='Viridis')]
py.iplot(data, filename='pandas-heatmap')
