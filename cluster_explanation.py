"""
***********************************************************
(c) Ian Davidson and Davidson Lab. Please cite

The Cluster Description Problem-Complexity Results, Formulations and Approximations, I Davidson, A Gourru and SS. Ravi
Advances in Neural Information Processing Systems, 2018, 6191-6201
***********************************************************
"""

import numpy as np
import pandas as pd
from mip import Model, minimize, BINARY, xsum, OptimizationStatus, MINIMIZE
import sys
import configparser
from time import sleep

# try:
#     config_filename = sys.argv[1]
# except IndexError:
#     print("IndexError: Please provide a configuration file name."
#           "\n Suggested syntax: python3 cluster_explanation.py config_filename.txt", file=sys.stderr)
#     exit(1)

config_filename = 'mushroom/configMushroom.ini'
try:
    config_file = open(config_filename, "r")
except FileNotFoundError:
    print("FileNotFoundError:", config_filename,
          "was not found. Please check the name and directory of your configuration file.",
          file=sys.stderr)
    exit(1)

config = configparser.ConfigParser()
config.read_file(config_file)


nodes_filename = config["File Names"]["nodes"]
hashtags_filename = config["File Names"]["hashtags"]
communities_filename = config["File Names"]["communities"]
user_hashtag_matrix_filename = config["File Names"]["user_hashtag_matrix"]

solver_name = config["Parameters"]["solver"]
optimization_runtime = config["Parameters"].getint("optimization_time")
write_model_file = config["Parameters"].getboolean("write_model_file")
model_filename = config["Parameters"]["model_filename"]

min_post_count = config["Parameters"].getint("minimum_post_count")
min_unique_tags = config["Parameters"].getint("minimum_unique_tags")
minimum_overlap = config["Parameters"].getint("minimum_overlap")

cover_or_forget = config["Parameters"].getboolean("cover_or_forget")
apart_constraint = config["Parameters"].getboolean("apart_constraint")
together_constraint = config["Parameters"].getboolean("together_constraint")
use_maximal_itemsets = config["Parameters"].getboolean("use_maximal_itemsets")


# Reading in the node names
# No header
nodes = pd.read_csv(nodes_filename, header=None)

# Reading in the hashtag names
# No header
hashtags = pd.read_csv(hashtags_filename, header=None)

# Reading in the communities list.
# Header is default [node_name, community_number].
communities = pd.read_csv(communities_filename)
communities_tagname = communities.columns[1]

# Reading in the NxT matrix for user-hashtag usages
# Rows = Users
# Columns = Tags
user_hashtag_matrix = pd.read_csv(user_hashtag_matrix_filename, header=None)

if use_maximal_itemsets:
    maximal_list = config['Parameters']['maximal_itemsets']
    maximal_list = maximal_list.strip().split('\n')
    maximal = {}
    for string in maximal_list:
        split = string.split(', ')
    #     maximal[split[0]] = []
    #     for item in split:
    #         if item != split[0]:
    #             maximal[split[0]].append(item)
    # print(maximal)
    # exit()
    # for key in maximal:
    #     print(key)
        key = split[0]
        key_index = hashtags.loc[hashtags[0] == key].index[0]
        line = user_hashtag_matrix[key_index]
        new_key = key + ", "

        for item in split:
            if item == split[0]:
                continue
            match_index = hashtags.loc[hashtags[0] == item].index[0]
            line += user_hashtag_matrix[match_index]
        user_hashtag_matrix[user_hashtag_matrix.columns[-1]+1] = line

        for item2 in split:
            if item2 == split[0]:
                continue
            new_key += item2
            if item2 != split[-1]:
                new_key += ", "
        df = pd.DataFrame(data={new_key})
        hashtags = hashtags.append(df, ignore_index=True)

if together_constraint:
    together_list = config['Parameters']['together']
    together_list = together_list.strip().split('\n')
    together = {}
    for string in together_list:
        split = string.split(', ')
        together[split[0]] = []
        for item in split:
            if item != split[0]:
                together[split[0]].append(item)
    dropped = []
    not_dropped = []
    for key in together:
        key_index = hashtags.loc[hashtags[0] == key].index[0]
        for match in together[key]:
            match_index = hashtags.loc[hashtags[0] == match].index[0]
            user_hashtag_matrix[key_index] += user_hashtag_matrix[match_index]
            if key_index not in not_dropped:
                not_dropped.append(key_index)
            if match_index not in dropped:
                dropped.append(match_index)
    for saved in not_dropped:
        if saved in dropped:
            dropped.remove(saved)
    dropped.sort()
    hashtags.drop(dropped, inplace=True)
    user_hashtag_matrix.drop(dropped, axis=1, inplace=True)

large_post_values = user_hashtag_matrix.sum(axis=1) >= min_post_count
# Converting the matrix to Boolean states of whether the user used the hashtag or not
user_hashtag_matrix = user_hashtag_matrix >= 1
user_hashtag_matrix = user_hashtag_matrix.astype(int)
# Delete users with less than "min_unique_tags" unique tags used
large_tag_values = user_hashtag_matrix.sum(axis=1) >= min_unique_tags

# Dropping users with small numbers of tag usage
nodes = nodes[large_tag_values & large_post_values]
communities = communities[large_tag_values & large_post_values]
user_hashtag_matrix = user_hashtag_matrix[large_tag_values & large_post_values]

if config['Parameters'].getboolean('weighted_metric'):
    s = 1 / user_hashtag_matrix.sum(axis=0)

# n = number of nodes
# k = number of clusters
# K = 0, 1, 2, ..., k-2, k-1
# t = number of hashtags
# T = 0, 1, 2, ..., t-2, t-1
n = len(nodes)
k = max(communities[communities_tagname])
t = len(hashtags)
T = range(t)
K = range(k)

w = minimum_overlap

# Reindexing the data so that index accessing works later
nodes.reset_index(inplace=True, drop=True)
communities.reset_index(inplace=True, drop=True)
user_hashtag_matrix.reset_index(inplace=True, drop=True)

if together_constraint:
    hashtags.reset_index(inplace=True, drop=True)
    user_hashtag_matrix.columns = np.arange(0, t)
    if config['Parameters'].getboolean('weighted_metric'):
        s.reset_index(inplace=True, drop=True)

# print(communities.where(communities["communities"] == 1)["communities"].sum())
# print(communities.where(communities["communities"] == 2)["communities"].sum()/2)
# print(communities.where(communities["communities"] == 3)["communities"].sum()/3)
# print(communities.where(communities["communities"] == 4)["communities"].sum()/4)
# print(communities.where(communities["communities"] == 5)["communities"].sum()/5)
slice1 = user_hashtag_matrix[communities["communities"] == 1]
slice2 = user_hashtag_matrix[communities["communities"] == 2]
union1 = slice1.sum(axis=0)
union2 = slice2.sum(axis=0)
# print(union1, union2)
setA = (union1 == 0) & (union2 > 0)
tempIndex = 0
unionTagIndexes = []
for item in setA:
    if item:
        unionTagIndexes.append(tempIndex)
        # print(hashtags.iloc[tempIndex][0])
    tempIndex += 1
sample = 0
unionTags = hashtags.iloc[unionTagIndexes]
# for row in user_hashtag_matrix[communities["communities"] == 1].index:
#     for column in range(len(user_hashtag_matrix.iloc[row])):
#         if column not in unionTags.index:
#             user_hashtag_matrix.iloc[row][column] = 0

# Initializing the model
model = Model(sense=MINIMIZE, solver_name=solver_name)

# Initializing binary variables
# There are k x t variables, stored in a matrix of k rows by t columns
x = [[model.add_var(var_type=BINARY) for j in T] for i in K]

if cover_or_forget:
    I = config['Parameters']['number_to_forget_for_each_cluster'].strip().split(', ')
    for item in range(len(I)):
        I[item] = int(I[item])
    N = range(n)
    z = [model.add_var(var_type=BINARY) for i in N]

# objective function: argminx sum[i,j](X[i][j])
if config['Parameters'].getboolean('weighted_metric'):
    model.objective = minimize(xsum(x[i][j] * s[j] for i in K for j in T))
else:
    model.objective = minimize(xsum(x[i][j] for i in K for j in T))

# Set coverage requirement for each different cluster
if cover_or_forget:
    for cluster in K:
        for i in communities[communities[communities_tagname] == cluster + 1].index:
            S = np.zeros((n, t))
            # Sa[i][j] = 1 iff the ith instance is actually in cluster a and has tag j
            for j in T:
                if user_hashtag_matrix.iloc[i][j] == 1:
                    S[i][j] = 1
            model += z[i] + xsum(x[cluster][j] * S[i][j] for j in T) >= 1
    for cluster2 in K:
        model += xsum(z[i] for i in communities[communities['communities'] == cluster2 + 1].index) <= I[cluster2]

else:
    for cluster in K:
        for i in communities[communities[communities_tagname] == cluster + 1].index:
            S = np.zeros((n, t))
            # Sa[i][j] = 1 iff the ith instance is actually in cluster a and has tag j
            for j in T:
                if user_hashtag_matrix.iloc[i][j] == 1:
                    S[i][j] = 1
            model += xsum(x[cluster][j] * S[i][j] for j in T) >= 1

# No cluster overlap
for j in T:
    model += xsum(x[i][j] for i in K) <= w

if apart_constraint:
    apart_list = config['Parameters']['apart']
    apart_list = apart_list.strip().split('\n')
    apart = {}
    for string in apart_list:
        split = string.split(', ')
        apart[split[0]] = []
        for item in split:
            if item != split[0]:
                apart[split[0]].append(item)
    for cluster in K:
        for key in apart:
            key_index = hashtags.loc[hashtags[0] == key].index[0]
            for match in apart[key]:
                match_index = hashtags.loc[hashtags[0] == match].index[0]
                model += x[cluster][key_index] + x[cluster][match_index] <= 1

model.optimize(max_seconds=optimization_runtime)
if write_model_file:
    model.write(model_filename)

sleep(1)

foundClusters = []
if model.status == OptimizationStatus.OPTIMAL or model.status == OptimizationStatus.FEASIBLE:
    print('Tags in each cluster:')
    for v in model.vars:
        if abs(v.x) > 1e-6 and int(v.name[4:-1]) < (k * t): # only printing non-zeros
            if int(v.name[4:-1]) // t + 1 not in foundClusters:
                print("\nCluster", int(v.name[4:-1]) // t + 1)
                foundClusters.append(int(v.name[4:-1]) // t + 1)
            print(hashtags.iloc[(int(v.name[4:-1]) % t)][0])
    if cover_or_forget:
        print('\nForgotten nodes:')
        for v in model.vars:
            if abs(v.x) > 1e-6:  # only printing non-zeros
                if int(v.name[4:-1]) >= (k * t):
                    print(nodes.iloc[(int(v.name[4:-1]) - (k * t))][0])
