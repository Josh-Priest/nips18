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

config_filename = 'chess/configChess.ini'
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



# Reindexing the data so that index accessing works later
nodes.reset_index(inplace=True, drop=True)
communities.reset_index(inplace=True, drop=True)
user_hashtag_matrix.reset_index(inplace=True, drop=True)

if together_constraint:
    hashtags.reset_index(inplace=True, drop=True)
    user_hashtag_matrix.columns = np.arange(0, t)
    if config['Parameters'].getboolean('weighted_metric'):
        s.reset_index(inplace=True, drop=True)


# TODO: Begin ANTI-COVER
slice1 = user_hashtag_matrix[communities["communities"] == 1]
slice2 = user_hashtag_matrix[communities["communities"] == 2]
union1 = slice1.sum(axis=0)
union2 = slice2.sum(axis=0)
# print(union1, union2)
setA = (union1 >= 10) & (union2 < 10)
tempIndex = 0
unionTagIndexes = []
for item in setA:
    if item:
        unionTagIndexes.append(tempIndex)
        # print(hashtags.iloc[tempIndex][0])
    tempIndex += 1
sample = 0
unionTags = hashtags.iloc[unionTagIndexes]

print(unionTags)
exit()
user_hashtag_matrix = user_hashtag_matrix[unionTags.index]
user_hashtag_matrix = user_hashtag_matrix[communities.communities != 2]

nodes = nodes[communities.communities != 2]
nodes.reset_index(inplace=True, drop=True)
communities = communities[communities.communities != 2]
communities.reset_index(inplace=True, drop=True)
hashtags = hashtags.iloc[unionTags.index]
hashtags.reset_index(inplace=True, drop=True)

user_hashtag_matrix.reset_index(inplace=True, drop=True)
user_hashtag_matrix.columns = np.arange(len(hashtags))

# Initializing the model
model = Model(sense=MINIMIZE, solver_name=solver_name)


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
r = len(hashtags)
R = range(r)
j = len(nodes)
J = range(j)
# Initializing binary variables
# There are k x t variables, stored in a matrix of k rows by t columns
x = [model.add_var(var_type=BINARY) for j in R]

model.objective = minimize(xsum(x[i] for i in R))

nodes_forgotten = 0
for a in J:
    tagset = user_hashtag_matrix.iloc[a][user_hashtag_matrix.iloc[a] == 1]
    if len(tagset.index) == 0:
        nodes_forgotten += 1
    else:
        model += xsum(x[i] for i in tagset.index) >= 1


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

print("Nodes Forgotten: ", nodes_forgotten)
