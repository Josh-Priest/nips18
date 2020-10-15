"""
***********************************************************
(c) Ian Davidson and Davidson Lab. Please cite

The Cluster Description Problem-Complexity Results, Formulations and Approximations, I Davidson, A Gourru and SS. Ravi
Advances in Neural Information Processing Systems, 2018, 6191-6201
***********************************************************
"""

import numpy as np
import pandas as pd
import sys
import configparser
from mlxtend.frequent_patterns import fpgrowth, fpmax

pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None)


config_file = open("config.ini", "r")


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


# large_post_values = user_hashtag_matrix.sum(axis=1) >= min_post_count
# # Converting the matrix to Boolean states of whether the user used the hashtag or not
# user_hashtag_matrix = user_hashtag_matrix.astype(int)
# # Delete users with less than "min_unique_tags" unique tags used
# large_tag_values = user_hashtag_matrix.sum(axis=1) >= min_unique_tags

# Dropping users with small numbers of tag usage
# nodes = nodes[large_tag_values & large_post_values]
# communities = communities[large_tag_values & large_post_values]
# user_hashtag_matrix = user_hashtag_matrix[large_tag_values & large_post_values]

# n = number of nodes
# k = number of clusters
# K = 0, 1, 2, ..., k-2, k-1
# t = number of hashtags
# T = 0, 1, 2, ..., t-2, t-1
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
    for key in together:
        key_index = hashtags.loc[hashtags[0] == key].index[0]
        for match in together[key]:
            match_index = hashtags.loc[hashtags[0] == match].index[0]
            user_hashtag_matrix[key_index] += user_hashtag_matrix[match_index]
            dropped.append(match_index)
    hashtags.drop(dropped, inplace=True)
    user_hashtag_matrix.drop(dropped, axis=1, inplace=True)

cluster = 5
user_hashtag_matrix = user_hashtag_matrix.where(communities["communities"] == cluster)
user_hashtag_matrix = user_hashtag_matrix.dropna()

nodes = nodes.where(communities["communities"] == cluster)
nodes = nodes.dropna()

communities = communities.where(communities["communities"] == cluster)
communities = communities.dropna()
communities[communities_tagname] = communities[communities_tagname].astype(int)


# Reindexing the data so that index accessing works later
user_hashtag_matrix = user_hashtag_matrix >= 1

nodes.reset_index(inplace=True, drop=True)
communities.reset_index(inplace=True, drop=True)
user_hashtag_matrix.reset_index(inplace=True, drop=True)

t = len(hashtags)
n = len(nodes)
k = max(communities[communities_tagname])
t = len(hashtags)
T = range(t)
K = range(k)

w = minimum_overlap

# dropcol = []
# maxtagusage = .7
# colsums = user_hashtag_matrix.sum(axis=0)
#
# for item in colsums.index:
#     if colsums[item]/n > maxtagusage:
#         dropcol.append(item)
#
# print(dropcol)
#
# user_hashtag_matrix.drop(dropcol, axis=1, inplace=True)
# hashtags.drop(dropcol, inplace=True)

nodes.reset_index(inplace=True, drop=True)
communities.reset_index(inplace=True, drop=True)
user_hashtag_matrix.reset_index(inplace=True, drop=True)

t = len(hashtags)
n = len(nodes)
k = max(communities[communities_tagname])
t = len(hashtags)
T = range(t)
K = range(k)



if together_constraint:
    hashtags.reset_index(inplace=True, drop=True)
    user_hashtag_matrix.columns = np.arange(0, t)



tags = []
for item in hashtags[0]:
    tags.append(item)
user_hashtag_matrix.columns = tags


# data = fpmax(user_hashtag_matrix, min_support=0.2941, use_colnames=True)
data = fpmax(user_hashtag_matrix, min_support=0.489, use_colnames=True)
name = "maximal " + str(cluster)
file = open(name, "w")
print(data, file=file)
# print(fpmax(user_hashtag_matrix, min_support=0.3, use_colnames=True))
# print(fpmax(user_hashtag_matrix, min_support=0.25, use_colnames=True))
# print(fpmax(user_hashtag_matrix, min_support=0.2, use_colnames=True))


