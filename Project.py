#!/usr/bin/env python
# coding: utf-8

# In[22]:


#!pip install bctpy
#!pip install nilearn
import networkx as nx
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import nilearn
import bct
import seaborn as sns

# Create empty lists
global_efficiency=[]
cluster_coef=[]
transitivity=[]
rich_club=[]
assortativity=[]
modularity=[]
subj=[]
degree=[]
modularity_partition=[]
# Go to directory with csv files
os.chdir("C:/Users/ddcc2/Documents/connectome_aal")
# run through loop for each csv file (wont run while other csv files are in the folder - need to delete or remove)
for filename in os.listdir("C:/Users/ddcc2/Documents/connectome_aal/"):
    # read csv file into panda dataframe
    SC = pd.read_csv(filename,delimiter=' ', header=None)
    # create numpy array
    np_SC=SC.to_numpy()
    #Compute strength of all 90 AAL nodes for supplementary analysis
    subj.append(filename)
    degree.append(bct.strengths_und(np_SC))
    # append global efficiency value
    global_efficiency.append(bct.efficiency_wei(np_SC,local=False))
    # append transitivity value
    transitivity.append(bct.transitivity_wu(np_SC))
   
    
    



# Node strength
degree_weights=[]
degree_weights=pd.DataFrame(degree,subj)
print(degree_weights)
degree_weights.to_csv('strength_weights.csv')


#Save transitivity results as csv
data_frame_trans=pd.DataFrame(transitivity,subj)
data_frame_trans.to_csv('transitivity.csv')
data_frame_trans


#Save global efficiency results as csv
data_frame_global=pd.DataFrame(global_efficiency,subj)
data_frame_global.to_csv('global_efficiency.csv')
data_frame_global




