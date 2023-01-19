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
    #run louvain algorithm to get optimal partitions 
    #mod=bct.modularity_louvain_und(np_SC, gamma=1, hierarchy=False, seed=None)
    # append the modularity value (last value of mod) of partition to modularity list
    #modularity.append(mod[-1])
    subj.append(filename)
    degree.append(bct.strengths_und(np_SC))
    # append global efficiency value
    #global_efficiency.append(bct.efficiency_wei(np_SC,local=False))
    # append transitivity value
    #transitivity.append(bct.transitivity_wu(np_SC))
    # append Assortativity value
    #assortativity.append(bct.assortativity_wei(np_SC))
    # append Rich Club Coefficient value
    #r=bct.rich_club_wu(np_SC)
    #rich_club.append(np.nanmean(r))
    # average clustering coefficient
    #f=bct.clustering_coef_wu(np_SC)
    #cluster_coef.append(np.average(f))
    
    


# In[6]:


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
edge_weight=[]
subj=[]
# Go to directory with csv files
os.chdir("C:/Users/ddcc2/Documents/connectome_aal")
# run through loop for each csv file (wont run while other csv files are in the folder - need to delete or remove)
for filename in os.listdir("C:/Users/ddcc2/Documents/connectome_aal/"):
    print(filename)
    # read csv file into panda dataframe
    #SC = pd.read_csv(filename,delimiter=' ', header=None)
    # create numpy array
    #subj.append(filename)
    #np_SC=SC.to_numpy()
    #edge_weight.append(np_SC[0,1])


# In[26]:


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
os.chdir("C:/Users/ddcc2/Documents/connectome_aal")

SC = pd.read_csv("C:/Users/ddcc2/Documents/connectome_aal/test.csv",delimiter=',', header=None)
np_SC=SC.to_numpy()
print(np_SC)
bct.transitivity_wu(np_SC)


# In[26]:


bct.degrees_und(np_SC)


# In[27]:


# Node strength
degree_weights=[]
degree_weights=pd.DataFrame(degree,subj)
print(degree_weights)
degree_weights.to_csv('strength_weights.csv')


# In[15]:


print(np_SC[0,1])
print(np_SC.shape)
data_frame_edge_weight_01=[]
data_frame_edge_weight_01=pd.DataFrame(edge_weight,subj)
print(data_frame_edge_weight_01)


# In[43]:


#Save transitivity results as csv
data_frame_trans=pd.DataFrame(transitivity,subj)
data_frame_trans.to_csv('transitivity.csv')
data_frame_trans


# In[44]:


#Save rich club results as csv
data_frame_mod=pd.DataFrame(rich_club,subj)
data_frame_mod.to_csv('rich_club.csv')
data_frame_mod


# In[45]:


#Save assortativity results as csv
data_frame_mod=pd.DataFrame(assortativity,subj)
data_frame_mod.to_csv('assortativity.csv')
data_frame_mod


# In[46]:


#Save cluster results as csv
data_frame_mod=pd.DataFrame(cluster_coef,subj)
data_frame_mod.to_csv('results_cluster_coef.csv')
data_frame_mod


# In[47]:


#Save modularity results as csv
data_frame_mod=pd.DataFrame(modularity,subj)
data_frame_mod.to_csv('results_modularity.csv')
data_frame_mod


# In[48]:


#Save global efficiency results as csv
data_frame_global=pd.DataFrame(global_efficiency,subj)
data_frame_global.to_csv('global_efficiency.csv')
data_frame_global


# In[1]:


SC = pd.read_csv('HCA6002236_V1_MR_connectome.csv',delimiter=' ', header=None)
print(SC.values.sum())
SC_threshold=SC.clip(lower=1000)
SC_threshold=SC_threshold.replace(to_replace=1000,value=0)
# create numpy array
np_SC_threshold=SC_threshold.to_numpy()
np_SC=SC.to_numpy()
print(SC_threshold)
sns.set(rc={'figure.figsize':(15,15)})
G=nx.from_numpy_array(np_SC_threshold)
pos=nx.circular_layout(G)

# display the network using previously determined node positions
nx.draw(G, 
    pos=pos,
    with_labels=True)
#labels = nx.get_edge_attributes(G,'weight')
#print(labels)
#nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

plt.show()


# In[ ]:


import networkx as nx
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import nilearn
import bct
import seaborn as sns

global_efficiency=[]
local_efficiency=[]
transitivity=[]
modularity=[]
subj=[]
modularity_partition=[]
# Go to directory with csv files
os.chdir("C:/Users/ddcc2/Documents/connect_aal/")
SC = pd.read_csv("HCA6010538_V1_MR_connectome.csv",delimiter=' ', header=None)
np_SC=SC.to_numpy()

#print(mod)
#print(SC)
#print(np_SC)

print(bct.efficiency_wei(np_SC))

# Calculate clustering coefficient
f=bct.clustering_coef_wu(np_SC)
np.average(f)
np.sum(np_SC)

print(bct.rich_club_wu(np_SC))


# In[33]:





# In[34]:





# In[4]:


# Graph origional panda dataframe
sns.set(rc={'figure.figsize':(11.7,8.27)})
correlation_mat=SC.corr()
corr_graph=sns.heatmap(correlation_mat,robust=True,cmap='coolwarm')
corr_graph.invert_yaxis()
corr_graph
plt.show()


# In[19]:


SC


# In[8]:


correlation_mat


# In[11]:


print(clustering_coef[46])
print(clustering_coef[39])
#SC


# In[15]:




