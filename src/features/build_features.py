
# coding: utf-8

# # Correlate timecourses with templates  
# 
# Use the z-scored timeseries to correlate one patient's individual volumes with the average activity maps of the other patients.

# ### import modules

# In[2]:


import os
import pickle

import numpy as np
import pandas as pd

from nilearn import input_data, plotting, image

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


sns.set_context('poster')


# In[ ]:


# after converstion to .py, we can use __file__ to get the module folder
try:
    thisDir = os.path.realpath(__file__)
# in notebook form, we take the current working directory (we need to be in 'notebooks/' for this!)
except:
    thisDir = '.'
# convert relative path into absolute path, so this will work with notebooks and py modules
supDir = os.path.abspath(os.path.join(os.path.dirname(thisDir), '..'))

supDir


# ### get data
#
## In[6]:
#
#
#data_df = pd.read_csv(
#    '../data/interim/csv/info_epi_zscored_zdiff_summarymaps_2dpredclean_df.csv',
#    index_col=[0],
#    header=0)
#
#data_df.index = [data_df.loc[:, 'wada'], data_df.index]
#data_df = data_df.sort_index()
#
#
#
## In[7]:
#
#
#data_df.shape
#
#
#
## In[8]:
#
#
#data_df[::20]
#
#

# ### masker

# #### whole brain masker

# In[13]:


mask_file = os.path.join(supDir,'data','external','MNI152_T1_2mm_brain_mask.nii.gz')
whole_brain_masker = input_data.NiftiMasker(mask_file).fit()

#
## In[14]:
#
#
#plotting.plot_roi(whole_brain_masker.mask_img_)
#
#

# #### template image from our previous work (independent patient sample)

# In[15]:


template_im = os.path.join(supDir,'models','tMap_diff_left.nii.gz')

#
## In[16]:
#
#
#plotting.plot_stat_map(template_im,title=template_im)
#
#

# ### make mask with all positive values

# In[17]:


data = whole_brain_masker.transform(template_im)


# In[18]:


my_data = data>0


# In[19]:


template_mask = whole_brain_masker.inverse_transform(my_data)
masker = input_data.NiftiMasker(template_mask).fit()

#
## In[20]:
#
#
#plotting.plot_roi(masker.mask_img_)
#
#

# ### get data from mask

# In[21]:


template_df = pd.DataFrame(masker.transform(template_im),index=['template'])
template_df


# ### get data from target patient
# 
# We get every single volume
#
## In[22]:
#
#
#def make_timecourse_df(this_idx, col_name, data_df=data_df, masker=masker):
#    """extract data from 4d-image. The image is taken from the data_df table"""
#
#    # select the image using row and column indexing
#    this_im = data_df.loc[this_idx, col_name]
#    # extract whole-brain data
#    this_data = masker.transform(this_im)
#    # transform into df with default numbering of rows and columns
#    this_df = pd.DataFrame(this_data)
#
#    return this_df
#
#
#
## In[23]:
#
#
#p_name = data_df.index[0]
#
#
#
## In[24]:
#
#
#this_df = make_timecourse_df(p_name, 'z-scored-diff')
#
#
#
## In[25]:
#
#
#this_df.tail()
#
#

# #### toolbox version


def make_timecourse_df(pFolder,pName, masker=masker):

    this_im = os.path.join(pFolder, ''.join(['z_imDiff_', pName, '.nii.gz']))
    this_data = masker.transform(this_im)
    this_df = pd.DataFrame(this_data)

    return this_df



# ### correlate each individual volume of a patient with the template

# In[26]:


def make_corrs(this_df, template_df):
    """correlate each volume of target patient with template"""

    d = {}

    for vol in this_df.index:

        this_vol = this_df.loc[vol, :]
        d[vol] = {}
        all_corrs = np.corrcoef(this_vol, template_df)
        my_corrs = all_corrs[0, 1:]
        for n, i in enumerate(template_df.index):
            d[vol][i] = my_corrs[n]

    df = pd.DataFrame(d)

    return df

#
## In[27]:
#
#
#from datetime import datetime
#
#
#
## In[28]:
#
#
#print('%s'%datetime.now())
#corr_df = make_corrs(this_df, template_df)
#print('%s'%datetime.now())
#
#
#
## In[29]:
#
#
#corr_df
#
#
#
## In[3]:
#
#
#with open('../models/colors.p', 'rb') as f:
#    color_dict = pickle.load(f)
#
#
#
## In[4]:
#
#
##### selection of main colors  
#my_cols = {}
#for i, j in zip(['red', 'blue', 'yellow'], ['left', 'right', 'bilateral']):
#    my_cols[j] = color_dict[i]
#
#my_cols
#
#

# ### make individual plot
#
## In[29]:
#
#
#with open('../models/conds.p', 'rb') as f:
#    conds = pickle.load(f)
#
#
#
## In[30]:
#
#
#def make_plot(df, ax, color_dict=color_dict, conds=conds):
#    """show correlations of single patient's volumes with groups"""
#
#    # find the volumes when activity starts (CAVE: make sure that blocks
#    # are 10 volumes long, because this is hard-coded here)
#    act_blocks = np.where(np.array(conds) == 1)[-1][::10]
#
#    # draw each block with height -1 to +1 (needs to be trimmed to actual
#    # data range later, otherwise too large)
#    for i in act_blocks:
#        ax.fill_between([i, i + 10], [-1, -1], [1, 1],
#                        color=color_dict['trans'],
#                        alpha=0.7)
#
#    # the correlations with each group are in rows, we loop through each group/row
#    for i in df.index:
#        ax.plot(
#            df.loc[i, :], label=i, alpha=0.8, linewidth=4, color=color_dict['blue'])
#
#    # dotted line indicates zero/no correlation
#    ax.axhline(0, linestyle=':', color=color_dict['black'], linewidth=3)
#
#    # define range on y-axis by rounded-up max value
#    my_max = (df.max().max() + 0.05).round(1)
#    ax.set_ylim(-my_max, my_max)
#    ax.set_yticks(np.linspace(-my_max, my_max, 5))
#
#    ax.set_xlim(0, 200)
#    ax.set_xticks(np.arange(0, 201, 20))
#
#    #ax.legend(loc='best', fontsize=20)
#
#    ax.set_xlabel('Volume/TR')
#    ax.set_ylabel('Pearson Correlation')
#
#    sns.despine(trim=True, offset=10)
#
#    return ax
#
#

# ### do everything for one patient
#
## In[35]:
#
#
#def make_p(p_name):
#    """everything for one patient"""
#
#    this_df = make_timecourse_df(p_name, 'z-scored-diff')
#    corr_df = make_corrs(this_df, template_df)
#
#    return corr_df
#
#
#
## In[36]:
#
#
#print('%s'%datetime.now())
#p_df = make_p(p_name)
#print('%s'%datetime.now())
#
#
#
## In[37]:
#
#
#p_df
#
#

# #### toolbox version


def make_p(pFolder,pName,template_df=template_df,masker=masker):

    this_df = make_timecourse_df(pFolder, pName, masker=masker)
    corr_df = make_corrs(this_df, template_df)
    out_name = os.path.join(pFolder, ''.join([ pName, '_corr_df.csv']))
    corr_df.to_csv(out_name)
    
    return out_name



# ### show individual results
#
## In[34]:
#
#
#fig,ax = plt.subplots(1,1,figsize=(16, 6))
#ax = make_plot(p_df, ax=ax)
#plt.tight_layout()
#plt.show()
#
#

# ### do this for all patients
#
## In[39]:
#
#
##for p_name in tqdm(data_df.index):
#for p_name in data_df.index:
#    print('%s %s'%(p_name,datetime.now()))
#    
#    p_df = make_p(p_name)
#    out_name = '../data/interim/csv/%s_corr_df.csv' % p_name[-1]
#    p_df.to_csv(out_name)
#    data_df.loc[p_name,'corr_df'] = out_name
#
#data_df.to_csv('../data/interim/csv/info_epi_zscored_zdiff_summarymaps_2dpredclean_corr_df.csv')
#
#

# ### summary
# 
# We now have a table for each patient, where the correlations of each of the 200 volumes with the template are stored.
# 
# 
# **************
# 
# < [Previous](08-mw-methods-plot-correlations.ipynb) | [Contents](00-mw-overview-notebook.ipynb) | [Next >](10-mw-train-test-classifier.ipynb)
