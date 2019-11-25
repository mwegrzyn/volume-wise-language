
# coding: utf-8

# # Make correlation plots over time
# 
# The present notebook again plots volume-wise correlations. Unlike the preceding notebook, this is now done on a group level, which also means that we have multiple observations for each volume and can include a measure of uncertainty into our plots.

# ### import modules

# In[1]:


import os
import pickle

import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('poster')
sns.set_style('ticks')


# In[2]:


# after converstion to .py, we can use __file__ to get the module folder
try:
    thisDir = os.path.realpath(__file__)
# in notebook form, we take the current working directory (we need to be in 'notebooks/' for this!)
except:
    thisDir = '.'
# convert relative path into absolute path, so this will work with notebooks and py modules
supDir = os.path.abspath(os.path.join(os.path.dirname(thisDir), '..'))

supDir


# ### get meta df

# We need this e.g. to get information about conclusiveness
#
## In[3]:
#
#
#data_df = pd.read_csv(
#    '../data/processed/csv/info_epi_zscored_zdiff_summarymaps_2dpredclean_corr_counts_df.csv',
#    index_col=[0, 1],
#    header=0)
#
#
#
## In[4]:
#
#
#data_df.tail()
#
#

# #### conclusiveness filters
#
## In[5]:
#
#
#is_conclusive = data_df.loc[:, 'pred'] != 'inconclusive'
#
#
#
## In[6]:
#
#
#is_conclusive.sum()
#
#

# ### get data
#
## In[7]:
#
#
#def make_group_df(data_df,metric='corr_df'):
#    '''load correlation data of all patients'''
#    
#    group_df = pd.DataFrame()
#    
#    for p in data_df.index:
#        # get data
#        filename = data_df.loc[p, metric]
#        this_df = pd.read_csv(filename, index_col=[0], header=0)
#        # add patient infos to index
#        this_df.index = pd.MultiIndex.from_tuples([p])
#        
#        group_df = pd.concat([group_df, this_df])
#
#    # reorder the colums and make sure volumes are integer values
#    group_df.columns = group_df.columns.astype(int)
#
#    # sort across rows, then across columns, to make sure that volumes
#    # are in the right order
#    group_df = group_df.sort_index(axis=0)
#    group_df = group_df.sort_index(axis=1)
#
#    # check if everything is in its right place
#    assert all(group_df.columns == range(200)), 'wrong order of volumes'
#            
#    return group_df
#
#
#
## In[8]:
#
#
#group_df = make_group_df(data_df)
#
#
#
## In[9]:
#
#
#group_df.tail()
#
#

# #### filter data
#
## In[10]:
#
#
## only conclusive cases
#conclusive_df = group_df[is_conclusive]
## only inconclusive cases
#inconclusive_df = group_df[is_conclusive == False]
## all cases unfiltered
#withinconclusive_df = group_df.copy()
#
#
#
## In[11]:
#
#
#print(conclusive_df.shape, inconclusive_df.shape, withinconclusive_df.shape)
#
#

# ### get design

# In[12]:


conds_file = os.path.join(supDir,'models','conds.p')
with open(conds_file, 'rb') as f:
    conds = pickle.load(f)

#
## In[13]:
#
#
#print(conds)
#
#

# ### get colors

# In[14]:


colors_file = os.path.join(supDir,'models','colors.p')
with open(colors_file, 'rb') as f:
    color_dict = pickle.load(f)


# In[15]:


my_cols = {}
for i, j in zip(['red', 'blue', 'yellow'], ['left', 'right', 'bilateral']):
    my_cols[j] = color_dict[i]

#
## In[16]:
#
#
#sns.palplot([my_cols[k] for k in my_cols])
#
#

# ### plot corrs
#
## In[17]:
#
#
#def make_plot(df, ax, my_cols=my_cols, color_dict=color_dict):
#    '''group-levels volume-wise correlations, with mean and confidence interval'''
#
#    # find the volumes when activity starts (cave: blocks have to be 10
#    # volume long, as is hard-coded herewith
#    act_blocks = np.where(np.array(conds) == 1)[-1][::10]
#
#    # draw each block with height -1 to +1 (needs to be trimmed to actual
#    # data range later, otherwise too large)
#    for i in act_blocks:
#        ax.fill_between([i, i + 10], [-1, -1], [1, 1],
#                        color=color_dict['trans'],
#                        alpha=0.7)
#
#    mean_df = df.groupby(level=0).mean()
#
#    n = df.shape[0]
#    std_df = df.groupby(level=0).std()
#    se_df = (std_df / np.sqrt(n))
#    ci_df = se_df * 1.96    
#    
#    for c in mean_df.index:
#        
#        this_mean = mean_df.loc[c,:]
#        this_ci = ci_df.loc[c,:]
#        
#        try:
#            ax.plot(this_mean, '-',color=my_cols[c], label=c, alpha=0.7, linewidth=4)
#
#            ax.fill_between(
#                this_ci.index,
#                this_mean - this_ci,
#                this_mean + this_ci,
#                color=my_cols[c],
#                alpha=0.3)
#        except:
#            ax.plot(this_mean, '-',color=color_dict['black'], alpha=0.7, linewidth=4)
#    
#    # dotted line indicates zero/no correlation
#    ax.axhline(0, linestyle=':', color=color_dict['black'], linewidth=3)
#
#    ax.set_xlim(0, 200)
#    ax.set_xticks(np.arange(0, 201, 20))
#
#    ax.set_ylim(-0.1,0.1)
#    #ax.legend(loc='best', fontsize=20)
#
#    ax.set_xlabel('volume')
#    ax.set_ylabel('correlation')
#
#    sns.despine(trim=True, offset=10)
#
#    return ax
#
#
#
## In[18]:
#
#
#my_labels = ['left', 'bilateral', 'right']
#
#
#
## In[19]:
#
#
#fig = plt.figure(figsize=(16, 4))
#
#ax = plt.subplot(111)
#ax = make_plot(conclusive_df, ax)
#
##plt.legend(loc=(1,0.7))
#plt.savefig('../reports/figures/12-timecourse200.png',dpi=300,bbox_inches='tight')
#plt.show()
#
#

# ### for inconclusive cases
#
## In[20]:
#
#
#fig = plt.figure(figsize=(16,4))
#
#ax = plt.subplot(111)
#ax = make_plot(inconclusive_df, ax)
#
#plt.legend(loc=(1,0.7))
#plt.show()
#
#

# ### average over one cycle

# #### design for a 20-TR cycle

# In[21]:


trs = np.concatenate([np.array([-999,-999]), np.array(list(range(0,20)) * 10)])[:200]

#
## In[22]:
#
#
#print(trs)
#
#
#
## In[23]:
#
#
#def make_tr_plot(df, trs=trs, my_cols=my_cols, color_dict=color_dict,ax=ax):
#    '''make group-level volume-wise plot with mean and CI, but averaged across one
#    cycle of rest and activity
#    '''
#    
#    ax.fill_between([9.5, 9.5 + 10], [-1, -1], [1, 1],
#                    color=color_dict['trans'],
#                    alpha=0.7)
#    
#    ax.axhline(0, linewidth=3, linestyle=':', c='k')
#        
#    for i, g in enumerate(['left', 'bilateral', 'right']):
#
#            this_df = df.loc[g,:].T
#
#            # the two volumes are before the experiment started so to speak
#            # because of the HRF-delay,so we drop them
#            this_df = this_df.iloc[2:, :]
#            this_df.index = [trs[2:], this_df.index]
#
#            this_tr_df = this_df.groupby(level=0).mean().T
#
#            x = this_tr_df.columns
#            y = this_tr_df.mean()
#            y_std = this_tr_df.std()
#            n = this_tr_df.shape[0]
#            y_se = y_std / np.sqrt(n)
#            y_ci = y_se * 1.96
#
#            ax.plot(x, y, '-o', c=my_cols[g], label=g)
#            ax.fill_between(
#                x, y - y_ci, y + y_ci, alpha=0.4, color=my_cols[g])
#
#    ax.set_xlabel('volume')
#    ax.set_ylabel('correlation')
#
#    ax.set_xlim(-0.5, 19.5)
#    ax.set_xticks(np.arange(0, 21, 2))
#    ax.set_xticklabels(np.arange(2, 23, 2))
#
#    ax.set_ylim(-0.1,0.1)
#
#    sns.despine(trim=True,offset=5)
#
#    ax.legend(loc='best')
#
#    return ax
#
#
#
## In[24]:
#
#
#fig = plt.figure(figsize=(9,6))
#
#ax = plt.subplot(111)
#ax = make_tr_plot(conclusive_df,ax=ax)
#
#plt.legend(loc=(1.1,0.35))
#plt.savefig('../reports/figures/12-timecourse20.png',dpi=300,bbox_inches='tight')
#plt.show()
#
#

# ### combine
#
## In[25]:
#
#
#sns.set_style('dark')
#
#
#
## In[26]:
#
#
#fig = plt.figure(figsize=(16, 12))
#
#ax1 = fig.add_axes([0,1,1,1], xticklabels=[], yticklabels=[])
#ax1.imshow(Image.open('../reports/figures/12-timecourse200.png'))
#
#ax2 = fig.add_axes([0.2,0.5,.8,1], xticklabels=[], yticklabels=[])
#ax2.imshow(Image.open('../reports/figures/12-timecourse20.png'))
#
#plt.text(-0.01,1.02, 'A',transform=ax1.transAxes, fontsize=32)
#plt.text(0.19,-0.08, 'B',transform=ax1.transAxes, fontsize=32)
#
#plt.savefig('../reports/figures/12-all-timecourses.png',dpi=300,bbox_inches='tight')
#plt.show()
#
#
#
## In[27]:
#
#
#sns.set_style('ticks')
#
#

# ### show the correlations when rest is inverted
#
## In[28]:
#
#
#inv_df = conclusive_df*conds
#
#
#
## In[29]:
#
#
#inv_df.tail()
#
#
#
## In[30]:
#
#
#fig = plt.figure(figsize=(16, 4))
#
#ax = plt.subplot(111)
#ax = make_plot(inv_df, ax)
#
#plt.legend(loc=(1,0.7))
#plt.show()
#
#
#
## In[31]:
#
#
#fig = plt.figure(figsize=(9,6))
#
#ax = plt.subplot(111)
#ax = make_tr_plot(inv_df,ax=ax)
#
#plt.legend(loc=(1.1,0.35))
#plt.show()
#
#

# ## For individual patients

# ### get data
#
## In[32]:
#
#
#p_name = 'patID'
#
#
#
## In[33]:
#
#
#p_df = pd.read_csv('../data/interim/csv/%s_corr_df.csv'%p_name,index_col=[0],header=0)
#p_df
#
#
#
## In[34]:
#
#
#p_inv_df = p_df*conds
#p_inv_df
#
#

# ### get classifier

# In[35]:


clf_file = os.path.join(supDir,'models','volume_clf.p')
with open(clf_file,'rb') as f:
    d = pickle.load(f)


# In[36]:


clf = d['clf']
my_scaler = d['scaler']
my_labeler = d['labeler']


# ### predict label of each volume

# In[37]:


def make_preds(this_df,clf,my_scaler,my_labeler):
    
    scaled_features = my_scaler.transform(this_df.T)
    predictions = clf.predict(scaled_features)
    labeled_predictions = my_labeler.inverse_transform(predictions)
    
    return labeled_predictions

#
## In[38]:
#
#
#labeled_predictions = make_preds(p_inv_df,clf,my_scaler,my_labeler)
#
#

# ### plot results

# In[39]:


def make_single_plot(p_df,predictions, ax, my_cols=my_cols, color_dict=color_dict):
    '''group-levels volume-wise correlations, with mean and confidence interval'''

    # find the volumes when activity starts (cave: blocks have to be 10
    # volume long, as is hard-coded herewith
    act_blocks = np.where(np.array(conds) == 1)[-1][::10]

    # draw each block with height -1 to +1 (needs to be trimmed to actual
    # data range later, otherwise too large)
    for i in act_blocks:
        ax.fill_between([i, i + 10], [-1, -1], [1, 1],
                        color=color_dict['trans'],
                        alpha=0.7)

    y_vals = p_df.values[-1]
    ax.plot(y_vals, '-',color=color_dict['black'], linewidth=2)
    
    for x_val,y_val,y_pred in zip(p_df.columns,y_vals,predictions):
        ax.plot(x_val,y_val,'o',color=my_cols[y_pred],markersize=9)
    
    # dotted line indicates zero/no correlation
    ax.axhline(0, linestyle=':', color=color_dict['black'], linewidth=3)
    
    ax.set_xlim(0, 200)
    ax.set_xticks(np.arange(0, 201, 20))
    ax.set_xticklabels(np.arange(0, 201, 20))

    my_max = abs(y_vals).max()
    ax.set_ylim(-my_max*1.1,+my_max*1.1)

    ax.set_xlabel('volume')
    ax.set_ylabel('correlation')

    sns.despine(trim=True, offset=10)

    return ax

#
## In[40]:
#
#
#fig,ax = plt.subplots(1,1,figsize=(16,4))
#ax = make_single_plot(p_df,labeled_predictions,ax=ax)
#sns.despine()
#plt.savefig('../examples/%s_timeAll200trs.png'%p_name,dpi=300,bbox_inches='tight')
#plt.show()
#
#

# In[41]:


def make_single_tr_plot(df, ax, trs=trs, my_cols=my_cols, color_dict=color_dict):
    
    ax.fill_between([9.5, 9.5 + 10], [-1, -1], [1, 1],
                    color=color_dict['trans'],
                    alpha=0.7)
    
    ax.axhline(0, linewidth=3, linestyle=':', c='k')

    this_df = df.copy().T

    # the two volumes are before the experiment started so to speak
    # because of the HRF-delay,so we drop them
    this_df = this_df.iloc[2:, :]
    this_df.index = [trs[2:], this_df.index]

    # make spaghetti plot
    copy_df = this_df.copy()
    bins = np.array([[x]*20 for x in range(10)]).ravel()[:198]
    copy_df.index = [bins,copy_df.index.get_level_values(0)]
    for b in copy_df.index.levels[0]:
        ax.plot(copy_df.loc[b],color='grey',alpha=0.5)
    
    # make area plot
    this_tr_mean = this_df.groupby(level=0).mean()
    this_tr_std = this_df.groupby(level=0).std()
    ax.plot(this_tr_mean.index,this_tr_mean.values, '-', c=color_dict['black'])
    
    this_inv_df = (df*conds).copy().T
    this_inv_df = this_inv_df.iloc[2:, :]
    this_inv_df.index = [trs[2:], this_inv_df.index]
    this_inv_tr_mean = this_inv_df.groupby(level=0).mean().T.values
    labeled_predictions = make_preds(this_inv_tr_mean,clf,my_scaler,my_labeler)

    for x_val,y_val,y_pred in zip(this_tr_mean.index,this_tr_mean.values,labeled_predictions):
        plt.plot(x_val,y_val,'o',color=my_cols[y_pred],markersize=12)
    
    lower_std_bound = (this_tr_mean - this_tr_std).T.values[-1]
    upper_std_bound = (this_tr_mean + this_tr_std).T.values[-1]
    
    ax.fill_between(
        this_tr_mean.index,
        lower_std_bound,
        upper_std_bound,
        alpha=0.1, 
        color=color_dict['black'])

    ax.set_xlabel('volume')
    ax.set_ylabel('correlation')

    ax.set_xlim(-0.5, 19.5)
    ax.set_xticks(np.arange(0, 21, 2))
    ax.set_xticklabels(np.arange(2, 23, 2))
    
    ax.set_ylim(lower_std_bound.min()*1.4,upper_std_bound.max()*1.4)
    sns.despine(trim=True, offset=10)

    return ax

#
## In[42]:
#
#
#fig,ax = plt.subplots(1,1,figsize=(10,6))
#ax = make_single_tr_plot(p_df,ax)
#sns.despine()
#plt.savefig('../examples/%s_timeCycle20trs.png'%p_name,dpi=300,bbox_inches='tight')
#plt.show()
#
#

# ### another patient, another plot
#
## In[43]:
#
#
#p_df = pd.read_csv('../data/interim/csv/pat083_corr_df.csv',index_col=[0],header=0)
#p_inv_df = p_df*conds
#labeled_predictions = make_preds(p_inv_df,clf,my_scaler,my_labeler)
#
#fig,ax = plt.subplots(1,1,figsize=(16,4))
#ax = make_single_plot(p_df,labeled_predictions,ax=ax)
#sns.despine()
#plt.show()
#
#
#
## In[44]:
#
#
#fig,ax = plt.subplots(1,1,figsize=(10,6))
#ax = make_single_tr_plot(p_df,ax=ax)
#sns.despine()
#plt.show()
#
#

# #### toolbox use


def make_p(pFolder,pName,clf=clf,my_scaler=my_scaler,my_labeler=my_labeler):
    
    filename = os.path.join(pFolder, ''.join([ pName, '_corr_df.csv']))
    p_df = pd.read_csv(filename, index_col=[0], header=0)
    p_inv_df = p_df*conds
    labeled_predictions = make_preds(p_inv_df,clf,my_scaler,my_labeler)
    
    fig = plt.figure(figsize=(16,4))
    with sns.axes_style("ticks"):
        ax = plt.subplot(111)
        ax = make_single_plot(p_df,labeled_predictions,ax)
    sns.despine()
    out_name200 = os.path.join(pFolder, ''.join([ pName, '_timeAll200trs.png']))
    plt.savefig(out_name200,dpi=300,bbox_inches='tight')
    plt.close()
    
    fig = plt.figure(figsize=(10,6))
    with sns.axes_style("ticks"):
        ax = plt.subplot(111)
        ax = make_single_tr_plot(p_df,ax)
    sns.despine()
    out_name20 = os.path.join(pFolder, ''.join([ pName, '_timeCycle20trs.png']))
    plt.savefig(out_name20,dpi=300,bbox_inches='tight')
    
    plt.close()
    
    return out_name200, out_name20



# ### summary
# 
# We see a robust pattern of volume-by-volume correlations with templates, indicating that on the group level, single volumes carry systematic diagnostic information. We also see that usually there is a plausible rank-order of correlations, with left and right at the extremes and bilateral in the middle.
# 
# 
# 
# **************
# 
# < [Previous](11-mw-logistic-regression.ipynb) | [Contents](00-mw-overview-notebook.ipynb) | [Next >](13-mw-make-group-predictions.ipynb)
