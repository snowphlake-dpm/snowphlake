# Author: Vikram Venkatraghavan, Amsterdam UMC

from matplotlib import pyplot as plt 
import numpy as np 
import pandas as pd 
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as pyo

import ggseg

from mapping_2D import dk_regions_2D, dk_dict_2D, aseg_dict_2D

def subtype_metrics(T):

    fig, ax = plt.subplots(3,1,figsize=(7, 7))
    ax[0].set_title('Metric: Residual sum of squares (RSS)')
    ax[0].plot(range(2,1+T.n_maxsubtypes),-np.diff(T.subtyping_model.rss_data),color='k')
    ax[0].plot(range(2,1+T.n_maxsubtypes),-np.diff(T.subtyping_model.rss_random))
    ax[0].set_xlabel('Number of subtypes')
    ax[0].set_ylabel('Change in RSS')
    ax[0].grid(visible=True,linestyle='--')
    ax[0].legend(['Data','Random'])
    m = np.max(-np.diff(T.subtyping_model.rss_random))
    ax[0].plot([2,T.n_maxsubtypes],[m,m],linestyle='dashed')

    ax[1].set_title('Silhouette score (SS)')
    ax[1].plot(range(1,1+T.n_maxsubtypes),T.subtyping_model.silhouette_score)
    ax[1].set_xlabel('Number of subtypes')
    ax[1].set_ylabel('SS')
    ax[1].grid(visible=True,linestyle='--')

    ax[2].set_title('Cophenetic correlation (CC)')
    ax[2].plot(range(1,1+T.n_maxsubtypes),T.subtyping_model.cophenetic_correlation)
    ax[2].set_xlabel('Number of subtypes')
    ax[2].set_ylabel('CC')
    ax[2].grid(visible=True,linestyle='--')
    fig.tight_layout()

    return fig, ax

def subtypes_piechart(S,diagnosis,diagnostic_labels_for_plotting,title = None,subtype_labels = None):

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    
    unique_subtypes = np.unique(S['subtypes'][~np.isnan(S['subtypes'])])
    if subtype_labels is None:
        subtype_labels = []
        for i in range(len(unique_subtypes)):
            subtype_labels.append('Subtype '+str(int(unique_subtypes[i])))
    
    n_data = np.zeros(len(unique_subtypes))
    for d in diagnostic_labels_for_plotting:
        idx_d = diagnosis == d
        for u in range(len(unique_subtypes)):
            n_data[u] = n_data[u] + np.sum(S['subtypes'][idx_d] == unique_subtypes[u])

    wedges, _ = ax.pie(n_data, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
            bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(subtype_labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)
    
    if title is not None:
        ax.set_title(title)

    plt.show()

    return fig, ax
    

def event_centers(T, S, color_list=['#000000'], chosen_subtypes = None,
        subtype_labels = None, orderBy = None, width=1200, height=900):
    
    """
    Creates event centers box plots for multiple subtypes
    
    :param T: Timeline object
    :param S:
    :param color_list: a list with color names corresponding to each subtype, len(color_list) = len(subtypes). Preferably hex values
    :param chosen_subtypes: a list with names of the subtypes to visualize
    :param subtype_lables: a list with names of the subtype labels 
    :param orderBy: string, name of the subtype to order the boxplots by; default None
    :param width: chosen width of the returned plot
    :param height: chosen height of the returned plot
    :return: plotly box figure
    """

    unique_subtypes = np.unique(S['subtypes'][~np.isnan(S['subtypes'])])
    if subtype_labels is None:
        subtype_labels = []
        for i in range(len(unique_subtypes)):
            subtype_labels.append('Subtype '+str(int(unique_subtypes[i])))
                    
    if orderBy is None:
        orderBy = subtype_labels[0]
                
    if chosen_subtypes is None:
        chosen_subtypes = subtype_labels
        
    num_subtypes = len(subtype_labels)
    
    labels = T.biomarker_labels
    labels_cleaned = map(lambda x: x.replace("-"," "), labels)
    labels_cleaned = map(lambda x: x.replace("_"," "), labels_cleaned)
    labels_cleaned = list(map(lambda x: x.lower(), labels_cleaned))
    
    # key: value --> ordering: region_name
    labels_dict = {num: label for num, label in enumerate(labels_cleaned)}
    
    color_map = {chosen_subtypes[i]: color_list[i] for i in range(len(color_list))}

    # EVENT-CENTERS
    evn = []
    reg = []
    subs = []

    for b in range(T.bootstrap_repetitions):
        for i, s in enumerate(subtype_labels):
            for r in range(len(labels)):
                
                # SUBTYPES 
                subs.append(s)
                
                # EVENT-CENTERS
                evn.append(T.bootstrap_sequence_model[b]['event_centers'][i][r])
                
                # CORRESPONDING REGIONS
                label_number = T.bootstrap_sequence_model[b]['ordering'][i][r]
                reg.append(labels_dict[label_number])
                
                    
    dic = {'Region':reg, 'Subtype':subs, 'Score':evn}
    df = pd.DataFrame(dic)
        
    fig = px.box(df[df['Subtype'].isin(chosen_subtypes)], 
                 x="Score", 
                 y="Region", 
                 color = 'Subtype',
                 color_discrete_map=color_map,
                 title=f"Event Centers", width=width, height=height, 
                 labels={"Score": "Disease Stage",  "Region": "Region Names"})
    
    df_sortBy = df[df['Subtype']==orderBy].drop(columns=['Subtype'])

    # GROUP BY MEDIAN
    df_sorted = df_sortBy.groupby('Region').quantile(q=0.5).sort_values(by='Score', ascending = True)

    # GROUP BY MEAN
    # df_sorted = df_sortBy.groupby('Region').aggregate('mean').sort_values(by='Score', ascending = True)


    labels_sorted = list(df_sorted.index)
    labels_sorted.reverse()

    fig.update_yaxes(categoryarray=labels_sorted, 
                    categoryorder="array", 
                    title_font_size = 18, 
                    tickfont_size=14)

    fig.update_xaxes(title_font_size = 18, 
                    tickfont_size = 14)
    
    fig.update_layout(xaxis = dict(tickmode = 'linear', 
                                   tick0 = 0.0, 
                                   dtick = 0.1),
                      title_font_size=34,
                      hovermode=False)

    fig.update_layout(legend_font_size=22)

    return fig


# ============= 2D PLOTTING =============================================================================================================================================================

def plot_dk_atlas(T,S, subtype_labels = None, subtype = None, slider = None):     
    """
    Creates a dictionary, which can be used as input to ggseg.plot_dk() function
    :param T: timeline object from snowphlake
    :param S: dictionary from snowphlake
    :param subtype_labels: a list with names of the subtypes (optional)
    :param subtype: name or index of the subtype to visualise (optional)  
    :param slider: int
    :returns a figures by plt.show() -> ggseg.plot_dk() 
    """
    mapped_dict = dk_regions_2D(T)    
    
    if slider is None:
        dk = dk_dict_2D(T, S, mapped_dict = mapped_dict, subtype = subtype)  
    else:
        dk_ = dk_dict_2D(T, S, mapped_dict = mapped_dict, subtype = subtype)
        dk = {k: v for k, v in dk_.items() if v <= slider}
           
    if subtype is None:
        # subtype = 'default = 0'
        pass
    else:
        return ggseg.plot_dk(dk, cmap='Reds_r', figsize=(6,6),
                  vminmax = [0,25],
                  background='black', edgecolor='white', bordercolor='gray', title=f'{subtype}',fontsize = 24)


def plot_aseg_atlas(T,S, subtype_labels = None, subtype = None, slider = None):     
    """
    Creates a dictionary, which can be used as input to ggseg.plot_aseg() function
    :param T: timeline object from snowphlake
    :param S: dictionary from snowphlake
    :param subtype_labels: a list with names of the subtypes (optional)
    :param subtype: name or index of the subtype to visualise (optional)  
    :param slider: int
    :returns a figures by plt.show() -> ggseg.plot_aseg()
    """
    if slider is None:  
        aseg = aseg_dict_2D(T,S, subtype = subtype)
    else:
        aseg_ = aseg_dict_2D(T,S, subtype = subtype)
        aseg = {k: v for k, v in aseg_.items() if v <= slider}

    if subtype is None:
        # subtype = 'Subtype 0'
        pass 
    else:
        return ggseg.plot_aseg(aseg, cmap='Reds_r', figsize=(6,6),
                vminmax = [0,25],
                background='black', edgecolor='white', bordercolor='gray', title=f'{subtype}', fontsize = 18)


def plot_ggseg(T,S, subtype_labels = None, subtype = None):     
    """
    Creates a dictionary, which can be used as input to ggseg.plot_dk() function
    :param T: timeline object from snowphlake
    :param S: dictionary from snowphlake
    :param subtype_labels: a list with names of the subtypes (optional)
    :param subtype: name or index of the subtype to visualise (optional)  
    :returns two figures -> ggseg.plot_dk() and ggseg.plot_aseg()
    """

    mapped_dict = dk_regions_2D(T)    
    dk = dk_dict_2D(T, S, mapped_dict = mapped_dict, subtype = subtype)  
    aseg = aseg_dict_2D(T,S, subtype = subtype)

    if subtype is None:
        subtype = 'default = 0'
    
    ggseg.plot_dk(dk, cmap='Reds_r', figsize=(8,8),
              vminmax = [0,25],
              background='k', edgecolor='w', bordercolor='gray', title=f'Subtype: {subtype}',fontsize = 24)

    ggseg.plot_aseg(aseg, cmap='Reds_r', figsize=(8,8),
                vminmax = [0,25],
                background='k', edgecolor='w', bordercolor='gray', title=f'Subcortical regions for Subtype: {subtype}',
                fontsize = 20)




