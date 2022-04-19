# Author: Vikram Venkatraghavan, Amsterdam UMC

from matplotlib import pyplot as plt 
import numpy as np 
import pandas as pd 
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as pyo

import ggseg

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
    
    color_map = {subtype_labels[i]: color_list[i] for i in range(len(color_list))}

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
#     df_sorted = df_sortBy.groupby('Region').aggregate('mean').sort_values(by='Score', ascending = True)


    labels_sorted = list(df_sorted.index)
    labels_sorted.reverse()

    fig.update_yaxes(categoryarray=labels_sorted)
    fig.update_yaxes(categoryorder="array")
    
    fig.update_layout(xaxis = dict(tickmode = 'linear', 
                                   tick0 = 0.0, 
                                   dtick = 0.1),
                      title_font_size=24,
                      hovermode=False)
#     fig.write_html('event_centers_figure_.html', auto_open=True)


    return fig




def mapping_dk(T):
    """
    Creates a dictionary of DK-atlas labels grouped into larger regions corresponding to T.biomarker_labels
    :param T: Timeline object from snowphlake
    :return: dictionary, key:value => T.biomarker_labels: [dk labels]
    """   
    org_cortical_mapping_left = [['lh_bankssts_volume','lh_transversetemporal_volume',
                                  'lh_superiortemporal_volume','lh_temporalpole_volume','lh_entorhinal_volume',
                                  'lh_middletemporal_volume','lh_inferiortemporal_volume','lh_fusiform_volume'], 
                                 ['lh_superiorfrontal_volume','lh_frontalpole_volume'], 
                                 ['lh_caudalmiddlefrontal_volume','lh_rostralmiddlefrontal_volume'], 
                                 ['lh_parsopercularis_volume','lh_parsorbitalis_volume','lh_parstriangularis_volume'], 
                                 ['lh_medialorbitofrontal_volume'], ['lh_lateralorbitofrontal_volume'], 
                                 ['lh_precentral_volume','lh_paracentral_volume'], ['lh_postcentral_volume'], 
                                 ['lh_superiorparietal_volume','lh_precuneus_volume'], ['lh_inferiorparietal_volume','lh_supramarginal_volume'], 
                                 ['lh_lateraloccipital_volume'], ['lh_cuneus_volume','lh_pericalcarine_volume'], 
                                 ['lh_lingual_volume'], ['lh_insula_volume'], ['lh_caudalanteriorcingulate_volume','lh_rostralanteriorcingulate_volume'], 
                                 ['lh_posteriorcingulate_volume','lh_isthmuscingulate_volume'], 
                                 ['lh_parahippocampal_volume']]

    list_imaging_cortical_left = ['Temporal_lobe_left','Superior_frontal_gyrus_left',
                                  'Middle_frontal_gyrus_left','Inferior_frontal_gyrus_left', 
                                  'Gyrus_rectus_left','Orbitofrontal_gyri_left','Precentral_gyrus_left',
                                  'Postcentral_gyrus_left','Superior_parietal_gyrus_left', 
                                  'Inferolateral_remainder_of_parietal_lobe_left',
                                  'Lateral_remainder_of_occipital_lobe_left','Cuneus_left','Lingual_gyrus_left', 
                                  'Insula_left','Gyrus_cinguli_anterior_part_left','Gyrus_cinguli_posterior_part_left',
                                  'Parahippocampal_and_ambient_gyri_left']

    org_cortical_mapping_right = [['rh_bankssts_volume','rh_transversetemporal_volume',
                                  'rh_superiortemporal_volume','rh_temporalpole_volume','rh_entorhinal_volume',
                                  'rh_middletemporal_volume','rh_inferiortemporal_volume','rh_fusiform_volume'], 
                                 ['rh_superiorfrontal_volume','rh_frontalpole_volume'], 
                                 ['rh_caudalmiddlefrontal_volume','rh_rostralmiddlefrontal_volume'], 
                                 ['rh_parsopercularis_volume','rh_parsorbitalis_volume','rh_parstriangularis_volume'], 
                                 ['rh_medialorbitofrontal_volume'], ['rh_lateralorbitofrontal_volume'], 
                                 ['rh_precentral_volume','rh_paracentral_volume'], ['rh_postcentral_volume'], 
                                 ['rh_superiorparietal_volume','rh_precuneus_volume'], ['rh_inferiorparietal_volume','rh_supramarginal_volume'], 
                                 ['rh_lateraloccipital_volume'], ['rh_cuneus_volume','rh_pericalcarine_volume'], 
                                 ['rh_lingual_volume'], ['rh_insula_volume'], ['rh_caudalanteriorcingulate_volume','rh_rostralanteriorcingulate_volume'], 
                                 ['rh_posteriorcingulate_volume','rh_isthmuscingulate_volume'], 
                                 ['rh_parahippocampal_volume']]

    list_imaging_cortical_right = ['Temporal_lobe_right',
                                   'Superior_frontal_gyrus_right',
                                  'Middle_frontal_gyrus_right',
                                   'Inferior_frontal_gyrus_right', 
                                  'Gyrus_rectus_right',
                                   'Orbitofrontal_gyri_right',
                                   'Precentral_gyrus_right',
                                  'Postcentral_gyrus_right',
                                   'Superior_parietal_gyrus_right', 
                                  'Inferolateral_remainder_of_parietal_lobe_right',
                                  'Lateral_remainder_of_occipital_lobe_right',
                                   'Cuneus_right',
                                   'Lingual_gyrus_right', 
                                  'Insula_right',
                                   'Gyrus_cinguli_anterior_part_right',
                                   'Gyrus_cinguli_posterior_part_right',
                                  'Parahippocampal_and_ambient_gyri_right']
    
    # DK-labels in left hemisphere grouped into cortical regions corresponding to T.biomarker_labels
    dk_left = [org_cortical_mapping_left[0] + org_cortical_mapping_left[16],
         org_cortical_mapping_left[1] + org_cortical_mapping_left[2]+org_cortical_mapping_left[3]+
         org_cortical_mapping_left[4]+org_cortical_mapping_left[5]+org_cortical_mapping_left[6],
         org_cortical_mapping_left[7]+org_cortical_mapping_left[8]+org_cortical_mapping_left[9],
         org_cortical_mapping_left[10]+org_cortical_mapping_left[11]+org_cortical_mapping_left[12],
         org_cortical_mapping_left[14]+org_cortical_mapping_left[15],
         org_cortical_mapping_left[13]]
    
    # DK-labels in right hemisphere grouped into cortical regions corresponding to T.biomarker_labels
    dk_right = [org_cortical_mapping_right[0] + org_cortical_mapping_right[16],
         org_cortical_mapping_right[1] + org_cortical_mapping_right[2]+org_cortical_mapping_right[3]+
         org_cortical_mapping_right[4]+org_cortical_mapping_right[5]+org_cortical_mapping_right[6],
         org_cortical_mapping_right[7]+org_cortical_mapping_right[8]+org_cortical_mapping_right[9],
         org_cortical_mapping_right[10]+org_cortical_mapping_right[11]+org_cortical_mapping_right[12],
         org_cortical_mapping_right[14]+org_cortical_mapping_right[15],
         org_cortical_mapping_right[13]]
    
    # clean region names
    for l in range(len(dk_left)):
        for i in range(len(dk_left[l])):
            dk_left[l][i]=dk_left[l][i].replace('_volume','_left')
            dk_left[l][i]=dk_left[l][i].replace('lh_','')
    
    for l in range(len(dk_right)):
        for i in range(len(dk_right[l])):
            dk_right[l][i]=dk_right[l][i].replace('_volume','_right')
            dk_right[l][i]=dk_right[l][i].replace('rh_','')
    
    dk = dk_left + dk_right

    
    regions = list(map(lambda x: x.lower(), T.biomarker_labels[12:]))
    
    # final dictionary of key: value pairs corresponding to T.biomarker_label: list(DK-labels)
    dic = dict(zip(regions, dk))
    
    return dic


def dk_dict(T,S, mapped_dict, subtype_labels = None, subtype = None):
    
    """
    Creates a dictionary, which can be used as input to ggseg.plot_dk() function
    :param T: timeline object from snowphlake
    :param S: dictionary from snowphlake
    :param mapped_dict: mapping_dk() output; a dictionary with key: values --> T.biomarker_labels: list(DK-labels)
    :param subtype: name or index of the subtype from subtype_lables (optional)  
    :param subtype_labels: a list with names of the subtypes (optional)
    :return: dictionary with scores for each DK region for chosen subtype
    """
    
    unique_subtypes = np.unique(S['subtypes'][~np.isnan(S['subtypes'])])
    if subtype_labels is None:
        subtype_labels = {f'Subtype {i}': i for i in range(len(unique_subtypes))}
        if subtype is None:
            subtype = next(iter(subtype_labels))
    elif subtype is None:
        subtype = subtype_labels[0]  
        
    # clean names from capital letters
    labels = list(map(lambda x: x.lower(), T.biomarker_labels))
       
    dic = dict(zip(labels, T.sequence_model['ordering'][subtype_labels[subtype]]))
                
    # flat lost of dict values (single list of DK-labels)
    dk_flat = [x for v in mapped_dict.values() for x in v]
        
    #Match T.biomarker_labels to DK labels
    list_plot = list()
    for key in mapped_dict.keys():
        for item in mapped_dict[key]:
            list_plot.append(dic[key])
    
    # Dict for dk-label: T.label value
    dic_dk = dict(zip(dk_flat, list_plot))
    
    return dic_dk


def aseg_dict(T, S, subtype_labels = None, subtype = None):
    
    """
    Creates a dictionary, which can be used as input to ggseg.plot_dk() function
    :param T: timeline object from snowphlake
    :param S: dictionary from snowphlake
    :param subtype_labels: a list with names of the subtypes (optional)
    :param subtype: name or index of the subtype from subtype_lables (optional)  
    :return: dictionary with scores for each DK region for chosen subtype
    """

    unique_subtypes = np.unique(S['subtypes'][~np.isnan(S['subtypes'])])
    if subtype_labels is None:
        subtype_labels = {f'Subtype {i}': i for i in range(len(unique_subtypes))}
        if subtype is None:
            subtype = next(iter(subtype_labels))
    elif subtype is None:
        subtype = subtype_labels[0]
    
    dic_aseg = dict(zip(T.biomarker_labels, T.sequence_model['ordering'][subtype_labels[subtype]]))
        
    return dic_aseg


def plot_ggseg(T,S, subtype_labels = None, subtype = None):     

    """
    Creates a dictionary, which can be used as input to ggseg.plot_dk() function
    :param T: timeline object from snowphlake
    :param S: dictionary from snowphlake
    :param subtype_labels: a list with names of the subtypes (optional)
    :param subtype: name or index of the subtype to visualise (optional)  
    :returns two figures -> ggseg.plot_dk() and ggseg.plot_aseg()
    """

    mapped_dict = mapping_dk(T)    
    dk = dk_dict(T, S, mapped_dict = mapped_dict, subtype = subtype)  
    aseg = aseg_dict(T,S, subtype = subtype)

    if subtype is None:
        subtype = 'default = 0'
    
    ggseg.plot_dk(dk, cmap='Reds_r', figsize=(8,8),
              vminmax = [0,25],
              background='k', edgecolor='w', bordercolor='gray', title=f'Subtype: {subtype}',fontsize = 24)

    ggseg.plot_aseg(aseg, cmap='Reds_r', figsize=(8,8),
                vminmax = [0,25],
                background='k', edgecolor='w', bordercolor='gray', title=f'Subcortical regions for Subtype: {subtype}',
                fontsize = 24)


