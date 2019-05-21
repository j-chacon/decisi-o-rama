# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:23:02 2019

@author: jchaconhurtado
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=0, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, ci=None, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            _conf_int = '\n({0}, {1})'.format(valfmt(data[i,j] - ci[i,j]), 
                                               valfmt(data[i,j] + ci[i,j]))
            text = im.axes.text(j, i, valfmt(data[i, j], None) + _conf_int, **kw)
            texts.append(text)

    return texts
#%%
def plot_s1(s1, s1_conf, label='S1', of_label=None, var_label=None):
    n_vars = len(s1[0])
    n_of = len(s1)
    total_width = 0.8
    spacing = total_width/n_vars
    x_bar = np.array([i for i in range(n_vars)])
    
    if of_label is None:
        of_label = ['OF '+str(i) for i in range(n_of)]
    print(of_label)
    if var_label is None:
        var_label = ['X'+str(i) for i in range(n_vars)]
    print(var_label)
    
    for i in range(n_of):
        plt.bar(i*spacing + x_bar, s1[i], yerr=s1_conf[i],
                label=of_label[i], width=spacing)
  
    plt.xticks(range(n_vars), var_label)
    
    plt.ylabel(label)
    plt.legend()
    plt.show()
    
    return
        
# plot_s1(ss[0], ss_conf[0])    
#%%
def plot_s2(s2, s2_conf, label='S2', of_label=None, var_label=None):
    n_vars = len(s2[0])
    n_of = len(s2)
    
    if of_label is None:
        of_label = ['OF '+str(i) for i in range(n_of)]
    
    if var_label is None:
        var_label = ['X'+str(i) for i in range(n_vars)]
        
    for i in range(n_of):
        fig, ax = plt.subplots()
        im, cbar = heatmap(s2[i],
                           var_label,
                           var_label,
                           ax=ax,
                           cmap='YlGn', 
                           cbarlabel=label)
        
        annotate_heatmap(im, ci=s2_conf[i], valfmt="{x:.2f}")
        plt.title(of_label[i])
        # plt.
        # fig.tight_layout()
        plt.show()
        
# plot_s2(ss[-1], ss_conf[-1])