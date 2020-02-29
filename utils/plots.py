import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import Bbox


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

def bar_metrics(resultsDict):
    df = pd.DataFrame.from_dict(resultsDict)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
    pallette = plt.cm.get_cmap('tab20c',len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns,colors))
    fig = plt.figure(figsize=(20, 15))

    #MAE plot
    ax1 = fig.add_subplot(2,2,1)
    df.loc['mae'].sort_values().plot(kind='bar', colormap='Paired',color=[color_dict.get(x, '#333333') for x in df.loc['mae'].sort_values().index])
    plt.legend()
    plt.title("MAE Metric, lower is better")
    ax2 = fig.add_subplot(2,2,2)
    df.loc['rmse'].sort_values().plot(kind='bar', colormap='Paired',color=[color_dict.get(x, '#333333') for x in df.loc['rmse'].sort_values().index])
    plt.legend()
    plt.title("RMSE Metric, lower is better")
    ax3 = fig.add_subplot(2,2,3)
    df.loc['mape'].sort_values().plot(kind='bar', colormap='Paired',color=[color_dict.get(x, '#333333') for x in df.loc['mape'].sort_values().index])
    plt.legend()
    plt.title("MAPE Metric, lower is better")
    ax4 = fig.add_subplot(2,2,4)
    df.loc['r2'].sort_values(ascending=False).plot(kind='bar', colormap='Paired',color=[color_dict.get(x, '#333333') for x in df.loc['r2'].sort_values(ascending=False).index])
    plt.legend()
    plt.title("R2 Metric, higher is better")
    plt.tight_layout()
    extent1 = full_extent(ax1).transformed(fig.dpi_scale_trans.inverted())
    extent2 = full_extent(ax2).transformed(fig.dpi_scale_trans.inverted())
    extent3 = full_extent(ax3).transformed(fig.dpi_scale_trans.inverted())
    extent4 = full_extent(ax4).transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('results/mae.png', bbox_inches=extent1)
    plt.savefig('results/rmse.png', bbox_inches=extent2)
    plt.savefig('results/mape.png', bbox_inches=extent3)
    plt.savefig('results/r2.png', bbox_inches=extent4)
    plt.show()