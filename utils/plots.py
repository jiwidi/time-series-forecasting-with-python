import matplotlib.pyplot as plt
import pandas as pd

def bar_metrics(resultsDict):
    df = pd.DataFrame.from_dict(resultsDict)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
    pallette = plt.cm.get_cmap('tab20c',len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns,colors))
    fig = plt.figure(figsize=(20, 15))

    for irow in range(len(df)):
        ax = fig.add_subplot(2,2,irow+1)
        df.iloc[irow].sort_values().plot(kind='bar', colormap='Paired',color=[color_dict.get(x, '#333333') for x in df.iloc[irow].sort_values().index])
        plt.legend()

    plt.tight_layout()
    plt.show()