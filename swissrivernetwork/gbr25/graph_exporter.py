import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from mpl_toolkits.axes_grid1 import make_axes_locatable

from swissrivernetwork.util.datetime import to_datetime, from_unix_days

'''
This Script exports the Graph such that we can use Graph Neural Networks on it
'''


def remove_node(x, e, node_to_remove):
    new_x = torch.cat([x[:node_to_remove], x[node_to_remove + 1:]], dim=0)
    mask = e != node_to_remove
    new_e = e[:, mask[0] & mask[1]]
    new_e[new_e > node_to_remove] -= 1  # update index
    return new_x, new_e


def plot_graph(
        nodes, e, information=None, color=None, vmin=None, vmax=None, colorbarlabel=None, noisy_node=None, cmap=None,
        skipcolorbar=False, title=None, skipmargin=False, use_static_color=False
):
    node_positions = nodes[:, :2].numpy()
    edges = e.numpy()

    # plt.figure(figsize=(16,10), layout='tight')
    # fig, ax = plt.subplots()
    ax = plt.gca()

    if color is not None:
        min_color = min(list(color.values()))
        max_color = max(list(color.values()))
        print('min_color', min_color, 'max_color', max_color)

    # Plot Edges
    for start, end in e.T:
        x_coords = [node_positions[start, 0], node_positions[end, 0]]
        y_coords = [node_positions[start, 1], node_positions[end, 1]]
        ax.plot(x_coords, y_coords, 'k-', alpha=0.8, zorder=1, linewidth=0.8)

    # get min and max coordintaes    
    if not skipmargin:
        margin = 0.05
        x_coords = node_positions[:, 0]
        y_coords = node_positions[:, 1]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        print('margin ratio: (delta_x): ', max_x - min_x, 'delta y:', max_y - min_y)
        margin_x = margin * (max_x - min_x)
        margin_y = margin * (max_y - min_y)
        ax.set_xlim(min_x - margin_x, max_x + margin_x)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)

    # Set equal aspect ratio
    # ax.set_aspect('equal') # more problems than solutions

    # or magma_r
    if cmap is None:
        cmap = 'viridis_r' if noisy_node is None else 'Blues'  # use other color map for degeneration
    # cmap = 'coolwarm' # only for diff plots
    size = 50 if noisy_node is None else 1000

    # size = 20

    scatter = None
    for x, y, station in nodes:
        text = str(station.item())
        if information is not None and text in information.keys():
            c = color[text]
            if not use_static_color:
                scatter = ax.scatter(
                    x, y, c=c, cmap=cmap, s=size, edgecolors='k', zorder=3, vmin=vmin, vmax=vmax, linewidth=0.3
                )
            else:
                scatter = ax.scatter(
                    x, y, color=c, s=size, edgecolors='k', zorder=3, vmin=vmin, vmax=vmax, linewidth=0.3
                )  # single color
            # scatter = ax.scatter(x, y, c=c, cmap=cmap, s=size, zorder=3, vmin=vmin, vmax=vmax) # no edgecolor
        else:
            if text == noisy_node:
                ax.scatter(x, y, c='red', s=size, edgecolors='k', zorder=2)
            else:
                ax.scatter(x, y, c='white', s=size, edgecolors='k', zorder=2)
                # ax.scatter(x, y, c='red', s=size, edgecolors='k', zorder=2)

        # plt.scatter(node_positions[:, 0], node_positions[:, 1], c='red', s=500, edgecolors='k', zorder=2)

    for x, y, station in nodes:
        text = str(station.item())
        # do not replace the text
        # if information is not None and text in information.keys():
        #    text = information[text]
        if noisy_node is None:  # do not print text on noisy graph
            # ax.text(x, y, text, fontsize=6, ha='center', va='center')
            pass  # do not print text

    # plt.gca().set_aspect(10/16, adjustable='box')
    # ax.set_aspect(10/16, adjustable='box')

    # this?
    # for spine in ax.spines.values():
    #    spine.set_visible(False)

    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    if title is not None:
        plt.title(title)
    else:
        plt.gca().axis('off')

    if scatter is not None and not skipcolorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="5%")
        plt.colorbar(scatter, cax=cax, label=colorbarlabel, shrink=0.5)


def plot_nan_locations(total):
    plt.figure(figsize=(10, 6))
    sns.heatmap(total.isna(), cbar=False, cmap='viridis')
    plt.title('Nan Values Locations')
    plt.xlabel('Stations')
    plt.ylabel('Time')


def plot_values(total, title='Water Temperatures'):
    plt.figure(figsize=(10, 6))
    sns.heatmap(total.drop(columns=['epoch_day', 'has_nan']), cmap='viridis')
    plt.title(title)
    plt.xlabel('Stations')
    plt.ylabel('Time')


def plot_linegraph_values(total, title='Water Temperatures'):
    plt.figure(figsize=(10, 6))

    for column in total.drop(columns=['epoch_day', 'has_nan']):
        sns.lineplot(data=total, x=total['epoch_day'], y=column, label=column)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()


def graph_export(x, e, dump_dir, graph_name):
    from swissrivernetwork.reader.water_reader import RawBafuReaderFactory

    # Plot Rhine Figure:
    plot_graph(x, e)
    plt.show()

    # Read the Water Data and Create a Union of all values (remove missing data)
    stations = [str(row[2].item()) for row in x]
    if '-1' in stations:
        stations.remove('-1')
    print(stations, len(stations))
    water_reader = RawBafuReaderFactory.create_water_temperature_alltime_reader()

    dfs = [water_reader[station] for station in stations]

    # rename all the dataframes:
    stripped_dfs = []
    for station, df in zip(stations, dfs):
        df = df[['epoch_day', 'Wert']].rename(columns={'Wert': station})
        stripped_dfs.append(df)

    # join data frames
    total = stripped_dfs[0]
    for df in stripped_dfs[1:]:
        total = total.merge(df, on='epoch_day', how='outer')

    total['has_nan'] = total.isna().any(axis=1)  # create has_nan values:

    # persist the water data:
    total.to_csv(f'{dump_dir}/water_temperature_{graph_name}.csv', index=False)

    # inspect the water data:

    # plot_nan_locations(total)
    plot_values(total)
    # plot_linegraph_values(total)

    # select rows with no missing data
    total_values_only = total[total['has_nan'] == False]
    plot_values(total_values_only)
    plot_linegraph_values(total_values_only)
    plt.show()
    print(total)

    # Drop the Huge Data Table, where each node has all available Data


if __name__ == '__main__':
    from swissrivernetwork.reader.graph_reader import ResourceRiverReaderFactory

    # use 2010 version:
    GRAPH_VERSION = ['1990', '2010'][1]

    # Read Graph Structure (Rhine only)    
    rhine_reader = ResourceRiverReaderFactory.rhein_reader(f'-{GRAPH_VERSION}')

    x, e = rhine_reader.read()
    # print(x, e)

    # show stations connected to -1
    x, e = remove_node(x, e, 2)  # station 2106 => idx=2
    x, e = remove_node(x, e, 0)  # station -1 => idx=0
    # persist graph data:
    torch.save((x, e), f'swissrivernetwork/gbr25/dump/graph_{GRAPH_VERSION}.pth')

    graph_export(rhine_reader, 'swissrivernetwork/gbr25/dump/', GRAPH_VERSION)
