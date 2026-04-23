"""Build the three graph splits for the Swiss River Network Benchmark.

Produces the ``swiss-1990``, ``swiss-2010``, and ``zurich`` datasets
under ``swissrivernetwork/benchmark/data/``. Takes **no CLI flags** —
edit the ``__main__`` block to change the input range or output layout.
Run this once before training or evaluation.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from swissrivernetwork.gbr25.graph_exporter import plot_graph, plot_linegraph_values, plot_values, remove_node
from swissrivernetwork.graph_preprocessing.gewaessernetz_processor import *
from swissrivernetwork.reader.air_reader import AirReaderFactory
from swissrivernetwork.reader.graph_reader import ResourceRiverReaderFactory
from swissrivernetwork.reader.hydro_to_meteo import Hydro2MeteoMapper
from swissrivernetwork.reader.water_reader import RawBafuReaderFactory
from swissrivernetwork.util.datetime import (
    from_human,
    from_unix_days,
    to_human,
    to_unix_days,
    unix_days,
    unix_timestamp,
)


def combine_graphs(x1, e1, x2, e2):
    offset = x1.shape[0]
    e2_offset = e2 + offset

    x = torch.cat([x1, x2], dim=0)
    edge_index = torch.cat([e1, e2_offset], dim=1)

    return x, edge_index


def remove_short_sequences(df, min_length):
    day_diff = df["epoch_day"].diff()
    breaks = day_diff != 1
    group_id = breaks.cumsum()
    group_sizes = group_id.value_counts()
    valid_groups = group_sizes[group_sizes >= min_length].index
    df_filtered = df[group_id.isin(valid_groups)]
    return df_filtered


def plot_sequence_historgram(df, title="Hisgoram of Day Sequences"):
    day_diff = df["epoch_day"].diff()
    breaks = day_diff != 1
    sequence_id = breaks.cumsum()
    sequence_lengths = sequence_id.value_counts().sort_index()

    plt.figure()
    plt.hist(sequence_lengths, bins=range(1, sequence_lengths.max() + 2), align="left")
    # plt.hist(sequence_lengths, align='right')
    plt.xlabel("Lenght of continous day sequences")
    plt.ylabel("Count")
    plt.title(title)


def print_start_end(df):
    start_date = to_human(from_unix_days(df["epoch_day"].values[0]))
    end_date = to_human(from_unix_days(df["epoch_day"].values[-1]))
    print("start:", start_date, "\tend:", end_date)


def graph_export(x, e, dump_dir, graph_name, train_test_split, filter_values):

    # persist graph:
    torch.save((x, e), f"swissrivernetwork/journal/dump/graph_{graph_name}.pth")

    # Plot Rhine Figure:
    plot_graph(x, e)
    plt.savefig(f"{dump_dir}/figure_{graph_name}.png", dpi=150)
    # plt.show()

    # Read the Water Data and Air Data and Create a Union of all values (remove missing data)
    stations = [str(row[2].item()) for row in x]
    if "-1" in stations:
        stations.remove("-1")
    print(stations, len(stations))
    water_reader = RawBafuReaderFactory.create_water_temperature_alltime_reader()
    air_reader = AirReaderFactory.create_alltime_reader()
    meteo_mapper = Hydro2MeteoMapper()

    dfs = [water_reader[station] for station in stations]
    dfs_air = [air_reader[meteo_mapper.meteo(station)] for station in stations]

    # rename all the dataframes:
    stripped_dfs = []
    for station, df in zip(stations, dfs):
        df = df[["epoch_day", "Wert"]].rename(columns={"Wert": f"{station}_wt"})
        stripped_dfs.append(df)
    stripped_dfs_air = []
    for station, df in zip(stations, dfs_air):
        df = df[["epoch_day", "tre200d0"]].rename(columns={"tre200d0": f"{station}_at"})
        stripped_dfs_air.append(df)

    # join data frames
    total = stripped_dfs[0]
    for df in stripped_dfs[1:]:
        total = total.merge(df, on="epoch_day", how="outer")
    for df in stripped_dfs_air:
        total = total.merge(df, on="epoch_day", how="outer")
    total = total.copy()  # defragmentation

    # Sort by epoch day?
    total = total.sort_values(by="epoch_day").reset_index(drop=True)
    assert total["epoch_day"].is_monotonic_increasing, "not monotonic increasing!"

    # Test for NaN in air temp:
    for col in total.columns:
        if "_at" in col:
            nans = total[col].isna().sum()
            if nans > 0:
                print("NAN IN AIR!!! (use interpolate)", col, nans)
                total[col] = total[col].interpolate(limit=30)

                # df['value'] = df['value'].interpolate()
                # assert total[col].isna().sum() == 0, f'unresolved NaN in {col}'

    # Mark NaN rows
    total["has_nan"] = total.isna().any(axis=1)  # create has_nan values:

    # persist the water data:
    # total.to_csv(f'{dump_dir}/{graph_name}_wt_at_raw.csv', index=False)

    # inspect the water data:

    # plot_nan_locations(total)
    plot_values(total)
    # plot_linegraph_values(total)

    # select rows with no missing data
    total_values_only = filter_values(total)
    print("Total filtered days", len(total_values_only), f"{len(total_values_only) / 365} years")
    assert total["epoch_day"].is_monotonic_increasing, "not monotonic increasing!"

    # Train Test Split:
    train_df = total_values_only[total_values_only["epoch_day"] < train_test_split]
    test_df = total_values_only[total_values_only["epoch_day"] >= train_test_split]

    # Validate no NaNs in Airtemp:
    for df in [train_df, test_df]:
        for col in df.columns:
            if "_at" in col:
                assert df[col].isna().sum() == 0, f"unresolved NaN in {col}"
    print("No NaN in air temperature detected!")

    # persist values
    train_df.to_csv(f"{dump_dir}/{graph_name}_train.csv", index=False)
    test_df.to_csv(f"{dump_dir}/{graph_name}_test.csv", index=False)

    # Statistics
    train_len = len(train_df)
    test_len = len(test_df)
    print(
        "length: train/test\ttrain:",
        train_len / 365,
        "y",
        "\ttest:",
        test_len / 365,
        "y\tsplit:",
        test_len / (train_len + test_len),
    )

    # start, end:
    print_start_end(train_df)
    print_start_end(test_df)

    plot_values(train_df, "Train Set")
    plot_values(test_df, "Test Set")
    plot_linegraph_values(train_df, "Train Set")
    plot_linegraph_values(test_df, "Test Set")
    plot_sequence_historgram(train_df, "Train Set")
    plot_sequence_historgram(test_df, "Test Set")


def filter_1990(total):
    start_date = to_unix_days(from_human(1, 1, 1990))
    total = total[total["epoch_day"] > start_date]
    total_values_only = total[total["has_nan"] == False]
    total_values_only = remove_short_sequences(total_values_only, 90)
    return total_values_only


def create_1990_graph():

    # Date conversion:
    date = to_human(from_unix_days(15700))
    print("date:", date)
    split = to_unix_days(from_human(1, 1, 2013))
    print("first day of test set:", split)

    # 1990: only Rhein ein Rohne?
    rhein_reader = ResourceRiverReaderFactory.rhein_reader("-1990")
    rhone_reader = ResourceRiverReaderFactory.rohne_reader("-1990")

    # Remove -1 Station in Rhine
    x1, e1 = rhein_reader.read()
    x1, e1 = remove_node(x1, e1, 2)  # station 2106 => idx=2
    x1, e1 = remove_node(x1, e1, 0)  # station -1 => idx=0

    x2, e2 = rhone_reader.read()
    x, e = combine_graphs(x1, e1, x2, e2)
    graph_export(x, e, "swissrivernetwork/journal/dump/", "swiss-1990", split, filter_1990)


def filter_2010(total):
    start_date = to_unix_days(from_human(1, 1, 2005))
    end_date = to_unix_days(from_human(31, 12, 2020))
    total = total[total["epoch_day"] > start_date]
    total = total[total["epoch_day"] <= end_date]
    total_filtered = remove_short_sequences(total, 90)
    return total_filtered


def create_2010_graph():

    # date = to_human(from_unix_days(14962+3500))
    # print('date:', date)
    # split = to_unix_days(from_human(1, 1, 2013))
    # print('first day of test set:', split)
    # exit()

    rhein_reader = ResourceRiverReaderFactory.rhein_reader("-2010")
    rhone_reader = ResourceRiverReaderFactory.rohne_reader("-2010")
    ticino_reader = ResourceRiverReaderFactory.ticino_reader()
    inn_reader = ResourceRiverReaderFactory.inn_reader()

    # Prepare Rhein
    x1, e1 = rhein_reader.read()
    x1, e1 = remove_node(x1, e1, 2)  # station 2106 => idx=2
    x1, e1 = remove_node(x1, e1, 0)  # station -1 => idx=0
    # plot_graph(x1, e1)
    # plt.show()

    # Preapre Rhone:
    x2, e2 = rhone_reader.read()
    # plot_graph(x2, e2)
    # plt.show()

    # Prepare Ticino:
    x3, e3 = ticino_reader.read()
    e3 = torch.cat((e3, torch.tensor([[2, 3, 3], [1, 2, 1]])), dim=1)
    x3, e3 = remove_node(x3, e3, 0)
    # plot_graph(x3, e3)
    # plt.show()

    # Preapre Inn:
    x4, e4 = inn_reader.read()
    print(x4, e4)
    e4 = torch.cat((e4, torch.tensor([[2], [1]])), dim=1)
    x4, e4 = remove_node(x4, e4, 0)
    # plot_graph(x4, e4)
    # plt.show()

    # Combine graph:
    x, e = combine_graphs(*combine_graphs(*combine_graphs(x1, e1, x2, e2), x3, e3), x4, e4)

    # Export:
    split = to_unix_days(from_human(1, 1, 2018))
    print("first day of test set:", split)
    graph_export(x, e, "swissrivernetwork/journal/dump/", "swiss-2010", split, filter_2010)


def read_zh_water_temperature(station):
    # Find File
    dir = "/home/benjamin/Downloads/Gewaesser_ZH/zh/126___wt_export/01-Wandel Jasmin/Wandel Jasmin"
    files = os.listdir(dir)
    files = [f for f in files if station in f and "Wassertemperatur" in f]
    assert len(files) == 1, f"Ambigious files for {station}"
    path = f"{dir}/{files[0]}"

    # Read CSV and convert Dates
    df = pd.read_csv(path, encoding="latin1", skiprows=3)
    df = df.rename(columns={";Datum": "Datum"})
    df["timestamp"] = unix_timestamp(df["Datum"], "%d.%m.%Y")
    df["epoch_day"] = unix_days(df["timestamp"])

    # Drop 0 Values (are masking missing values)
    df = df[df["Wert"] != 0]

    # Average by Day (and rename)
    df = df.groupby("epoch_day")["Wert"].mean().reset_index()
    df = df.rename(columns={"Wert": f"{station}_wt"})
    return df


def filter_zh(total):
    # start_date = to_unix_days(from_human(1, 1, 2005)) # TODO
    # end_date = to_unix_days(from_human(31, 12, 2020))
    # total = total[total['epoch_day'] > start_date]
    # total = total[total['epoch_day'] <= end_date]
    total_filtered = remove_short_sequences(total, 90)
    return total_filtered


def create_zh_graph():

    # Convert dates:
    # print('11477', to_human(from_unix_days(11477)))
    # print('1996', to_human(from_unix_days(1996)))
    # print('16400', to_human(from_unix_days(16400)))
    # print('1.1.2009', to_unix_days(from_human(1, 1, 2009))) # 19296
    # print('31.10.2022', to_unix_days(from_human(31, 10, 2022))) # 19296
    # exit()

    # Create Zurich Graph Like Structure:
    # (x, y, stationnr)
    df = pd.read_csv("~/Downloads/Gewaesser_ZH/zh/126___wt_export/Stations.csv")

    # Apply Whitelist:
    whitelist = [
        "552",
        "572",
        "586",
        "548",
        "527",
        "554",
        "531",
        "533",
        "534",
        "517",
        "520",
        "581",
        "522",
        "523",
        "570",
        "597",
    ]
    df = df[df["Stationsnummer"].astype(str).isin(whitelist)]

    xs = df["x"].values - 2000000  # use old coordinates
    ys = df["y"].values - 1000000
    nr = df["Stationsnummer"]
    data = np.column_stack((xs, ys, nr))
    g = torch.tensor(data, dtype=torch.int32)

    # Create look alike edges
    edges_source = []
    edges_target = []
    edge_index = torch.tensor((edges_source, edges_target))
    print(g, edge_index.shape)

    # plot graph:
    graph_name = "zurich"
    dump_dir = "swissrivernetwork/journal/dump/"
    # plot_graph(g, edge_index)

    ##################
    ## MAP ZH Edges ##
    ##################

    ## Read original BAFU Graph
    gwn = GewaesserNetz(
        "../data/gewaesser/gewaessernetz/gewaessernetz",
        types=["Bach", "Bachachs", "Bach_U", "Fluss", "See", "Kanal", "Fluss_U"],
    )
    gwn.read()
    nodes, edges = gwn.edges()

    # Filter Edges by coordinates:
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    zh_nodes = []
    zh_edges = []
    tolerance = 1000
    for e in edges:
        # filter for f_node:
        for node in [e.f_node, e.t_node]:
            if (
                node.x + tolerance > min_x
                and node.x < max_x + tolerance
                and node.y + tolerance > min_y
                and node.y < max_y + tolerance
            ):
                if e not in zh_edges:
                    zh_edges.append(e)
                if node not in zh_nodes:
                    zh_nodes.append(node)
                # if e.t_node not in zh_nodes:
                #    zh_nodes.append(e.t_node)

    # Remove BAFU Stations:
    for e in zh_edges:
        e.station = None

    # Map Stations To edges:
    for i, row in enumerate(g):
        station = row[2]
        sx = row[0]
        sy = row[1]
        min_distance = None
        min_edge = None
        for e in zh_edges:
            ex = (e.f_node.x + e.t_node.x) / 2
            ey = (e.f_node.y + e.t_node.y) / 2
            dist = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
            if min_distance is None or dist < min_distance:
                min_distance = dist
                min_edge = e
        min_edge.station = Station(None, station, None, None)  # set Station
        print("Min Edge Found!")

    ###
    # Clean up and create structure
    ###

    nodes = zh_nodes
    edges = zh_edges

    # prune nodes
    print("start pruning...")
    removed_edges = 1
    while removed_edges > 0:
        edges, removed_edges = prune_one_degree_edges(edges)
        nodes = prune_zero_degree_nodes(nodes)

    print("start collapsing ...")
    nodes, edges = collapse_degree_2_nodes(nodes, edges)
    print_total_degree(nodes)

    # Remove an edge:
    minimal_edge = None
    minimal_distance = None
    rx = 698060
    ry = 243830
    for e in edges:
        ex = (e.t_node.x + e.f_node.x) / 2
        ey = (e.t_node.y + e.f_node.y) / 2
        dist = np.sqrt((rx - ex) ** 2 + (ry - ey) ** 2)
        if minimal_distance is None or dist < minimal_distance:
            minimal_distance = dist
            minimal_edge = e
    minimal_edge.f_node.remove_edge(minimal_edge)
    minimal_edge.t_node.remove_edge(minimal_edge)
    edges.remove(minimal_edge)

    # Initialize two components:
    roots_x = [668917, 668566]
    roots_y = [269259, 254399]

    components = []
    for rx, ry in zip(roots_x, roots_y):
        min_distance = None
        min_node = None
        # map to closest node
        for n in nodes:
            dist = np.sqrt((rx - n.x) ** 2 + (ry - n.y) ** 2)
            if min_distance is None or dist < min_distance:
                min_distance = dist
                min_node = n

        # Set root at node
        root_station = min_node.edges[0]
        root_station.station = Station(None, -1, "root", None)

        # create spanning tree from this node.
        # stats:
        print("current nodes:", len(nodes), "current edges:", len(edges))

        nodes_st, edges_st = create_spanning_tree(nodes, edges, min_node)

        # nodes, edges = collapse_degree_2_nodes(nodes, edges)

        # print('start pruning...')
        # removed_edges = 1
        # while(removed_edges > 0):
        #    edges, removed_edges = prune_one_degree_edges(edges)
        #    nodes = prune_zero_degree_nodes(nodes)
        # print('pruning done.')

        # nodes, edges = collapse_degree_2_nodes(nodes, edges)

        # Convert to simple tree:
        stc = SimpleTreeConverter()
        st_root = stc.convert(nodes_st, edges_st, root_station)

        # convert to Data structure:
        data = stc.convert_pyg(st_root)

        print("pyg data:", data)
        components.append(data)

    # Fix the two components
    c1 = components[0]  # Glatt/Töss
    x1, e1 = c1.x, c1.edge_index
    x1, e1 = remove_node(x1, e1, 0)
    e1 = torch.cat((e1, torch.tensor([[0, 0, 1], [1, 2, 2]])), dim=1)

    c2 = components[1]  # Sihl/ Limmat
    x2, e2 = c2.x, c2.edge_index
    x2, e2 = remove_node(x2, e2, 0)
    x2, e2 = remove_node(x2, e2, 0)
    e2 = torch.cat((e2, torch.tensor([[1, 2, 2], [0, 0, 1]])), dim=1)

    g, edge_index = combine_graphs(x1, e1, x2, e2)
    plot_graph(g, edge_index)
    torch.save((g, edge_index), f"{dump_dir}/graph_{graph_name}.pth")
    print("final graph:", g, edge_index)

    # Merge them.

    zh_nodes = nodes
    zh_edges = edges
    plot_edges(zh_edges)
    plt.show()

    # Read Water Temperatures and Air Temperatures
    # Zurich Fluntern: SMA
    air_reader = AirReaderFactory.create_alltime_reader()
    stations = [str(row[2].item()) for row in g]

    dfs = [read_zh_water_temperature(station) for station in stations]
    dfs_air = [air_reader["SMA"] for _ in stations]  # Hackedihackhack

    # rename all the dataframes: (dfs are already stripped)
    stripped_dfs_air = []
    for station, df in zip(stations, dfs_air):
        df = df[["epoch_day", "tre200d0"]].rename(columns={"tre200d0": f"{station}_at"})
        stripped_dfs_air.append(df)

    # join data frames
    total = dfs[0]
    for df in dfs[1:]:
        total = total.merge(df, on="epoch_day", how="outer")
    for df in stripped_dfs_air:
        total = total.merge(df, on="epoch_day", how="outer")
    total = total.copy()  # defragmentation

    # filter start-end (1.1.2009) to (31.10.2022):
    total = total[total["epoch_day"] >= 14245]
    total = total[total["epoch_day"] <= 19296]

    # Sort by epoch day?
    total = total.sort_values(by="epoch_day").reset_index(drop=True)
    assert total["epoch_day"].is_monotonic_increasing, "not monotonic increasing!"

    # ~~~
    # Attention: Code duplication!
    # ~~~

    # Test for NaN in air temp:
    for col in total.columns:
        if "_at" in col:
            nans = total[col].isna().sum()
            if nans > 0:
                pass
                print("NAN IN AIR!!! (use interpolate)", col, nans)
                assert False, "nan in Input detected"
                # total[col] = total[col].interpolate(limit=30)

                # df['value'] = df['value'].interpolate()
                # assert total[col].isna().sum() == 0, f'unresolved NaN in {col}'

    # Mark NaN rows
    total["has_nan"] = total.isna().any(axis=1)  # create has_nan values:

    # plot_nan_locations(total)
    # plot_values(total)
    # plot_linegraph_values(total)

    # select rows with no missing data
    total_values_only = filter_zh(total)  # filter_values(total) # use 2010 filter?
    print("Total filtered days", len(total_values_only), f"{len(total_values_only) / 365} years")
    assert total["epoch_day"].is_monotonic_increasing, "not monotonic increasing!"

    # Train Test Split:
    train_test_split = to_unix_days(from_human(1, 1, 2020))
    train_df = total_values_only[total_values_only["epoch_day"] < train_test_split]
    test_df = total_values_only[total_values_only["epoch_day"] >= train_test_split]

    # Validate no NaNs in Airtemp:
    for df in [train_df, test_df]:
        for col in df.columns:
            if "_at" in col:
                assert df[col].isna().sum() == 0, f"unresolved NaN in {col}"
    print("No NaN in air temperature detected!")

    # persist values
    train_df.to_csv(f"{dump_dir}/{graph_name}_train.csv", index=False)
    test_df.to_csv(f"{dump_dir}/{graph_name}_test.csv", index=False)

    # Statistics
    train_len = len(train_df)
    test_len = len(test_df)
    print(
        "length: train/test\ttrain:",
        train_len / 365,
        "y",
        "\ttest:",
        test_len / 365,
        "y\tsplit:",
        test_len / (train_len + test_len),
    )

    # start, end:
    print_start_end(train_df)
    print_start_end(test_df)

    plot_values(train_df, "Train Set")
    plot_values(test_df, "Test Set")
    plot_linegraph_values(train_df, "Train Set")
    plot_linegraph_values(test_df, "Test Set")
    plot_sequence_historgram(train_df, "Train Set")
    plot_sequence_historgram(test_df, "Test Set")


if __name__ == "__main__":
    # reate_zh_graph()

    create_1990_graph()
    # create_2010_graph()
    plt.show()
    # plt.close('all')
