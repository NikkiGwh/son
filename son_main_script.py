
from matplotlib.figure import Figure
import networkx as nx
from pyparsing import with_attribute
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.utils import gen_batches
import math
import matplotlib.animation as animation
from enum import Enum
from functools import reduce

from IPython.core.display import HTML
from IPython.display import display


class CellOrderOne(Enum):
    RANDOM = 1
    HIGHEST_TRAFFIC_FIRST = 2
    LOWEST_TRAFFIC_FIRST = 3


class CellOrderTwo(Enum):
    RANDOM = 1
    HIGHEST_DEGREE_FIRST = 4
    LOWEST_DEGREE_FIRST = 5


class BaseStationOrder(Enum):
    MACRO_FIRST = "macro"
    MICRO_FIRST = "micro"
    FEMTO_FIRST = "femto"
    PICO_FIRST = "pico"


class BinPackingType(Enum):
    BEST_FIT = 1
    WORST_FIT = 2


class Son:
    def __init__(self, hour=1) -> None:
        # read input data
        self.network_filename = "./son_input.xlsx"
        self.bs_staions_pd = pd.read_excel(io=self.network_filename, sheet_name="baseStations",
                                           header=0, index_col=0, dtype={"initial_cells": object})
        self.cells_pd = pd.read_excel(io=self.network_filename,
                                      sheet_name="cells", header=0, index_col=0)
        adjacencies_pd = pd.read_excel(io=self.network_filename,
                                       sheet_name="adjacencies", index_col=0)
        self.traffic_profile_pd = pd.read_excel(
            io=self.network_filename, sheet_name="traffic_profiles", index_col=0)
        # initialize network
        self.graph: nx.Graph = nx.convert_matrix.from_pandas_adjacency(adjacencies_pd)
        self.initialize_node_attributes()
        self.initialize_edge_attributes()
        self.hour = hour
        self.update_network_attributes()

    ############ network initialization methods ##########

    def initialize_node_pos(self):
        """get node positions as dict

        Returns:
            pos (dict[str, list[int]]): as dict of pos-list for each node
        """
        pos: dict[str, list[int]] = {}
        # add all bs_stations positions
        for index, row in self.bs_staions_pd.iterrows():
            pos[str(index)] = [row["pos_x"], row["pos_y"]]

        # add all cell positions
        for index, row in self.cells_pd.iterrows():
            pos[str(index)] = [row["pos_x"], row["pos_y"]]
        return pos

    def initialize_node_attributes(self):
        """initialize all static node attributes

        bs_node (active, pos_x, pos_y, type, tx_power,
                 static_power, capacity, frequency, initial_cells)

        cell_node (pos_x, pos_y, max_traffic, noise, interference,
                   shadow_component, type, traffic_profile_id)
        """
        # add all bs_stations attributes
        data: dict[str, dict] = {}
        for index, row in self.bs_staions_pd.iterrows():
            data[str(index)] = row.to_dict()
            data[str(index)]["active"] = True

        # add all cell attributes
        for index, row in self.cells_pd.iterrows():
            data[str(index)] = row.to_dict()
        nx.classes.function.set_node_attributes(self.graph, data)

    def initialize_edge_attributes(self):
        """initialize static edge attributes and initial active values form pd list(active, distance, sinr)"""
        for _, edge in enumerate(self.graph.edges.data()):
            # set initial active from pd dataFrame
            initial_cell_list = str.split(self.bs_staions_pd.at[str(edge[1]), "initial_cells"], ",")
            if (edge[0] in initial_cell_list):
                self.graph[edge[0]][edge[1]]["active"] = True
            else:
                self.graph[edge[0]][edge[1]]["active"] = False

            # set distance and sinr
            cell_node = self.graph.nodes[edge[0]]
            bs_node = self.graph.nodes[edge[1]]
            edge[2]["distance"] = self.get_euclidean_distance(
                (cell_node["pos_x"], cell_node["pos_y"]), (bs_node["pos_x"], bs_node["pos_y"]))
            edge[2]["sinr"] = self.get_sinr(bs_node["range"], edge[2]["distance"])

    def get_node_style(self):
        node_colors: list[str] = []
        sizes: list[int] = []
        node_edge_colors: list[str] = []
        node_alphas: list[float] = []

        alpha_value_dict = {"active": 1, "inactive": 0.5, "cell": 1}
        node_edge_color_dict = {"active": "#aaff80",
                                "inactive": "grey", "cell": "black", "overload": "#ff0000"}
        node_color_dict = {"macro": "orange", "micro": "blue",
                           "femto": "yellow", "inactive": "grey", "cell": "black"}
        node_sizes_dict = {"macro": 300, "micro": 100, "femto": 80, "cell": 50}

        for _, node in enumerate(self.graph.nodes.data()):
            if (node[1]["type"] == "micro"):
                sizes.append(node_sizes_dict["micro"])
                if node[1]["active"]:
                    node_colors.append(node_color_dict["micro"])
                    if node[1]["load"] > 1:

                        node_edge_colors.append(node_edge_color_dict["overload"])
                    else:
                        node_edge_colors.append(node_edge_color_dict["active"])
                    node_alphas.append(alpha_value_dict["active"])
                else:
                    node_colors.append(node_color_dict["inactive"])
                    node_edge_colors.append(node_edge_color_dict["inactive"])
                    node_alphas.append(alpha_value_dict["inactive"])

            elif node[1]["type"] == "macro":
                sizes.append(node_sizes_dict["macro"])
                if node[1]["active"]:
                    node_colors.append(node_color_dict["macro"])
                    if node[1]["load"] > 1:
                        node_edge_colors.append(node_edge_color_dict["overload"])
                    else:
                        node_edge_colors.append(node_edge_color_dict["active"])
                    node_alphas.append(alpha_value_dict["active"])
                else:
                    node_colors.append((node_color_dict["inactive"]))
                    node_edge_colors.append(node_edge_color_dict["inactive"])
                    node_alphas.append(alpha_value_dict["inactive"])

            elif node[1]["type"] == "cell":
                node_colors.append("black")
                node_alphas.append(alpha_value_dict["cell"])
                node_edge_colors.append(node_edge_color_dict["cell"])
                sizes.append(node_sizes_dict["cell"])
        return (node_colors, sizes, node_edge_colors, node_alphas)

    def get_edge_style(self):
        colors: list[str] = []
        width: list[float] = []

        for _, edge in enumerate(self.graph.edges.data()):
            if (edge[2]["active"] == True):
                colors.append("green")
                width.append(2)
            else:
                colors.append("grey")
                width.append(0.5)
        return (colors, width)

    ###################### network update methods ###################

    def update_hour(self, hour: int):
        """loads new traffic values for according hour and updates network attributes accordingly"""
        self.hour = hour
        self.update_network_attributes()

    def update_network_attributes(self):
        """update all dynamic node and edge attributes in the right order"""
        self.update_cell_node_attributes()
        self.update_bs_node_attributes()
        # update_edge_attributes()

    def update_bs_node_attributes(self):
        """update dynamic bs_node attributes (throughput, traffic, dynamic_power)
        also deactivate bs stations which don't have active edges
        """
        # deactivate bs stations which don't have active edges
        # set throughput, dynamic_power to 0 for inactive bs nodes
        for _, bs_node in enumerate(
            filter(
                lambda x: self.filter_bs_nodes(node=x),
                self.graph.nodes.data())):

            connected_cells = [x[0] for x in self.graph[bs_node[0]].items() if x[1]["active"]]
            if len(connected_cells) == 0:
                self.graph.nodes[bs_node[0]]["active"] = False
                self.graph.nodes[bs_node[0]]["throughput"] = 0
                self.graph.nodes[bs_node[0]]["dynamic_power"] = 0
                self.graph.nodes[bs_node[0]]["traffic"] = 0
                self.graph.nodes[bs_node[0]]["load"] = 0
            else:
                self.graph.nodes[bs_node[0]]["active"] = True

        # calculation for active bs
        for _, bs_node in enumerate(filter(self.filter_active_bs_nodes, self.graph.nodes.data())):
            traffic_sum = 0.0
            # calculate total traffic for bs depending on current connected cells
            for _, neighbor in enumerate(self.graph[bs_node[0]].items()):
                if neighbor[1]["active"] is True:
                    traffic_sum += self.graph.nodes[neighbor[0]]["traffic"]

            # set traffic attribute
            bs_node[1]["traffic"] = traffic_sum
            # calculate total throughput for bs
            bs_node[1]["throughput"] = self.get_throughput(bs_node[1]["capacity"], traffic_sum)
            # calculate load for bs
            bs_node[1]["load"] = self.get_load(bs_node[1]["capacity"], traffic_sum)
            # calculate total dynamic_power for bs
            bs_node[1]["dynamic_power"] = self.get_transmission_power(
                bs_node[1]["tx_power"], traffic_sum)

    def update_cell_node_attributes(self):
        """update dynamic cell_node attributes depending on active edge and hour (sinr, traffic)"""

        for _, cell_node in enumerate(filter(self.filter_cell_nodes, self.graph.nodes.data())):
            # update current traffic considering hour
            load = self.traffic_profile_pd.at[self.hour, cell_node[1]["traffic_profile_id"]]
            cell_node[1]["traffic"] = load * cell_node[1]["max_traffic"]

            # update current sinr for cell_node depending on connected bs
            for _, neighbor in enumerate(self.graph[cell_node[0]].items()):
                if neighbor[1]["active"] is True:
                    cell_node[1]["sinr"] = neighbor[1]["sinr"]
                    break

    def update_edge_attributes(self):
        """updates dynamic edge attributes (currently no dynamic attributes so not used)"""
        for _, edge in enumerate(self.graph.edges.data()):
            print(edge)
            print("-------")

    def set_edge_active(self, cell_node_name: str, bs_node_name: str, active: bool):
        """activates or deactivates edge and triggers node attribute updates

        Args:
            cell_node_name (str): id of the cell node
            bs_node_name (str): id of the bs node
        """
        # deactivate all other edges for cell_node_name which are regarding not bs_node_name if
        # new edge gets activated
        for _, edge in enumerate(self.graph[cell_node_name].items()):
            edge[1]["active"] = active if edge[0] == bs_node_name else False if active else edge[1]["active"]

        # update all node attributes
        self.update_network_attributes()

    def set_edges_active(self, active: bool):
        """deactivates or activates all edges and triggers node attribute updates

        Args:
            active (bool): whether to activate / deactivate all edges
        """
        nx.classes.function.set_edge_attributes(self.graph, active, "active")
        self.update_network_attributes()

    ################### network metric helpers ######################

    def get_throughput(self, max_traffic: float, traffic: float):
        return round(max_traffic / (1 + traffic), 4)

    def get_load(self, capacity: float, traffic: float):
        return round(traffic / capacity, 4)

    def get_transmission_power(self, receiver_sensitivity: float, traffic: float):
        return round(receiver_sensitivity * traffic, 4)

    def get_sinr(self, bs_range: float, distance: float):
        return round(bs_range/(distance**2), 4)

    def get_euclidean_distance(self, pos1: tuple[float, float], pos2: tuple[float, float]):
        return round(math.dist(pos1, pos2), 4)

    def vec_length(self, v):
        return math.sqrt(np.dot(v, v))

    def vec_cos(self, a, b):
        return np.dot(a, b)/(self.vec_length(a)*self.vec_length(b))

    ########################## network filter and sort helpers ###################

    def filter_active_bs_nodes(self, node):
        if node[1]["type"] != "cell":
            return node[1]["active"] == True
        else:
            return False

    def filter_inactive_bs_nodes(self, node):
        if node[1]["type"] != "cell":
            return node[1]["active"] == False
        else:
            return False

    def filter_bs_nodes(self, node):
        return node[1]["type"] != "cell"

    def filter_cell_nodes(self, node):
        return node[1]["type"] == "cell"

    def filter_edges_after_cell_name(self, edge, name: str):
        return edge[0] == name

    def sort_cells_after(self, cell_node, cell_order_one: CellOrderOne,
                         cell_order_two: CellOrderTwo):

        if cell_order_one != CellOrderOne.RANDOM and cell_order_two != CellOrderTwo.RANDOM:
            return cell_node[1]["traffic"] if cell_order_one == CellOrderOne.LOWEST_TRAFFIC_FIRST else -cell_node[1]["traffic"], self.graph.degree[cell_node[0]] if cell_order_two == CellOrderTwo.LOWEST_DEGREE_FIRST else -self.graph.degree[cell_node[0]]
        elif cell_order_one != CellOrderOne.RANDOM and cell_order_two == CellOrderTwo.RANDOM:
            return cell_node[1]["traffic"] if cell_order_one == CellOrderOne.LOWEST_TRAFFIC_FIRST else -cell_node[1]["traffic"]
        elif cell_order_one == CellOrderOne.RANDOM and cell_order_two != CellOrderTwo.RANDOM:
            return self.graph.degree[cell_node[0]] if cell_order_two == CellOrderTwo.LOWEST_DEGREE_FIRST else -self.graph.degree[cell_node[0]]
        else:
            return 0

    ################################ network objective methods ##########################

    def get_average_network_load(self):
        """accumulates average workload over all active base stations in % of base station capacities
        Returns:
            average network load as float
        """
        load_sum = 0
        count = 0
        for _, bs_node in enumerate(filter(self.filter_active_bs_nodes, self.graph.nodes.data())):
            load_sum += bs_node[1]["load"]
            count += 1

        return round(load_sum / count, 4)

    def get_avg_overlad(self):
        """returns accumulated avg overload over all base stations in %
        Returns:
            avg total network overload as float
        """
        overload_sum = 0
        count = 0
        for _, bs_node in enumerate(filter(self.filter_active_bs_nodes, self.graph.nodes.data())):
            count += 1
            if bs_node[1]["load"] > 1:
                overload_sum += (bs_node[1]["load"] - 1)

        return round(overload_sum / count, 4)

    def get_total_energy_consumption(self):
        """get total energy consumption of base station, including dynamic and static power consumption

        Returns:
            sum of all energy consumptions for all base stations
        """
        energy_consumption = 0
        for _, bs_node in enumerate(filter(self.filter_active_bs_nodes, self.graph.nodes.data())):
            energy_consumption += bs_node[1]["dynamic_power"] + bs_node[1]["static_power"]

        return round(energy_consumption, 4)

    def get_average_sinr(self):
        """get average sinr over all cells

        Returns:
            average sinr over all cells
        """
        sinr_sum = 0
        count = 0
        for _, cell_node in enumerate(
                filter(self.filter_cell_nodes, self.graph.nodes.data())):
            sinr_sum += cell_node[1]["sinr"]
            count += 1

        return round(sinr_sum/count, 4)

    ################################ bin packing methods ################################

    def best_fit(
            self, cell_node, bs_order: list[BaseStationOrder] | None = None):
        best_fit_bs_name = None,
        best_fit_capacity = None
        best_fit_sinr = None
        if bs_order is None:
            open_bin_bs_list = []
            for _, neighbor in enumerate(self.graph[cell_node[0]].items()):
                # if bs still has capacity -> add to open_bin_bs_list
                if self.graph.nodes[neighbor[0]]["traffic"] < self.graph.nodes[neighbor[0]][
                        "capacity"]:
                    open_bin_bs_list.append(neighbor[0])

            # find best fitting bs for current cell traffic and sinr
            if len(open_bin_bs_list) > 0:

                for bs_name in open_bin_bs_list:
                    free_capacity = self.graph.nodes[bs_name]["capacity"] - (
                        self.graph.nodes[bs_name]["traffic"] + cell_node[1]["traffic"])

                    if best_fit_capacity is None or (
                            free_capacity >= 0 and free_capacity < best_fit_capacity):
                        best_fit_bs_name = bs_name
                        best_fit_capacity = free_capacity
                        best_fit_sinr = self.graph[cell_node[0]][bs_name]["sinr"]
                    elif free_capacity >= 0 and free_capacity == best_fit_capacity and self.graph[cell_node[0]][bs_name]["sinr"] > best_fit_sinr:
                        best_fit_bs_name = bs_name
                        best_fit_capacity = free_capacity
                        best_fit_sinr = self.graph[cell_node[0]][bs_name]["sinr"]
        # if bs_order is given -> iterate through bs types
        else:
            for bs_type in bs_order:
                open_bin_bs_list = []
                for _, neighbor in enumerate(self.graph[cell_node[0]].items()):
                    # if bs still has capacity and is of type bs_type -> add to open_bin_bs_list
                    if self.graph.nodes[neighbor[0]]["traffic"] < self.graph.nodes[neighbor[0]][
                            "capacity"] and self.graph.nodes[neighbor[0]]["type"] == bs_type.value:
                        open_bin_bs_list.append(neighbor[0])

                # find best fitting bs for current cell traffic and sinr
                if len(open_bin_bs_list) > 0:

                    for bs_name in open_bin_bs_list:
                        free_capacity = self.graph.nodes[bs_name]["capacity"] - (
                            self.graph.nodes[bs_name]["traffic"] + cell_node[1]["traffic"])

                        if best_fit_capacity is None or (
                                free_capacity >= 0 and free_capacity < best_fit_capacity):
                            best_fit_bs_name = bs_name
                            best_fit_capacity = free_capacity
                            best_fit_sinr = self.graph[cell_node[0]][bs_name]["sinr"]
                        elif free_capacity >= 0 and free_capacity == best_fit_capacity and self.graph[cell_node[0]][bs_name]["sinr"] > best_fit_sinr:
                            best_fit_bs_name = bs_name
                            best_fit_capacity = free_capacity
                            best_fit_sinr = self.graph[cell_node[0]][bs_name]["sinr"]

                    # leave loop if cell is found for desired bs_type
                    if best_fit_bs_name is not None:
                        break
        # if no bs with open capacity, take the bs with lowest overload
        if best_fit_capacity is None or best_fit_capacity < 0:
            current_overload = None
            for _, neighbor in enumerate(self.graph[cell_node[0]].items()):
                if current_overload is None or self.graph.nodes[neighbor[0]]["capacity"] - self.graph.nodes[neighbor[0]][
                        "traffic"] > current_overload:
                    current_overload = self.graph.nodes[neighbor[0]
                                                        ]["capacity"] - self.graph.nodes[neighbor[0]]["traffic"]
                    best_fit_bs_name = neighbor[0]
                    best_fit_sinr = self.graph[cell_node[0]][neighbor[0]]["sinr"]
                elif self.graph.nodes[neighbor[0]]["capacity"] - self.graph.nodes[neighbor[0]][
                        "traffic"] == current_overload and self.graph[cell_node[0]][neighbor[0]]["sinr"] > best_fit_sinr:
                    current_overload = self.graph.nodes[neighbor[0]
                                                        ]["capacity"] - self.graph.nodes[neighbor[0]]["traffic"]
                    best_fit_bs_name = neighbor[0]
                    best_fit_sinr = self.graph[cell_node[0]][neighbor[0]]["sinr"]

        # activate best-fit edge and update node attributes
        self.set_edge_active(
            cell_node_name=cell_node[0],
            bs_node_name=best_fit_bs_name, active=True)

    def worst_fit(
            self, cell_node, bs_order: list[BaseStationOrder] | None = None):
        worst_fit_bs_name = None,
        worst_fit_capacity = None
        worst_fit_sinr = None
        if bs_order is None:
            open_bin_bs_list = []
            for _, neighbor in enumerate(self.graph[cell_node[0]].items()):
                # if bs still has capacity -> add to open_bin_bs_list
                if self.graph.nodes[neighbor[0]]["traffic"] < self.graph.nodes[neighbor[0]][
                        "capacity"]:
                    open_bin_bs_list.append(neighbor[0])

            # find best fitting bs for current cell traffic and sinr
            if len(open_bin_bs_list) > 0:

                for bs_name in open_bin_bs_list:
                    free_capacity = self.graph.nodes[bs_name]["capacity"] - (
                        self.graph.nodes[bs_name]["traffic"] + cell_node[1]["traffic"])
                    if worst_fit_capacity is None or (
                            free_capacity >= 0 and free_capacity > worst_fit_capacity):
                        worst_fit_bs_name = bs_name
                        worst_fit_capacity = free_capacity
                        worst_fit_sinr = self.graph[cell_node[0]][bs_name]["sinr"]
                    elif free_capacity >= 0 and free_capacity == worst_fit_capacity and self.graph[cell_node[0]][bs_name]["sinr"] > worst_fit_sinr:
                        worst_fit_bs_name = bs_name
                        worst_fit_capacity = free_capacity
                        worst_fit_sinr = self.graph[cell_node[0]][bs_name]["sinr"]
        # if bs_order is given -> iterate through bs types
        else:
            for bs_type in bs_order:
                open_bin_bs_list = []
                for _, neighbor in enumerate(self.graph[cell_node[0]].items()):
                    # if bs still has capacity and is of type bs_type -> add to open_bin_bs_list
                    if self.graph.nodes[neighbor[0]]["traffic"] < self.graph.nodes[neighbor[0]][
                            "capacity"] and self.graph.nodes[neighbor[0]]["type"] == bs_type.value:
                        open_bin_bs_list.append(neighbor[0])

                # find best fitting bs for current cell traffic and sinr
                if len(open_bin_bs_list) > 0:

                    for bs_name in open_bin_bs_list:
                        free_capacity = self.graph.nodes[bs_name]["capacity"] - (
                            self.graph.nodes[bs_name]["traffic"] + cell_node[1]["traffic"])

                        if worst_fit_capacity is None or (
                                free_capacity >= 0 and free_capacity > worst_fit_capacity):
                            worst_fit_bs_name = bs_name
                            worst_fit_capacity = free_capacity
                            worst_fit_sinr = self.graph[cell_node[0]][bs_name]["sinr"]
                        elif free_capacity >= 0 and free_capacity == worst_fit_capacity and self.graph[cell_node[0]][bs_name]["sinr"] > worst_fit_sinr:
                            worst_fit_bs_name = bs_name
                            worst_fit_capacity = free_capacity
                            worst_fit_sinr = self.graph[cell_node[0]][bs_name]["sinr"]

                    # leave loop if cell is found for desired bs_type
                    if worst_fit_bs_name is not None:
                        break
        # if no bs with open capacity, take the bs with lowest overload
        if worst_fit_capacity is None or worst_fit_capacity < 0:
            current_overload = None
            for _, neighbor in enumerate(self.graph[cell_node[0]].items()):
                if current_overload is None or self.graph.nodes[neighbor[0]]["capacity"] - self.graph.nodes[neighbor[0]][
                        "traffic"] > current_overload:
                    current_overload = self.graph.nodes[neighbor[0]
                                                        ]["capacity"] - self.graph.nodes[neighbor[0]]["traffic"]
                    worst_fit_bs_name = neighbor[0]
                    worst_fit_sinr = self.graph[cell_node[0]][neighbor[0]]["sinr"]
                elif self.graph.nodes[neighbor[0]]["capacity"] - self.graph.nodes[neighbor[0]][
                        "traffic"] == current_overload and self.graph[cell_node[0]][neighbor[0]]["sinr"] > worst_fit_sinr:
                    current_overload = self.graph.nodes[neighbor[0]
                                                        ]["capacity"] - self.graph.nodes[neighbor[0]]["traffic"]
                    worst_fit_bs_name = neighbor[0]
                    worst_fit_sinr = self.graph[cell_node[0]][neighbor[0]]["sinr"]

        # activate best-fit edge and update node attributes
        self.set_edge_active(
            cell_node_name=cell_node[0],
            bs_node_name=worst_fit_bs_name, active=True)

    def find_activation_profile_bin_packing(
            self, cell_order_1: CellOrderOne = CellOrderOne.RANDOM,
            cell_order_2: CellOrderTwo = CellOrderTwo.RANDOM, bs_order: list[BaseStationOrder] |
            None = None, bin_packing: BinPackingType = BinPackingType.BEST_FIT):

        # deactivate all edges and update all node attributes
        self.set_edges_active(False)

        # prepare cell iteration order
        cell_node_list = list(filter(self.filter_cell_nodes, self.graph.nodes.data()))
        cell_node_list.sort(key=lambda x: self.sort_cells_after(
            x, cell_order_one=cell_order_1, cell_order_two=cell_order_2))

        # perform bin packing for each cell and activate single edges
        for _, cell_node in enumerate(cell_node_list):
            if bin_packing == BinPackingType.BEST_FIT:
                self.best_fit(cell_node, bs_order=bs_order)
            elif bin_packing == BinPackingType.WORST_FIT:
                self.worst_fit(cell_node, bs_order=bs_order)

    ############### get encodings, apply encodings, save/load encodings ###################

    def get_edge_activation_encoding_from_graph(self):
        """get the current activation profile of the network edges as list decoding

        Returns:
            list of binary str with each str representing activation profile of base stations for
            each cell
        """
        cell_encoding_list: list[str] = []
        for _, cell_node in enumerate(
                filter(self.filter_cell_nodes, self.graph.nodes.data())):
            cell_encoding_str = ""
            for _, edge in enumerate(self.graph[cell_node[0]].items()):
                if edge[1]["active"]:
                    cell_encoding_str += "1"
                else:
                    cell_encoding_str += "0"
            cell_encoding_list.append(cell_encoding_str)
        return cell_encoding_list

    def apply_edge_activation_encoding_to_graph(self, encoding: list[str]):
        """takes activation list encoding and applies it on network accordingly

        Args:
            encoding (list[str]): activation list of cell edges
        """
        for cell_node_index, cell_node in enumerate(
                filter(self.filter_cell_nodes, self.graph.nodes.data())):
            for edge_index, edge in enumerate(self.graph[cell_node[0]].items()):
                if encoding[cell_node_index][edge_index] == "1":
                    self.set_edge_active(cell_node[0], edge[0], True)
                    break
                else:
                    self.set_edge_active(cell_node[0], edge[0], False)

    def save_edge_activation_profile_to_file(
            self, encoding: list[list[str]],
            result_sheet_name: str):
        df = pd.DataFrame(encoding)

        with pd.ExcelWriter(self.network_filename, mode="a", if_sheet_exists="replace") as writer:  # pylint: disable=abstract-class-instantiated
            df.to_excel(writer, result_sheet_name, header=False, index=False)

    def get_edge_activation_encoding_from_file(self, result_sheet_name: str):
        df = pd.read_excel(self.network_filename, result_sheet_name,
                           dtype=object, header=None, index_col=None)
        encodings: list[list[str]] = df.values.tolist()
        return encodings

    ################ start calculation methods without visualization and with saving the results ############

    def start_calculation_bin_packing_save(
            self, sheet_name: str,
            cell_order_1: CellOrderOne = CellOrderOne.HIGHEST_TRAFFIC_FIRST,
            cell_order_2: CellOrderTwo = CellOrderTwo.LOWEST_DEGREE_FIRST, bs_order: None |
            list[BaseStationOrder] = None, bin_packing: BinPackingType = BinPackingType.BEST_FIT):
        """call this to start simulation with desired bin backing settings for 24 hour and saving 
        activation profiles to sheet
        TODO complete doc 
        """
        result: list[list[str]] = []
        for i in range(24):
            # update hour
            self.update_hour(i+1)
            self.find_activation_profile_bin_packing(
                cell_order_1=cell_order_1, cell_order_2=cell_order_2, bs_order=bs_order,
                bin_packing=bin_packing)
            result.append(self.get_edge_activation_encoding_from_graph())

        self.save_edge_activation_profile_to_file(result, sheet_name)

    ############## visualization methods ####################

    def animate_from_encoding(
            self, i, fig: Figure, pos, activation_matrix: list[list[str]],
            fig_title: str = ""):
        fig.clear()
        plt.title(str(i+1) + "'s hour" + " - " + fig_title)

        # update hour
        self.update_hour(i+1)
        # apply activation profile -> update measures
        self.apply_edge_activation_encoding_to_graph(activation_matrix[i])

        node_colors, node_sizes, node_edge_colors, node_alphas = self.get_node_style()
        edge_colors, edge_width = self.get_edge_style()
        options_edges = {
            'width': edge_width,
            "edge_color": edge_colors,
            "pos": pos,
        }
        options_nodes = {
            "alpha": node_alphas,
            'node_color': node_colors,
            'node_size': node_sizes,
            "pos": pos,
            "edgecolors": node_edge_colors,
        }
        nx.drawing.nx_pylab.draw_networkx_edges(self.graph, **options_edges)
        nx.drawing.nx_pylab.draw_networkx_nodes(self.graph, **options_nodes)

    def start_animation_result_from_sheet(self, sheet_name: str):
        """start simulation over 24 hours with visualization
        TODO finish doc
        """
        # load activation profile from sheet_name
        activation_matrix = self.get_edge_activation_encoding_from_file(
            sheet_name)

        pos = self.initialize_node_pos()
        fig, _ = plt.subplots(figsize=(10, 6))

        anim = animation.FuncAnimation(
            fig, lambda i: self.animate_from_encoding(
                i, fig=fig, pos=pos, activation_matrix=activation_matrix, fig_title=sheet_name),
            frames=24, interval=1000, repeat=True)

        display(HTML(anim.to_jshtml()))
        plt.close(fig)

    def draw_current_network(self):
        pos = self.initialize_node_pos()
        node_colors, node_sizes, node_edge_colors, node_alphas = self.get_node_style()
        edge_colors, edge_width = self.get_edge_style()
        options_edges = {
            'width': edge_width,
            "edge_color": edge_colors,
            "pos": pos,
        }
        options_nodes = {
            "alpha": node_alphas,
            'node_color': node_colors,
            'node_size': node_sizes,
            "pos": pos,
            "edgecolors": node_edge_colors,
        }
        plt.title(f"{self.hour}'s hour")
        nx.drawing.nx_pylab.draw_networkx_edges(self.graph, **options_edges)
        nx.drawing.nx_pylab.draw_networkx_nodes(self.graph, **options_nodes)
        plt.show()

    def draw_24_hour_profile_results(self, result_sheet_name_list: list[str]):

        _, ax_sinr = plt.subplots(layout="constrained")
        _, ax_load = plt.subplots(layout="constrained")
        _, ax_overload = plt.subplots(layout="constrained")
        _, ax_energy = plt.subplots(layout="constrained")
        ax_load.set_ylim(0, 1)

        avg_sinr_values_dict: dict[str, list[float]] = {}
        overload_values_dict: dict[str, list[float]] = {}
        energy_values_dict: dict[str, list[float]] = {}
        avg_load_values_dict: dict[str, list[float]] = {}

        for _, sheet_name in enumerate(result_sheet_name_list):
            # load 24 activation profile for sheet_name
            activation_profile_24_hour_matrix = self.get_edge_activation_encoding_from_file(
                sheet_name)

            overload_list = []
            energy_list = []
            avg_load_list = []
            avg_sinr_list = []

            for i_hour, current_activation_profile in enumerate(activation_profile_24_hour_matrix):

                # update current hour
                self.update_hour(i_hour + 1)
                # apply activation profile of current hour for sheet_name
                self.apply_edge_activation_encoding_to_graph(current_activation_profile)
                # append avg_sinr to avg_sinr_list for sheet_name
                avg_sinr_list.append(self.get_average_sinr())
                avg_load_list.append(self.get_average_network_load())
                overload_list.append(self.get_avg_overlad())
                energy_list.append(self.get_total_energy_consumption())
            # insert avg_sinr_list dict for plotting
            avg_sinr_values_dict[sheet_name] = avg_sinr_list
            overload_values_dict[sheet_name] = overload_list

            energy_values_dict[sheet_name] = energy_list
            avg_load_values_dict[sheet_name] = avg_load_list

        x = np.arange(24)  # the label locations
        width = 1/(len(result_sheet_name_list)+1)  # the width of the bars
        multiplier = 0
        for sheet_name, avg_sinr_24_hour_list in avg_sinr_values_dict.items():
            offset = width * multiplier
            ax_sinr.bar(x + offset, avg_sinr_24_hour_list, width, label=sheet_name)
            ax_energy.bar(x + offset, energy_values_dict[sheet_name], width, label=sheet_name)
            ax_overload.bar(x + offset, overload_values_dict[sheet_name], width, label=sheet_name)
            ax_load.bar(x + offset, avg_load_values_dict[sheet_name], width, label=sheet_name)
            multiplier += 1

        # labels and legend
        ax_sinr.set_ylabel('sinr')
        ax_sinr.set_title('avg_sinr per hour')
        ax_sinr.legend(loc='lower left', bbox_to_anchor=(0, 1.05))

        ax_overload.set_ylabel('avg_overload in %')
        ax_overload.set_title('avg_overload per hour')
        ax_overload.legend(loc='lower left', bbox_to_anchor=(0, 1.05))

        ax_load.set_ylabel('avg load in %')
        ax_load.set_title('avg_load per hour')
        ax_load.legend(loc='lower left', bbox_to_anchor=(0, 1.05))

        ax_energy.set_ylabel('energy consumption')
        ax_energy.set_title('total_energy_consumption per hour')
        ax_energy.legend(loc='lower left', bbox_to_anchor=(0, 1.05))
        plt.show()


################# script main #######################

def main():
    son = Son()

    # for testing single hours
    # son.update_hour(9)

    # apply bin bin packing an save 24 hour profiles
    # bestFit_highestTrafficFirst_HighestDegreeFirst_bsOrderNone
    son.start_calculation_bin_packing_save(
        sheet_name="bf_htf_hdf_bsn",
        cell_order_1=CellOrderOne.HIGHEST_TRAFFIC_FIRST,
        cell_order_2=CellOrderTwo.HIGHEST_DEGREE_FIRST, bs_order=None,
        bin_packing=BinPackingType.BEST_FIT)

    # worstFit_highestTrafficFirst_HighestDegreeFirst_bsOrderNone
    son.start_calculation_bin_packing_save(
        sheet_name="wf_htf_tdf_bsn",
        cell_order_1=CellOrderOne.HIGHEST_TRAFFIC_FIRST,
        cell_order_2=CellOrderTwo.HIGHEST_DEGREE_FIRST, bs_order=None,
        bin_packing=BinPackingType.WORST_FIT)

    # draw avg_sinr graphs
    son.draw_24_hour_profile_results(
        ["bf_htf_hdf_bsn", "nsga2_l_s_0.5_0.5", "nsga2_l_s_0_1", "nsga2_l_s_1_0"])


def animate_result(sheet: str = "nsga2_l_s_0.5_0.5"):
    son = Son()
    son.start_animation_result_from_sheet(
        "nsga2_l_s_0.5_0.5")


if __name__ == "__main__":
    main()
