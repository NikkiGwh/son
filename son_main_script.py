from xmlrpc.client import boolean
from matplotlib.figure import Figure
import networkx as nx
from pyparsing import with_attribute
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.utils import gen_batches
import math
import matplotlib.animation as animation
from enum import Enum
from functools import reduce, total_ordering

from IPython.core.display import HTML
from IPython.display import display
import pandas as pd

from networkx.readwrite import json_graph
import json


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


class NodeType(Enum):
    CELL = "cell"
    MICRO = "micro"
    MACRO = "macro"
    FEMTO = "femto"
    PICO = "pico"


default_node_params = {
    "macro": {
        "type": "macro",
        "tx_power": 20,
        "static_power": 4.0,
        "standby_power": 0,
        "antennas": 8,
        "frequency": 0,
        "wave_length": 80,
        "channel_bandwidth": 10
    },
    "micro": {
        "type": "micro",
        "tx_power": 1,
        "static_power": 1.5,
        "standby_power": 0,
        "antennas": 3,
        "frequency": 0,
        "wave_length": 80,
        "channel_bandwidth": 10
    },
    "femto": {
        "type": "femto",
        "tx_power": 1,
        "static_power": 4.0,
        "standby_power": 0,
        "antennas": 4,
        "frequency": 0,
        "wave_length": 60,
        "channel_bandwidth": 10
    },
    "pico": {
        "type": "pico",
        "tx_power": 1,
        "static_power": 4.0,
        "standby_power": 0,
        "antennas": 4,
        "frequency": 0,
        "wave_length": 40,
        "channel_bandwidth": 10
    },
    "cell": {
        "type": "cell",
        "shadow_component": 0,
        "noise": 0
    }
}


class Son:
    def __init__(self, adjacencies_file_name: str = "", parameter_config_file_name: str = "") -> None:

        # initialize super parameter
        self.min_rssi = 0.8
        self.network_node_params = default_node_params
        # initialize network
        self.graph = nx.Graph()

        self.initialize_edges()
        self.update_network_attributes()

        if parameter_config_file_name != "":
            self.load_parameter_config_from_json_file(parameter_config_file_name)
        if adjacencies_file_name != "":
            self.load_graph_from_json_adjacency_file(
                adjacencies_file_name, keep_activation_profile=True)

    ############ network initialization methods ##########

    def initialize_edges(self):
        edge_list_with_attributes: list[tuple[str, str, dict]] = []
        self.graph.clear_edges()
        # only add edges for cell-bs pairs which hav minimum rssi signal strength
        for _, cell in enumerate(filter(self.filter_user_nodes, self.graph.nodes.data())):
            for _, bs_node in enumerate(
                    filter(self.filter_bs_nodes, self.graph.nodes.data())):
                current_rssi = self.get_rssi_cell(cell[0], (cell[0], bs_node[0]))
                current_distance = self.get_euclidean_distance(
                    (cell[1]["pos_x"], cell[1]["pos_y"]), (bs_node[1]["pos_x"], bs_node[1]["pos_y"]))

                if current_rssi >= self.min_rssi:
                    attribute_dic = {}
                    attribute_dic["rssi"] = current_rssi
                    attribute_dic["distance"] = current_distance
                    edge_list_with_attributes.append(
                        (cell[0], bs_node[0], attribute_dic))

        self.graph.add_edges_from(edge_list_with_attributes)

        # set initial active edges attribute
        for _, edge in enumerate(self.graph.edges.data()):
            self.graph[edge[0]][edge[1]]["active"] = False

    def get_node_style(self):
        node_colors: list[str] = []
        sizes: list[int] = []
        node_edge_colors: list[str] = []
        node_alphas: list[float] = []

        alpha_value_dict = {"active": 1, "inactive": 0.5, "cell": 1}
        node_edge_color_dict = {"active": "#aaff80",
                                "inactive": "grey", "cell": "black", "overload": "#ff0000"}
        node_color_dict = {"macro": "orange", "micro": "blue", "pico": "pink",
                           "femto": "yellow", "inactive": "grey", "cell": "black"}
        node_sizes_dict = {"macro": 300, "micro": 100, "femto": 80, "pico": 50, "cell": 50}

        for _, node in enumerate(self.graph.nodes.data()):
            if (node[1]["type"] == NodeType.MICRO.value):
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

            elif node[1]["type"] == NodeType.MACRO.value:
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

            elif node[1]["type"] == NodeType.FEMTO.value:
                sizes.append(node_sizes_dict["femto"])
                if node[1]["active"]:
                    node_colors.append(node_color_dict["femto"])
                    if node[1]["load"] > 1:
                        node_edge_colors.append(node_edge_color_dict["overload"])
                    else:
                        node_edge_colors.append(node_edge_color_dict["active"])
                    node_alphas.append(alpha_value_dict["active"])
                else:
                    node_colors.append((node_color_dict["inactive"]))
                    node_edge_colors.append(node_edge_color_dict["inactive"])
                    node_alphas.append(alpha_value_dict["inactive"])

            elif node[1]["type"] == NodeType.PICO.value:
                sizes.append(node_sizes_dict["pico"])
                if node[1]["active"]:
                    node_colors.append(node_color_dict["pico"])
                    if node[1]["load"] > 1:
                        node_edge_colors.append(node_edge_color_dict["overload"])
                    else:
                        node_edge_colors.append(node_edge_color_dict["active"])
                    node_alphas.append(alpha_value_dict["active"])
                else:
                    node_colors.append((node_color_dict["inactive"]))
                    node_edge_colors.append(node_edge_color_dict["inactive"])
                    node_alphas.append(alpha_value_dict["inactive"])

            elif node[1]["type"] == NodeType.CELL.value:
                node_colors.append(node_color_dict["cell"])
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

    def update_network_attributes(self):
        """update all dynamic node and edge attributes in the right order"""
        self.update_user_node_attributes()
        self.update_bs_node_attributes()
        self.update_user_node_dl_datarate()
        self.update_edge_attributes()

    def update_bs_node_attributes(self):
        """update dynamic bs_node attributes (traffic, total_power, active)
        also deactivate bs stations which don't have active edges
        """
        # deactivate bs stations which don't have active edges
        # set traffic, total_power to 0 for inactive bs nodes
        for _, bs_node in enumerate(
            filter(
                lambda x: self.filter_bs_nodes(node=x),
                self.graph.nodes.data())):

            connected_users = [x[0] for x in self.graph[bs_node[0]].items() if x[1]["active"]]
            if len(connected_users) == 0:
                # calculation for inactive users
                bs_node[1]["active"] = False
                self.graph.nodes[bs_node[0]]["total_power"] = 0
                bs_node[1]["traffic"] = 0
                bs_node[1]["load"] = 0
            else:
                bs_node[1]["active"] = True

        # calculation for active bs
        for _, bs_node in enumerate(filter(self.filter_active_bs_nodes, self.graph.nodes.data())):
            traffic_sum = 0.0
            # calculate total traffic for bs depending on current connected users -> per user add 1
            for _, user_edge in enumerate(self.graph[bs_node[0]].items()):
                if user_edge[1]["active"] is True:
                    traffic_sum += 1

            # set traffic attribute
            bs_node[1]["traffic"] = traffic_sum
            # set load attribute
            bs_node[1]["load"] = traffic_sum / self.network_node_params[bs_node[1]
                                                                        ["type"]]["antennas"]if traffic_sum > 0 else 0
            # calculate total power consumption for bs -> call only after traffic and load estimations are done
            bs_node[1]["total_power"] = self.get_total_power(bs_node[0])

    def update_user_node_attributes(self):
        """ update dynamic user_node attributes depending on active edge and hour (sinr, rssi)
        not dl_datarate because this value has to be updated separately after other attributes of bs_nodes and
        user_nodes were updated

        """

        for _, user_node in enumerate(filter(self.filter_user_nodes, self.graph.nodes.data())):
            user_node[1]["sinr"] = self.get_sinr(user_node[0])
            connected_bs = self.get_connected_bs_nodeid(user_node[0])
            user_node[1]["rssi"] = self.get_rssi_cell(
                user_node[0], (user_node[0], connected_bs)) if connected_bs is not None else 0

            # user_node[1]["dl_datarate"] = self.get_dl_datarate_user(user_node[0])

    def update_user_node_dl_datarate(self):
        """updates downlink data rate for users -> call after update_user_nodes and update_bs_nodes"""
        for _, user_node in enumerate(filter(self.filter_user_nodes, self.graph.nodes.data())):
            user_node[1]["dl_datarate"] = self.get_dl_datarate_user(user_node[0])

    def update_edge_attributes(self):
        """ updates dynamic edge attributes (rssi)
        """
        for _, user_node in enumerate(filter(self.filter_user_nodes, self.graph.nodes.data())):
            for _, bs_node_id in enumerate(self.graph[user_node[0]]):
                self.graph[bs_node_id][user_node[0]]["rssi"] = user_node[1]["rssi"]

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

    def move_node(self, node_id: str, pos: tuple[float, float], update_network=True):
        self.graph.nodes.data()[node_id]["pos_x"] = round(pos[0], 2)
        self.graph.nodes.data()[node_id]["pos_y"] = round(pos[1], 2)
        self.initialize_edges()

        if update_network:
            self.update_network_attributes()

    def add_bs_node(
            self, pos: tuple[float, float],
            bs_type: str,
            update_network=True):

        i = self.graph.number_of_nodes()
        node_id = "bs_" + str(i)
        node_list = list(self.graph.nodes)
        while node_list.count(node_id) > 0:
            i += 1
            node_id = "bs_" + str(i)
        self.graph.add_node(
            node_id, pos_x=pos[0],
            pos_y=pos[1],
            type=bs_type, load=0, active=True)

        self.initialize_edges()
        if update_network:
            self.update_network_attributes()

    def add_user_node(
            self, pos: tuple[float, float], update_network=True):
        i = self.graph.number_of_nodes()
        node_id = "cell_" + str(i)
        node_list = list(self.graph.nodes)

        while node_list.count(node_id) > 0:
            i += 1
            node_id = "cell_" + str(i)
        self.graph.add_node(
            node_id, pos_x=pos[0],
            pos_y=pos[1],
            type="cell", noise=self.network_node_params["cell"]["noise"],
            shadow_component=self.network_node_params["cell"]["shadow_component"])
        self.initialize_edges()
        if update_network:
            self.update_network_attributes()

    def remove_node(self, node_id: str, update_network=True):
        self.graph.remove_node(node_id)
        self.initialize_edges()
        if update_network:
            self.update_network_attributes()

    def apply_network_node_attributes(
            self, param_list: dict[str, dict[str, float]]):
        self.network_node_params = param_list
        self.initialize_edges()
        self.update_network_attributes()

    def set_edges_active(self, active: bool):
        """deactivates or activates all edges and triggers node attribute updates

        Args:
            active (bool): whether to activate / deactivate all edges
        """
        nx.classes.function.set_edge_attributes(self.graph, active, "active")
        self.update_network_attributes()

    ################### network metric helpers ######################

    def get_dl_datarate_user(self, cell_id: str):
        """
        call after bs_nodes and user_nodes were updated
        """
        connected_bs = self.get_connected_bs_nodeid(cell_id)
        if (connected_bs is None):
            return 0
        connected_bs_type = self.graph.nodes[connected_bs]["type"]
        channel_bandwidth = self.network_node_params[connected_bs_type]["channel_bandwidth"]
        sinr = self.graph.nodes[cell_id]["sinr"]
        channel_capacity = channel_bandwidth * math.log(1+sinr, 2)
        factor = self.network_node_params[connected_bs_type]["antennas"] / self.graph.nodes[
            connected_bs]["traffic"] if self.graph.nodes[connected_bs]["load"] > 1 else 1

        return channel_capacity * factor

    def get_total_power(self, bs_node_id: str):
        bs_type = self.graph.nodes[bs_node_id]["type"]

        if self.graph.nodes.data()[bs_node_id]["active"] == False:
            return self.network_node_params[bs_type]["standby_power"]

        fix_power = self.network_node_params[bs_type]["static_power"]
        antennas = self.graph.nodes.data()[bs_node_id]["traffic"] if self.graph.nodes.data()[
            bs_node_id]["load"] <= 1 else self.network_node_params[bs_type]["antennas"]
        transmission_chain_power = self.network_node_params[bs_type]["tx_power"] * antennas
        encoding_decoding_power = 0
        total_power = fix_power + transmission_chain_power + encoding_decoding_power

        return round(total_power, 4)

    def get_sinr(self, cell_id: str):

        # calculate interfering signals for cell_id
        # only bs nodes in range of cell have edge -> so no filtering required
        interference_signal = 0

        # calculate received signal power from active edge
        connected_bs_id = self.get_connected_bs_nodeid(cell_id)
        if connected_bs_id is None:
            return 0

        for _, bs_in_range_id in enumerate(self.graph[cell_id]):

            for _, interfering_cell_id in enumerate(self.graph[bs_in_range_id]):
                if self.graph[interfering_cell_id][bs_in_range_id]["active"] and interfering_cell_id != cell_id:
                    interference_signal += self.get_rssi_cell(
                        cell_id, (interfering_cell_id, bs_in_range_id))

        received_signal_power = self.get_rssi_cell(cell_id, (cell_id, connected_bs_id))
        return round(received_signal_power / (interference_signal + received_signal_power), 4)

    def get_rssi_cell(self, user_id: str, beam: tuple[str, str]):
        """ returns received signal power for cell with given beam (edge)

        Args:
            beam (tuple[str, str]): beam[0] -> cell_id, beam[1] -> bs_id

            calculate signal in cell from bs station of given beam (taking angle between this beam and
            optimal beam to bs into consideration)
        """
        beam_vector = self.get_directional_vec(target_node_id=beam[0], source_node_id=beam[1])
        optimal_beam_vector = self.get_directional_vec(user_id, beam[1])
        cos_beta = self.vec_cos(beam_vector, optimal_beam_vector)
        bs_type = self.graph.nodes[beam[1]]["type"]
        if cos_beta <= 0 or cos_beta > 1:
            return 0
        wave_length = self.network_node_params[bs_type]["wave_length"]
        transmission_power = self.network_node_params[bs_type]["tx_power"]
        distance = self.get_euclidean_distance(
            (self.graph.nodes.data()[user_id]["pos_x"],
             self.graph.nodes.data()[user_id]["pos_y"]),
            (self.graph.nodes.data()[beam[1]]["pos_x"],
             self.graph.nodes.data()[beam[1]]["pos_y"]))

        return round(transmission_power * cos_beta * math.pow((wave_length / (4*math.pi*distance)), 2), 4)

    def get_euclidean_distance(self, pos1: tuple[float, float], pos2: tuple[float, float]):
        return round(math.dist(pos1, pos2), 4)

    def vec_length(self, v):
        return round(math.sqrt(np.dot(v, v)), 4)

    def vec_cos(self, a, b):
        return round(np.dot(a, b)/(self.vec_length(a)*self.vec_length(b)), 4)

    def get_directional_vec(self, target_node_id: str, source_node_id: str):
        return np.array(
            [self.graph.nodes.data()[target_node_id]["pos_x"] - self.graph.nodes.data()[source_node_id]
             ["pos_x"],
             self.graph.nodes.data()[
                target_node_id]["pos_y"] - self.graph.nodes.data()[source_node_id]
             ["pos_y"]])

    ########################## network filter and sort helpers ###################

    def get_connected_bs_nodeid(self, cell_node_id) -> str | None:
        for _, bs_node in enumerate(self.graph[cell_node_id].items()):
            if bs_node[1]["active"] == True:
                return bs_node[0]
        return None

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

    def filter_user_nodes(self, node):
        return node[1]["type"] == "cell"

    def filter_edges_after_cell_name(self, edge, name: str):
        return edge[0] == name

    def sort_cells_after(self, cell_node,
                         cell_order_two: CellOrderTwo):

        if cell_order_two != CellOrderTwo.RANDOM:
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
            # load_sum += (bs_node[1]["traffic"] / bs_node[1]["antennas"])
            load_sum += (bs_node[1]
                         ["traffic"] / self.network_node_params[bs_node[1]["type"]]["antennas"])
            count += 1

        return round(load_sum / count, 4)

    def get_avg_overlad(self):
        """returns accumulated avg overload over all base stations in %
        Returns:
            avg total network overload as float
        """
        overload_sum = 0
        traffic_capacity_for_overload_bs_sum = 0
        for _, bs_node in enumerate(filter(self.filter_active_bs_nodes, self.graph.nodes.data())):
            if bs_node[1]["traffic"] > self.network_node_params[bs_node[1]["type"]]["antennas"]:
                overload_sum += (bs_node[1]["traffic"])
                traffic_capacity_for_overload_bs_sum += self.network_node_params[bs_node[1]
                                                                                 ["type"]][
                    "antennas"]

        return round(overload_sum / traffic_capacity_for_overload_bs_sum, 4)

    def get_total_energy_consumption(self):
        """get total energy consumption of base stations, including dynamic and static power consumption

        Returns:
            sum of all energy consumptions for all base stations
        """
        energy_consumption = 0
        for _, bs_node in enumerate(filter(self.filter_active_bs_nodes, self.graph.nodes.data())):
            energy_consumption += bs_node[1]["total_power"]

        return round(energy_consumption, 4)

    def get_average_sinr(self):
        """get average sinr over all cells

        Returns:
            average sinr over all cells
        """
        sinr_sum = 0
        count = 0
        for _, cell_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
            sinr_sum += cell_node[1]["sinr"]
            count += 1
        return round(sinr_sum/count, 4)

    def get_average_rssi(self):
        """get average rssi over all cells

        Returns:
            average rssi over all cells
        """
        rssi_sum = 0
        count = 0
        for _, cell_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
            rssi_sum += cell_node[1]["rssi"]
            count += 1

        return round(rssi_sum/count, 4)

    def get_average_dl_datarate(self):
        """get average download datarate over all cells

        Returns:
            average download datarate over all cells
        """
        dl_datarate_sum = 0
        count = 0
        for _, cell_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
            dl_datarate_sum += cell_node[1]["dl_datarate"]
            count += 1

        return round(dl_datarate_sum/count, 4)

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
                if self.graph.nodes[neighbor[0]]["load"] < 1:
                    open_bin_bs_list.append(neighbor[0])

            # find best fitting bs for current cell traffic and rssi
            if len(open_bin_bs_list) > 0:

                for bs_name in open_bin_bs_list:
                    free_antennas_with_new_user = self.graph.nodes[bs_name]["antennas"] - (
                        self.graph.nodes[bs_name]["traffic"] + 1)

                    if best_fit_capacity is None or (
                            free_antennas_with_new_user >= 0 and free_antennas_with_new_user <
                            best_fit_capacity):
                        best_fit_bs_name = bs_name
                        best_fit_capacity = free_antennas_with_new_user
                        best_fit_sinr = self.graph[cell_node[0]][bs_name]["rssi"]
                    elif free_antennas_with_new_user >= 0 and free_antennas_with_new_user == best_fit_capacity and self.graph[cell_node[0]][bs_name]["rssi"] > best_fit_sinr:
                        best_fit_bs_name = bs_name
                        best_fit_capacity = free_antennas_with_new_user
                        best_fit_sinr = self.graph[cell_node[0]][bs_name]["rssi"]
        # if bs_order is given -> iterate through bs types
        else:
            for bs_type in bs_order:
                open_bin_bs_list = []
                for _, neighbor in enumerate(self.graph[cell_node[0]].items()):
                    # if bs still has capacity and is of type bs_type -> add to open_bin_bs_list
                    if self.graph.nodes[neighbor[0]]["load"] < 1 and self.graph.nodes[neighbor[0]][
                            "type"] == bs_type.value:
                        open_bin_bs_list.append(neighbor[0])

                # find best fitting bs for current cell traffic and rssi
                if len(open_bin_bs_list) > 0:

                    for bs_name in open_bin_bs_list:
                        free_antennas_with_new_user = self.network_node_params[self.graph.nodes[bs_name]["type"]]["antennas"] - (
                            self.graph.nodes[bs_name]["traffic"] + 1)

                        if best_fit_capacity is None or (
                                free_antennas_with_new_user >= 0 and free_antennas_with_new_user <
                                best_fit_capacity):
                            best_fit_bs_name = bs_name
                            best_fit_capacity = free_antennas_with_new_user
                            best_fit_sinr = self.graph[cell_node[0]][bs_name]["rssi"]
                        elif free_antennas_with_new_user >= 0 and free_antennas_with_new_user == best_fit_capacity and self.graph[cell_node[0]][bs_name]["rssi"] > best_fit_sinr:
                            best_fit_bs_name = bs_name
                            best_fit_capacity = free_antennas_with_new_user
                            best_fit_sinr = self.graph[cell_node[0]][bs_name]["rssi"]

                    # leave loop if cell is found for desired bs_type
                    if best_fit_bs_name is not None:
                        break
        # if no bs with open capacity, take the bs with lowest overload
        if best_fit_capacity is None or best_fit_capacity < 0:
            current_overload = None
            for _, neighbor in enumerate(self.graph[cell_node[0]].items()):
                antennas = self.network_node_params[self.graph.nodes[neighbor[0]]["type"]][
                    "antennas"]
                if current_overload is None or antennas - self.graph.nodes[neighbor[0]][
                        "traffic"] > current_overload:
                    current_overload = antennas - self.graph.nodes[neighbor[0]]["traffic"]
                    best_fit_bs_name = neighbor[0]
                    best_fit_sinr = self.graph[cell_node[0]][neighbor[0]]["rssi"]
                elif antennas - self.graph.nodes[neighbor[0]][
                        "traffic"] == current_overload and self.graph[cell_node[0]][neighbor[0]]["rssi"] > best_fit_sinr:
                    current_overload = antennas - self.graph.nodes[neighbor[0]]["traffic"]
                    best_fit_bs_name = neighbor[0]
                    best_fit_sinr = self.graph[cell_node[0]][neighbor[0]]["rssi"]

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
                if self.graph.nodes[neighbor[0]]["load"] < 1:
                    open_bin_bs_list.append(neighbor[0])

            # find best fitting bs for current cell traffic and sinr
            if len(open_bin_bs_list) > 0:

                for bs_name in open_bin_bs_list:
                    antennas = self.network_node_params[self.graph.nodes[bs_name]["type"]][
                        "antennas"]
                    free_antennas_plus_new_user = antennas - (
                        self.graph.nodes[bs_name]["traffic"] + 1)
                    if worst_fit_capacity is None or (
                            free_antennas_plus_new_user >= 0 and free_antennas_plus_new_user >
                            worst_fit_capacity):
                        worst_fit_bs_name = bs_name
                        worst_fit_capacity = free_antennas_plus_new_user
                        worst_fit_sinr = self.graph[cell_node[0]][bs_name]["rssi"]
                    elif free_antennas_plus_new_user >= 0 and free_antennas_plus_new_user == worst_fit_capacity and self.graph[cell_node[0]][bs_name]["rssi"] > worst_fit_sinr:
                        worst_fit_bs_name = bs_name
                        worst_fit_capacity = free_antennas_plus_new_user
                        worst_fit_sinr = self.graph[cell_node[0]][bs_name]["rssi"]
        # if bs_order is given -> iterate through bs types
        else:
            for bs_type in bs_order:
                open_bin_bs_list = []
                for _, neighbor in enumerate(self.graph[cell_node[0]].items()):
                    # if bs still has capacity and is of type bs_type -> add to open_bin_bs_list
                    if self.graph.nodes[neighbor[0]]["load"] < 1 and self.graph.nodes[neighbor[0]][
                            "type"] == bs_type.value:
                        open_bin_bs_list.append(neighbor[0])

                # find best fitting bs for current cell traffic and sinr
                if len(open_bin_bs_list) > 0:

                    for bs_name in open_bin_bs_list:
                        antennas = self.network_node_params[self.graph.nodes[bs_name]["type"]][
                            "antennas"]
                        free_antennas_plus_new_user = antennas - (
                            self.graph.nodes[bs_name]["traffic"] + 1)

                        if worst_fit_capacity is None or (
                                free_antennas_plus_new_user >= 0 and free_antennas_plus_new_user >
                                worst_fit_capacity):
                            worst_fit_bs_name = bs_name
                            worst_fit_capacity = free_antennas_plus_new_user
                            worst_fit_sinr = self.graph[cell_node[0]][bs_name]["rssi"]
                        elif free_antennas_plus_new_user >= 0 and free_antennas_plus_new_user == worst_fit_capacity and self.graph[cell_node[0]][bs_name]["rssi"] > worst_fit_sinr:
                            worst_fit_bs_name = bs_name
                            worst_fit_capacity = free_antennas_plus_new_user
                            worst_fit_sinr = self.graph[cell_node[0]][bs_name]["rssi"]

                    # leave loop if cell is found for desired bs_type
                    if worst_fit_bs_name is not None:
                        break
        # if no bs with open capacity, take the bs with lowest overload
        if worst_fit_capacity is None or worst_fit_capacity < 0:
            current_overload = None
            for _, neighbor in enumerate(self.graph[cell_node[0]].items()):
                antennas = self.network_node_params[self.graph.nodes[neighbor[0]]["type"]][
                    "antennas"]
                if current_overload is None or antennas - self.graph.nodes[neighbor[0]][
                        "traffic"] > current_overload:
                    current_overload = self.graph.nodes[neighbor[0]
                                                        ]["antennas"] - self.graph.nodes[neighbor[0]]["traffic"]
                    worst_fit_bs_name = neighbor[0]
                    worst_fit_sinr = self.graph[cell_node[0]][neighbor[0]]["rssi"]
                elif antennas - self.graph.nodes[neighbor[0]][
                        "traffic"] == current_overload and self.graph[cell_node[0]][neighbor[0]]["rssi"] > worst_fit_sinr:
                    current_overload = antennas - self.graph.nodes[neighbor[0]]["traffic"]
                    worst_fit_bs_name = neighbor[0]
                    worst_fit_sinr = self.graph[cell_node[0]][neighbor[0]]["rssi"]

        # activate best-fit edge and update node attributes
        self.set_edge_active(
            cell_node_name=cell_node[0],
            bs_node_name=worst_fit_bs_name, active=True)

    def find_activation_profile_bin_packing(
            self, cell_order_2: CellOrderTwo = CellOrderTwo.RANDOM, bs_order: list
            [BaseStationOrder] | None = None, bin_packing: BinPackingType = BinPackingType.BEST_FIT):

        # deactivate all edges and update all node attributes
        self.set_edges_active(False)

        # prepare cell iteration order
        cell_node_list = list(filter(self.filter_user_nodes, self.graph.nodes.data()))
        cell_node_list.sort(key=lambda x: self.sort_cells_after(x, cell_order_two=cell_order_2))

        # perform bin packing for each cell and activate single edges
        for _, cell_node in enumerate(cell_node_list):
            if bin_packing == BinPackingType.BEST_FIT:
                self.best_fit(cell_node, bs_order=bs_order)
            elif bin_packing == BinPackingType.WORST_FIT:
                self.worst_fit(cell_node, bs_order=bs_order)

    ############### get encodings, apply encodings, save/load encodings ###################

    def get_json_adjacency_graph(self):
        return json.dumps(json_graph.adjacency.adjacency_data(self.graph))

    def save_json_adjacency_graph_to_file(self, filename: str):
        with open(filename, "w+", encoding="utf-8") as outfile:
            outfile.write(self.get_json_adjacency_graph())

    def get_graph_from_json_adjacency_string(self, json_string) -> nx.Graph:
        return json_graph.adjacency.adjacency_graph(json_string)

    def load_graph_from_json_adjacency_file(
            self, file_name: str, keep_activation_profile: bool = False):
        # Opening JSON file
        with open(file_name, 'r', encoding="utf-8") as openfile:
            # Reading from json file
            json_object = json.load(openfile)
            self.graph = self.get_graph_from_json_adjacency_string(json_object)
            if keep_activation_profile is False:
                self.initialize_edges()
            self.update_network_attributes()

    def load_parameter_config_from_json_file(
            self, file_name: str):
        # Opening JSON file
        with open(file_name, 'r', encoding="utf-8") as openfile:
            # Reading from json file
            json_object = json.load(openfile)
            self.apply_network_node_attributes(json_object)

    def get_edge_activation_encoding_from_graph(self):
        """get the current activation profile of the network edges as list decoding

        Returns:
            list of binary str with each str representing activation profile of base stations for
            each cell
        """
        cell_encoding_list: list[str] = []
        for _, cell_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
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
        for user_node_index, user_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
            for edge_index, edge in enumerate(self.graph[user_node[0]].items()):
                if encoding[user_node_index][edge_index] == "1":
                    self.set_edge_active(user_node[0], edge[0], True)
                    break
                else:
                    self.set_edge_active(user_node[0], edge[0], False)

    def save_edge_activation_profile_to_file(
            self, encoding: list[list[str]], result_file_name: str,
            result_sheet_name: str):
        df = pd.DataFrame(encoding)

        with pd.ExcelWriter(result_file_name, mode="w", if_sheet_exists="replace") as writer:  # pylint: disable=abstract-class-instantiated
            df.to_excel(writer, result_sheet_name, header=False, index=False)

    def get_edge_activation_encoding_from_file(self, result_file_name: str, result_sheet_name: str):
        df = pd.read_excel(result_file_name, result_sheet_name,
                           dtype=object, header=None, index_col=None)
        encodings: list[list[str]] = df.values.tolist()
        return encodings

    ################ start calculation methods without visualization and with saving the results ############

    def start_calculation_bin_packing_save(
            self, result_file_name: str, sheet_name: str,
            cell_order_2: CellOrderTwo = CellOrderTwo.LOWEST_DEGREE_FIRST, bs_order: None |
            list[BaseStationOrder] = None, bin_packing: BinPackingType = BinPackingType.BEST_FIT):
        """
        calls bin packing for current network and saves encoding to file
        """
        result: list[list[str]] = []

        self.find_activation_profile_bin_packing(cell_order_2=cell_order_2, bs_order=bs_order,
                                                 bin_packing=bin_packing)
        result.append(self.get_edge_activation_encoding_from_graph())

        self.save_edge_activation_profile_to_file(result, result_file_name, sheet_name)

    ############## visualization methods ####################

    def get_node_pos(self):
        """get node positions as dict

        Returns:
            pos (dict[str, list[int]]): as dict of pos-list for each node
        """
        pos: dict[str, list[int]] = {}

        for _, node in enumerate(self.graph.nodes.data()):
            pos[str(node[0])] = [node[1]["pos_x"], node[1]["pos_y"]]
        return pos

    def matplotlib_animate_from_encoding(
            self, i, fig: Figure, pos, activation_matrix: list[list[str]],
            fig_title: str = ""):
        fig.clear()
        plt.title(str(i+1) + "'s solution" + " - " + fig_title)

        # apply activation profile -> update network
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

    def start_matplotlib_animation_result_from_sheet(self, file_name: str, sheet_name: str):
        """start simulation over pareto front  with visualization
        TODO finish doc
        """
        # load activation profile from sheet_name
        activation_matrix = self.get_edge_activation_encoding_from_file(file_name,
                                                                        sheet_name)

        pos = self.get_node_pos()
        fig, _ = plt.subplots(figsize=(10, 6))

        anim = animation.FuncAnimation(
            fig, lambda i: self.matplotlib_animate_from_encoding(
                i, fig=fig, pos=pos, activation_matrix=activation_matrix, fig_title=sheet_name),
            frames=len(activation_matrix), interval=1000, repeat=True)
        # for showing animated plot in jupyter notebook
        # display(HTML(anim.to_jshtml()))
        # plt.close(fig)
        # for showing plot in normal script mode
        plt.show()

    def draw_current_network(self):
        pos = self.get_node_pos()
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
            "label": ["hallo", "hallo", "hallo"],
        }

        label_dic = {}
        for _, user_node in enumerate(filter(self.filter_user_nodes, self.graph.nodes.data())):
            label_dic[user_node[0]] = user_node[1]["rssi"]

        for _, bs_node in enumerate(filter(self.filter_bs_nodes, self.graph.nodes.data())):
            label_dic[bs_node[0]] = str(bs_node[1]["load"]) + "%"

        label_positions: dict[str, list[int]] = {}

        for _, pos_item in enumerate(pos.items()):
            label_positions[str(pos_item[0])] = [pos_item[1][0], pos_item[1][1] + 1]

        plt.title(f"network")
        nx.drawing.nx_pylab.draw_networkx_edges(self.graph, **options_edges)
        nx.drawing.nx_pylab.draw_networkx_nodes(self.graph, **options_nodes)
        nx.drawing.nx_pylab.draw_networkx_labels(
            self.graph, label_positions, label_dic, font_color="black")
        plt.show()

    def draw_measurement_diagrams(self, result_file_name: str, result_sheet_name_list: list[str]):

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
            # load activation profile for current network layout from file (pareto front solutions)
            activation_profile_matrix = self.get_edge_activation_encoding_from_file(
                result_file_name,
                sheet_name)

            overload_list = []
            total_power_list = []
            avg_load_list = []
            avg_sinr_list = []
            avg_rssi_list = []
            avg_dl_datarate_list = []

            for _, current_activation_profile in enumerate(activation_profile_matrix):

                # apply activation profile and update network
                self.apply_edge_activation_encoding_to_graph(current_activation_profile)
                # append avg_sinr to avg_sinr_list for sheet_name
                avg_sinr_list.append(self.get_average_sinr())
                avg_load_list.append(self.get_average_network_load())
                overload_list.append(self.get_avg_overlad())
                total_power_list.append(self.get_total_energy_consumption())
                avg_rssi_list.append(self.get_average_rssi())
                avg_dl_datarate_list.append(self.get_average_dl_datarate())
            # insert avg_sinr_list dict for plotting
            avg_sinr_values_dict[sheet_name] = avg_sinr_list
            overload_values_dict[sheet_name] = overload_list

            energy_values_dict[sheet_name] = total_power_list
            avg_load_values_dict[sheet_name] = avg_load_list

        width = 1/(len(result_sheet_name_list)+1)  # the width of the bars
        multiplier = 0
        for sheet_name, avg_sinr_24_hour_list in avg_sinr_values_dict.items():
            x = np.arange(1, len(avg_sinr_24_hour_list)+1)
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
    son = Son(adjacencies_file_name="test.json")
    # bestFit_highestTrafficFirst_HighestDegreeFirst_bsOrderNone
    # son.start_calculation_bin_packing_save(
    #     sheet_name="bf_htf_hdf_bsn",
    #     cell_order_2=CellOrderTwo.HIGHEST_DEGREE_FIRST, bs_order=None,
    #     bin_packing=BinPackingType.BEST_FIT)

    # worstFit_highestTrafficFirst_HighestDegreeFirst_bsOrderNone

    # son.start_calculation_bin_packing_save(
    #     sheet_name="wf_htf_tdf_bsn",
    #     cell_order_2=CellOrderTwo.HIGHEST_DEGREE_FIRST, bs_order=None,
    #     bin_packing=BinPackingType.WORST_FIT)

    son.find_activation_profile_bin_packing(
        cell_order_2=CellOrderTwo.HIGHEST_DEGREE_FIRST, bs_order=None,
        bin_packing=BinPackingType.BEST_FIT)

    # son.start_matplotlib_animation_result_from_sheet("son_input.xlsx", "wf_htf_tdf_bsn")

    # draw diagrams
    # son.draw_measurement_diagrams(
    #     ["wf_htf_tdf_bsn"])

    # son.save_json_adjacency_graph_to_file("test.json")
    # son.load_graph_from_json_adjacency_file("test.json")
    son.draw_current_network()


def animate_result(file: str, sheet: str):
    son = Son(adjacencies_file_name="test.json")
    son.start_matplotlib_animation_result_from_sheet(file, sheet)


if __name__ == "__main__":
    main()
