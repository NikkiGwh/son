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
        "tx_power": 31.622776601683793,
        "static_power": 4.0,
        "standby_power": 2.0,
        "antennas": 8,
        "frequency": 6.0,
        "wave_length": 0.04996540966666666,
        "channel_bandwidth": 10
    },
    "micro": {
        "type": "micro",
        "tx_power": 3.162277660168379,
        "static_power": 1.5,
        "standby_power": 0.75,
        "antennas": 3,
        "frequency": 16,
        "wave_length": 0.018737028625,
        "channel_bandwidth": 10
    },
    "femto": {
        "type": "femto",
        "tx_power": 0.1,
        "static_power": 4.0,
        "standby_power": 0,
        "antennas": 4,
        "frequency": 100,
        "wave_length": 0.00299792,
        "channel_bandwidth": 10
    },
    "pico": {
        "type": "pico",
        "tx_power": 0.25118864315095796,
        "static_power": 4.0,
        "standby_power": 0,
        "antennas": 4,
        "frequency": 100,
        "wave_length": 0.00299792,
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
        # this is -80 dbm
        self.min_rssi = 1.0000000000000001e-11
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
        for _, cell in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
            for _, bs_node in enumerate(
                    filter(self.filter_bs_nodes, self.graph.nodes.data())):
                current_rssi = self.get_rssi_cell(cell[0], (cell[0], bs_node[0]))
                current_distance = self.get_euclidean_distance(
                    (cell[1]["pos_x"], cell[1]["pos_y"]), (bs_node[1]["pos_x"], bs_node[1]["pos_y"]))
                if current_rssi >= self.min_rssi:
                    attribute_dic = {}
                    attribute_dic["rssi"] = current_rssi
                    attribute_dic["distance"] = current_distance
                    attribute_dic["active"] = False
                    edge_list_with_attributes.append(
                        (cell[0], bs_node[0], attribute_dic))
        self.graph.add_edges_from(edge_list_with_attributes)

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
        # self.update_edge_attributes()

    def update_bs_node_attributes(self):
        """update dynamic bs_node attributes (traffic, total_power, active)
        also deactivate (standby) bs stations which don't have active edges
        """
        # deactivate bs stations which don't have active edges and activate those which have active connections
        # set traffic, total_power to 0, total_power to standby for inactive bs nodes
        for _, bs_node in enumerate(
                filter(self.filter_bs_nodes, self.graph.nodes.data())):

            connected_users = [x[0] for x in self.graph[bs_node[0]].items() if x[1]["active"]]
            if len(connected_users) == 0:
                # calculation for inactive users
                bs_node[1]["active"] = False
                self.graph.nodes[bs_node[0]]["total_power"] = self.network_node_params[bs_node[1][
                    "type"]]["standby_power"]
               # self.graph.nodes[bs_node[0]]["energy_efficiency"] = 0
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
            bs_node[1]["total_power"] = self.get_total_bs_power(bs_node[0])

    def update_user_node_attributes(self):
        """ update dynamic user_node attributes depending on active edge (sinr, rssi)
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

    def move_node_by_pos(
            self, node_id: str, pos: tuple[float, float], initialize_edges=True,
            update_network=True):

        self.graph.nodes.data()[node_id]["pos_x"] = round(pos[0], 8)
        self.graph.nodes.data()[node_id]["pos_y"] = round(pos[1], 8)

        if initialize_edges:
            self.initialize_edges()

        if update_network:
            self.update_network_attributes()

    def move_node_by_vec(
            self, node_id: str, vec: tuple[float, float], initialize_edges=True,
            update_network=True):

        self.graph.nodes.data()[node_id]["pos_x"] = self.graph.nodes.data()[
            node_id]["pos_x"] + vec[0]
        self.graph.nodes.data()[node_id]["pos_y"] = self.graph.nodes.data()[
            node_id]["pos_y"] + vec[1]

        if initialize_edges:
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

    def get_total_bs_power(self, bs_node_id: str):
        bs_type = self.graph.nodes[bs_node_id]["type"]

        if self.graph.nodes.data()[bs_node_id]["active"] == False:
            return self.network_node_params[bs_type]["standby_power"]

        fix_power = self.network_node_params[bs_type]["static_power"]
        beams = self.graph.nodes.data()[bs_node_id]["traffic"] if self.graph.nodes.data()[
            bs_node_id]["load"] <= 1 else self.network_node_params[bs_type]["antennas"]

        transmission_chain_power = self.network_node_params[bs_type]["tx_power"] * beams
        encoding_decoding_power = 0
        total_power = fix_power + transmission_chain_power + encoding_decoding_power

        return total_power

    def get_energy_efficiency_bs(self, bs_node_id: str):
        if self.graph.nodes.data()[bs_node_id]["active"] == False:
            return 0
        bs_total_power_consumption = self.graph.nodes.data()[bs_node_id]["total_power"]

        user_dl_datarate_sum = 0

        for _, edge in enumerate(self.graph[bs_node_id].items()):
            if edge[1]["active"]:
                user_dl_datarate_sum += self.graph.nodes[edge[0]]["dl_datarate"]

        return user_dl_datarate_sum / bs_total_power_consumption

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
        return received_signal_power / (interference_signal + received_signal_power)

    def get_rssi_cell(
            self, user_id: str, beam: tuple[str, str],
            moving_vector: tuple[float, float] = (0.0, 0.0)) -> float:
        """ returns received signal power for cell with given beam (edge)

        Args:
            beam (tuple[str, str]): beam[0] -> cell_id, beam[1] -> bs_id

            calculate signal in cell from bs station of given beam (taking angle between this beam and
            optimal beam to bs into consideration)
        """
        beam_vector = self.get_directional_vec(target_node_id=beam[0], source_node_id=beam[1])
        optimal_beam_vector = self.get_directional_vec(user_id, beam[1])
        cos_beta = 1 if beam[0] == user_id else self.vec_cos(beam_vector, optimal_beam_vector)
        bs_type = self.graph.nodes[beam[1]]["type"]
        if cos_beta <= 0 or cos_beta > 1:
            return 0
        wave_length = self.network_node_params[bs_type]["wave_length"]
        transmission_power = self.network_node_params[bs_type]["tx_power"]
        distance = self.get_euclidean_distance(
            (self.graph.nodes.data()[user_id]["pos_x"] + moving_vector[0],
             self.graph.nodes.data()[user_id]["pos_y"] + moving_vector[1]),
            (self.graph.nodes.data()[beam[1]]["pos_x"],
             self.graph.nodes.data()[beam[1]]["pos_y"]))
        if distance == 0:
            return 0
        else:
            result_rssi = transmission_power * cos_beta * \
                math.pow((wave_length / (4*math.pi*distance)), 2)
        return result_rssi

    def get_euclidean_distance(self, pos1: tuple[float, float], pos2: tuple[float, float]):
        return math.dist(pos1, pos2)

    def vec_length(self, v):
        return math.sqrt(np.dot(v, v))

    def vec_cos(self, a, b):
        return np.dot(a, b)/(self.vec_length(a)*self.vec_length(b))

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
        if count == 0:
            return 0
        return load_sum / count

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

        return overload_sum / traffic_capacity_for_overload_bs_sum

    def get_total_energy_consumption(self):
        """get total energy consumption of base stations, including dynamic and static power consumption

        Returns:
            sum of all energy consumptions for all base stations
        """
        energy_consumption = 0
        for _, bs_node in enumerate(filter(self.filter_active_bs_nodes, self.graph.nodes.data())):
            energy_consumption += bs_node[1]["total_power"]

        return energy_consumption

    def get_energy_efficiency(self):
        """get energy efficiency of all base stations, including dynamic, static and standby power consumption

        Returns:
           energy efficiency of base stations
        """
        power_consumption_sum = self.get_total_energy_consumption()

        avg_dl_datarate = self.get_average_dl_datarate()

        if avg_dl_datarate <= 0 or power_consumption_sum <= 0:
            return 0
        return avg_dl_datarate / power_consumption_sum

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
        return sinr_sum/count

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
        return rssi_sum/count

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

        return dl_datarate_sum/count

    ################################ greedy assingment methods ################################

    def greedy_assign_user_to_bs(
            self, user_id: str, update_network_attributes=False, set_edge_activation=False):
        best_bs_id = ""
        best_rssi = -1
        for _, bs_id in enumerate(self.graph[user_id]):
            if set_edge_activation:
                self.graph[user_id][bs_id]["active"] = False
            if best_rssi < self.get_rssi_cell(user_id, (user_id, bs_id)):
                best_rssi = self.get_rssi_cell(user_id, (user_id, bs_id))
                best_bs_id = bs_id

        if set_edge_activation:
            self.graph[user_id][best_bs_id]["active"] = True

        if update_network_attributes:
            self.update_network_attributes()
        return best_bs_id

    def find_activation_profile_greedy_user(self, update_attributes=False):
        """iterrates through all user nodes, applies greedy assignments returns 
        activation profile dict
        """
        activation_dict: dict[str, str] = {}
        for _, user_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):

            best_bs_id = self.greedy_assign_user_to_bs(user_node[0], set_edge_activation=True)
            activation_dict[user_node[0]] = best_bs_id
        if (update_attributes):
            self.update_network_attributes()
        return activation_dict

    ############### get encodings, apply encodings, save/load encodings and adjacencies ###################

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

    def get_possible_activations_dict(self):
        """returns a dict with user_id's as keys and list of possible bs_id's as values
        useful for creating the encoding for pymoo optimization

        Returns:
            dict with user_id's as keys and list of possible bs_id's as values
        """
        possible_activations_dict: dict[str, list[str]] = {}
        for _, user_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
            possible_activations_dict[user_node[0]] = list(self.graph[user_node[0]])

        return possible_activations_dict

    def get_activation_dict(self):
        """get the current activation profile of the network edges as id list decoding

        Returns:
            dict of all user nodes as keys and the active bs_node ids as values
        """
        activation_dict: dict[str, str] = {}
        for _, user_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
            for _, edge in enumerate(self.graph[user_node[0]].items()):
                if edge[1]["active"]:
                    activation_dict[user_node[0]] = edge[0]
        return activation_dict

    def apply_activation_dict(
            self, activation_dict: dict[str, str],
            update_network_attributes=True, min_rssi=-1.0, greedy_assign_list=[]):
        """takes activation dict  and applies it on network accordingly
        and also repairs encoding if there are any violations against the current topology

        Args:
            encoding (list[int]): activation list of cell edges
        returns:
            repaired activation encoding
        """
        for _, user_id in enumerate(activation_dict):
            if activation_dict[user_id] in self.graph[user_id] and user_id not in greedy_assign_list:
                # apply valid activation
                for _, bs_id in enumerate(self.graph[user_id]):
                    if bs_id == activation_dict[user_id]:
                        self.graph[bs_id][user_id]["active"] = True
                    else:
                        self.graph[bs_id][user_id]["active"] = False
            else:
                # repair invalid acitvation with greedy assignment
                greedy_bs_id = self.greedy_assign_user_to_bs(user_id, set_edge_activation=True)
                activation_dict[user_id] = greedy_bs_id

        if update_network_attributes:
            self.update_network_attributes()

            if min_rssi != -1:
                changed = False
                for _, cell_node in enumerate(
                        list(filter(lambda x: x[1]["type"] == "cell", self.graph.nodes.data()))):
                    if cell_node[1]["rssi"] < min_rssi:
                        # find better rssi
                        greedy_bs_id = self.greedy_assign_user_to_bs(
                            cell_node[0], set_edge_activation=True)
                        activation_dict[cell_node[0]] = greedy_bs_id
                        changed = True
                if changed:
                    self.update_network_attributes()

        # return repaired activation encoding
        return activation_dict

    def valid_edge_activation_profile_encoding(
            self, activation_dict: dict[str, str]):
        """takes activation dict and checks if it violates the current topology

        Args:
            encoding (list[int]): activation list of cell edges

        returns:
            boolean (True) if there is no violation
        """

        for _, user_id in enumerate(activation_dict):
            if activation_dict[user_id] not in self.graph[user_id]:
                return False
        return True
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
        }

        label_dic = {}
        for _, user_node in enumerate(filter(self.filter_user_nodes, self.graph.nodes.data())):
            label_dic[user_node[0]] = user_node[0]

        for _, bs_node in enumerate(filter(self.filter_bs_nodes, self.graph.nodes.data())):
            label_dic[bs_node[0]] = str(bs_node[0])

        label_positions: dict[str, list[int]] = {}

        for _, pos_item in enumerate(pos.items()):
            label_positions[str(pos_item[0])] = [pos_item[1][0], pos_item[1][1] + 1]

        plt.title(f"network")
        nx.drawing.nx_pylab.draw_networkx_edges(self.graph, **options_edges)
        nx.drawing.nx_pylab.draw_networkx_nodes(self.graph, **options_nodes)
        # nx.drawing.nx_pylab.draw_networkx_labels(
        #     self.graph, label_positions, label_dic, font_color="black")
        plt.show()


################# script main #######################

def main():
    #  son = Son(adjacencies_file_name="datastore/hetNet/algorithm_config_3STATIC/ind_result_1.json",
    #           parameter_config_file_name="datastore/hetNet/hetNet_network_config.json")

    # son.draw_current_network()
    print("son_main_script main()")
    # nx.write_gml(son.graph, "graph.gml")
    # nx.write_graphml_lxml(son.graph, "graph.graphml")


if __name__ == "__main__":
    main()
