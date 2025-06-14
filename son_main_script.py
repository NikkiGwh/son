import json
import math
from enum import Enum
from networkx.readwrite import json_graph
import networkx as nx
import numpy as np
from sklearn.metrics import max_error
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
    "tx_power": 31.622776601683793, # i n watts
    "static_power": 450, # in watts
    "standby_power": 0,
    "antennas": 16,
    "frequency": 15.0, # in GHz
    "wave_length": 0.019986163866666667, # in m
    "channel_bandwidth": 80000000 # in Hz
  },
  "micro": {
    "type": "micro",
    "tx_power": 19.95262314968879,
    "static_power": 100,
    "standby_power": 0,
    "antennas": 4,
    "frequency": 26.0,
    "wave_length": 0.011530479153846154,
    "channel_bandwidth": 400000000
  },
  "femto": {
    "type": "micro",
    "tx_power": 31.622776601683793,
    "static_power": 100,
    "standby_power": 0,
    "antennas": 4,
    "frequency": 20,
    "wave_length": 0.01498962,
    "channel_bandwidth": 400000000
  },
  "pico": {
    "type": "micro",
    "tx_power": 31.622776601683793,
    "static_power": 100,
    "standby_power": 0,
    "antennas": 4,
    "frequency": 20,
    "wave_length": 0.01498962,
    "channel_bandwidth": 400000000
  },
    "cell": {
        "type": "cell",
        "shadow_component": 0,
        "noise": 0
    }
}

class Son:
    def __init__(self, adjacencies_file_name: str = "", parameter_config_file_name: str = "") -> None:

        # this is -80 dbm
        self.min_rssi = 1e-11
        self.max_cos = math.cos(math.pi / 4)
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
        # only add edges for cell-bs pairs which have minimum rssi signal strength
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

    ###################### network update methods ###################

    def update_network_attributes(self):
        """ update all dynamic node and edge attributes in the right order
            important !! before that you have set activations for all the edges 
        """

        self.update_bs_node_attributes()
        self.update_user_node_attributes()
        # self.update_user_node_dl_datarate()
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
            count_connected_users = len(connected_users)


            bs_node[1]["active"] = True if count_connected_users > 0 else False
            bs_node[1]["load"] = 0
             # set traffic attribute
            bs_node[1]["traffic"] = count_connected_users
            # set load attribute
            bs_node[1]["load"] = count_connected_users / self.network_node_params[bs_node[1]["type"]]["antennas"]
            # calculate total power consumption for bs -> call only after traffic and load estimations are done
            bs_node[1]["total_power"] = self.get_total_bs_power(bs_node[0]) if count_connected_users > 0 else self.network_node_params[bs_node[1]["type"]]["standby_power"]
            

    def update_user_node_attributes(self):
        """ update dynamic user_node attributes depending on active edge (sinr, rssi)
        no dl_datarate
        """

        for _, user_node in enumerate(filter(self.filter_user_nodes, self.graph.nodes.data())):
            user_node[1]["sinr"] = self.get_sinr(user_node[0])
            connected_bs = self.get_connected_bs_nodeid(user_node[0])
            user_node[1]["rssi"] = self.get_rssi_cell(user_node[0], (user_node[0], connected_bs)) if connected_bs is not None else 0
            user_node[1]["dl_datarate"] = self.get_dl_datarate_user(user_node[0])

    # def update_user_node_dl_datarate(self):
    #     """updates downlink data rate for users -> call after update_user_nodes and update_bs_nodes"""
    #     for _, user_node in enumerate(filter(self.filter_user_nodes, self.graph.nodes.data())):
    #         user_node[1]["dl_datarate"] = self.get_dl_datarate_user(user_node[0])
    

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

    def get_dl_datarate_user(self, user_id: str):
        """
        call after bs_nodes and user_nodes were updated
        """
        connected_bs = self.get_connected_bs_nodeid(user_id)
        if (connected_bs is None):
            return 0
        connected_bs_type = self.graph.nodes[connected_bs]["type"]
        channel_bandwidth = self.network_node_params[connected_bs_type]["channel_bandwidth"]
        sinr = self.graph.nodes[user_id]["sinr"]
        channel_capacity = channel_bandwidth * math.log(1+sinr, 2)
        factor = 1 / self.graph.nodes[connected_bs]["load"] if self.graph.nodes[connected_bs]["load"] > 1 else 1
       
        # wave_length = self.network_node_params[connected_bs_type]["wave_length"]
        distance = self.graph.nodes[user_id]["distance"]
        # distance_factor = math.pow((wave_length / (4*math.pi*distance)), 2)
        distance_factor = math.exp(-distance/3000)

        return round(channel_capacity * factor * distance_factor, 6)

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
        bs_total_power_consumption = self.graph.nodes[bs_node_id]["total_power"]

        user_dl_datarate_sum = 0

        for _, edge in enumerate(self.graph[bs_node_id].items()):
            if edge[1]["active"]:
                user_dl_datarate_sum +=  self.get_dl_datarate_user(edge[0])

        return user_dl_datarate_sum / bs_total_power_consumption

    def get_sinr(self, cell_id: str):

        # only calculate if the activation profile for the whole network was set before
        # calculate interfering signals for cell_id
        # only bs nodes in range of cell have edge -> so no filtering required
        # only bs nodes which transmitt at same frequency (same type) interfer each others signal
        
        # calculate received signal power from active edge
        connected_bs_id = self.get_connected_bs_nodeid(cell_id)
        if connected_bs_id is None:
            return 0
        interference_signal = 0
        for _, bs_in_range_id in enumerate(self.graph[cell_id]):
            if self.graph.nodes[bs_in_range_id]["type"] == self.graph.nodes[connected_bs_id]["type"] and bs_in_range_id != connected_bs_id :
                for _, interfering_cell_id in enumerate(self.graph[bs_in_range_id]):
                    if self.graph[interfering_cell_id][bs_in_range_id]["active"] and interfering_cell_id != cell_id:
                        interference_signal += self.get_rssi_cell(
                            cell_id, (interfering_cell_id, bs_in_range_id))

        received_signal_power = self.get_rssi_cell(cell_id, (cell_id, connected_bs_id))

        return received_signal_power / (received_signal_power + interference_signal)

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
        if cos_beta <= 0 or cos_beta > self.max_cos and beam[0] != user_id :
            return 0
        wave_length = self.network_node_params[bs_type]["wave_length"]
        transmission_power = self.network_node_params[bs_type]["tx_power"]
        distance = self.get_euclidean_distance(
            (self.graph.nodes.data()[user_id]["pos_x"] + moving_vector[0],
             self.graph.nodes.data()[user_id]["pos_y"] + moving_vector[1]),
            (self.graph.nodes.data()[beam[1]]["pos_x"],
             self.graph.nodes.data()[beam[1]]["pos_y"]))
        self.graph.nodes[user_id]["distance"] = distance
        if distance == 0:
            # beacuse otherwise programm crashes if user is exactly on topof bs station
            return 0
        else:
            result_rssi = transmission_power * cos_beta * math.pow((wave_length / (4*math.pi*distance)), 2)

        return result_rssi
    
    def get_bs_type_range(self, bs_type: str):
        wave_length = self.network_node_params[bs_type]["wave_length"]
        transmission_power = self.network_node_params[bs_type]["tx_power"]

        return wave_length / (4 * math.pi * (math.sqrt(self.min_rssi / transmission_power)))

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

        return round(energy_consumption, 4)

    def get_total_energy_efficiency(self):
        """get energy efficiency over all base stations, including dynamic, static and standby power consumption

        Returns:
           energy efficiency of base stations
        """
        power_consumption_sum = self.get_total_energy_consumption()

        dl_datarate_sum =  0
        for _, user_node in enumerate(filter(self.filter_user_nodes, self.graph.nodes.data())):
            dl_datarate_sum += self.get_dl_datarate_user( user_node[0])

        if dl_datarate_sum <= 0 or power_consumption_sum <= 0:
            return 0
        return dl_datarate_sum / power_consumption_sum
   
    def get_avg_energy_efficiency(self):
        """get average energy efficiency over all active base stations

        Returns:
           average energy efficiency over active base stations
        """

        energy_efficiencies_sum =  0
        active_bs_count = 0
        for _, bs_node in enumerate(filter(self.filter_active_bs_nodes, self.graph.nodes.data())):
            energy_efficiencies_sum += self.get_energy_efficiency_bs(bs_node[0])
            active_bs_count+=1

        return energy_efficiencies_sum / active_bs_count if energy_efficiencies_sum != 0 and active_bs_count != 0 else 0

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
        """get average rssi over all users

        Returns:
            average rssi over all users
        """
        rssi_sum = 0
        count = 0
        for _, cell_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
            rssi_sum += cell_node[1]["rssi"]
            count += 1
        return rssi_sum/count

    def get_average_dl_datarate_and_variance(self):
        """get average download datarate over all users
        only call this after all node attributes were updated
        Returns:
            tuple (average download datarate over all users, variance)
        """
        dl_datarate_sum = 0
        dl_rate_values = []
        count = 0
        for _, cell_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
            dl_datarate_sum += cell_node[1]["dl_datarate"]
            dl_rate_values.append(cell_node[1]["dl_datarate"])
            count += 1
        avg_dl_datarate = dl_datarate_sum/count
       
        squared_difference = 0
        for _, cell_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
            squared_difference += (avg_dl_datarate - cell_node[1]["dl_datarate"])**2
        
        variance_normal = 0
        if avg_dl_datarate != 0:
            variance_normal = squared_difference / avg_dl_datarate
       
        q75, q50, q25 = np.percentile(dl_rate_values, [75 ,50, 25], method="median_unbiased")
        iqr_result = q75 - q25
        max_dl_rate = np.amax(dl_rate_values)
        min_dl_rate = np.amin(dl_rate_values)
        return (avg_dl_datarate, variance_normal, iqr_result, q25, q50, q75, min_dl_rate, max_dl_rate)
    
    def get_dl_datarate_variance(self):
        """ returns the variance of the dl_datarate over all user devices
        only call this after all node attributes were updated
        Returns:
            varaince of the dl_datarate over all user devices
        """
        dl_datarate_sum = 0
        count = 0
        for _, cell_node in enumerate(
                filter(self.filter_user_nodes, self.graph.nodes.data())):
            dl_datarate_sum += cell_node[1]["dl_datarate"]
            count += 1

        return dl_datarate_sum/count

    
    def get_average_userNode_degree(self):
        #TODO check if this is right
        return len(self.graph.edges)/len(list(filter(self.filter_user_nodes, self.graph.nodes.data())))
    
    def get_total_UserNode_edges(self):
        return len(self.graph.edges) / 2

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

    def apply_activation_profile_greedy_user(self, update_attributes=False):
        """iterrates through all user nodes, applies greedy assignments returns 
        activation profile dict
        """
        activation_dict: dict[str, str] = {}
        for _, user_node in enumerate(filter(self.filter_user_nodes, self.graph.nodes.data())):
            best_bs_id = self.greedy_assign_user_to_bs(user_node[0], set_edge_activation=True)
            activation_dict[user_node[0]] = best_bs_id

        if (update_attributes):
            self.update_network_attributes()
        return activation_dict
    
    def find_activation_profile_greedy_user(self):
        """finds greedy activation profile for this network but doesn't apply it
        """
        activation_dict: dict[str, str] = {}
        for _, user_node in enumerate(filter(self.filter_user_nodes, self.graph.nodes.data())):
            best_bs_id = self.greedy_assign_user_to_bs(user_node[0], set_edge_activation=False)
            activation_dict[user_node[0]] = best_bs_id
            
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
            update_network_attributes=True, greedy_assign_list=[]):
        """takes activation dict  and applies it on network accordingly
        and also repairs encoding if there are any violations against the current topology
        important !! first assign activations to edges, thatn update_network_attributes()

        Args:
            encoding (dict[str, str]]): activation dict of cell-name to connected bs-name
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


################# script main #######################

def main():
    print("son_main_script main()")
    # nx.write_gml(son.graph, "graph.gml")
    # nx.write_graphml_lxml(son.graph, "graph.graphml")

if __name__ == "__main__":
    main()
