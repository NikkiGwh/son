from enum import Enum
import math
from copy import deepcopy
import sys
import time
from event import Event
from son_pymoo import AlgorithmEnum, CrossoverEnum, MutationEnum, ObjectiveEnum, RunningMode, SamplingEnum, start_optimization
from scipy.constants import speed_of_light
from son_main_script import Son, default_node_params
import networkx as nx
import numpy as np
import json
import multiprocessing
import os

class ErrorEnum(Enum):
    NO_MOVING_SELECTION = "NO_MOVING_SELECTION"
    NETWORK_MISSING = "NETWORK_MISSING"
    NAME_MISSING = "NAME_MISSING"
    NAME_ALREADY_EXISTS = "NAME_ALREADY_EXISTS"
    MOVING_SELECTION_EMPTY = "MOVING_SELECTION_EMPTY"

default_simulation_params = {
    "pop_size": 200,
    "n_offsprings": 40,
    "n_generations": 100,
    "termination": "",
    "sampling": SamplingEnum.HIGH_RSSI_FIRST_SAMPLING.value,
    "crossover": CrossoverEnum.UNIFORM_CROSSOVER.value,
    "mutation": MutationEnum.PM_MUTATION.value,
    "eliminate_duplicates": True,
    "objectives": [ObjectiveEnum.AVG_DL_RATE.value, ObjectiveEnum.POWER_CONSUMPTION.value],
    "algorithm": AlgorithmEnum.NSGA2.value,
    "moving_speed": 1.9,
    "reset_rate_in_ngen": 15,
    "moving_selection_percent": 30,
    "running_time_in_s": 120,
    "iterations": 1,
    "greedy_to_moving": False,
    "use_greedy_assign": False,
    "moving_selection_name": "",
    "running_mode": RunningMode.STATIC.value
} | default_node_params

class Network_Simulation_State():
    def __init__(self, graph: Son, script_mode=False, network_name="", config_file_path="", fps = 1) -> None:
        self.script_mode = script_mode
        self.fps = fps
        self.optimization_process = multiprocessing.Process()
        self.config_params = default_simulation_params
        self.current_config_name = ""
        self.current_network_name = ""
        self.finished = False
        self.topology_changed = False
        self.dt_since_last_history_update = 0
        self.ticks_since_last_history_update = 0
        self.optimization_running = False
        self.dt_since_last_activation_profile_fetch = 0
        self.n_gen_since_last_fetch = 0
        self.ngen_total = 0
        self.pymoo_is_reset_ready = True
        self.dt_since_last_evo_reset = 0
        self.objective_history = []
        self.ngen_since_last_evo_reset = 0
        self.running_time_in_s = 0
        self.running_ticks = 0
        self.iterations = 0
        self.moving_users = {}
        self.show_moving_users = False
        self.queue_flags = {"activation_dict": False, "objective_space": False,
                            "just_resetted": False,
                            "n_gen_since_last_fetch": 0, "n_gen": 0,
                            "n_gen_since_last_reset": 0}
        self.pymoo_message_queue = multiprocessing.Queue()
        self.editor_message_queue = multiprocessing.Queue()

        self.son = graph
        self.activation = {}
        self.onFinishedEvent = Event()

        if network_name != "":
            self.current_network_name = network_name
            # load network graph
            self.load_current_adjacencies()

            if config_file_path != "":
                    # load param config script config folder
                    self.load_param_config_from_file(config_file_path)
                    # load moving selections
                    self.load_current_moving_selection()
            
            if script_mode:
                # set iterations to -1 so that optimiaztion starts itself
                self.iterations = -1
                # start optimization -> happens in running_method
                # terminate all processes -> in optimization_finished

    
    ####### event hanlder methods
    def AddSubscribersForFinishedEvent(self,objMethod):
        self.onFinishedEvent += objMethod
         
    def RemoveSubscribersForFinishedEvent(self,objMethod):
        self.onFinishedEvent -= objMethod
    
    ######## physics helper methods
    def get_max_x_y(self, graph: nx.Graph) -> tuple[float, float]:
        max_x = 100
        max_y = 100
        for _, node in enumerate(graph.nodes.data()):
            if node[1]["pos_x"] > max_x:
                max_x = node[1]["pos_x"]
            if node[1]["pos_y"] > max_y:
                max_y = node[1]["pos_y"]

        return (max_x, max_y)
    def dbm_to_watts(self, dbm: float):

        return math.pow(10, dbm/10) / 1000

    def watts_to_dbm(self, watts: float):

        return 10 * math.log(watts * 1000, 10)

    def ghz_to_wave_length_m(self, frequency: float):

        return speed_of_light / frequency / 1000000000
    
    def hz_to_mhz(self, frequcnecy):
        return frequcnecy / 1000
    
    def mhz_to_hz(self, frequency):
        return frequency * 1000

    
    ######## get folder directories and list of filenames etc
    def get_current_network_directory(self):
       if self.current_network_name == "from file" or self.current_network_name =="":
           return ""
       return os.getcwd() + "/datastore/" + self.current_network_name + "/"
    
    def get_current_config_directory(self):
        if self.current_network_name == "from file" or self.current_network_name == "" or self.current_config_name == "from file" or self.current_config_name =="":
            return ""
        return os.getcwd() + "/datastore/" + self.current_network_name + "/" + self.current_config_name + "/"
    
    def get_current_moving_selection_directory(self):
        if self.current_network_name == "from file" or self.current_network_name =="":
            return ""
        return os.getcwd() + "/datastore/" + self.current_network_name + "/moving_selections/"
    
    def get_config_names_for_current_network(self) -> list[str]:
        result_list = ["from file"]

        if self.get_current_network_directory() != "":
            directory_path = self.get_current_network_directory()
            directory_contents = os.listdir(directory_path)
            for item in directory_contents:
                if os.path.isdir(
                        os.path.join(directory_path, item)) and item != "moving_selections":
                    result_list.append(item)

        return result_list

    def get_moving_selections_for_current_network(self) -> list[str]:
        result_list = ["from file"]

        if self.get_current_moving_selection_directory() != "":
            directory_contents = os.listdir(self.get_current_moving_selection_directory())

            for item in directory_contents:
                result_list.append(item.replace(".json", ""))

        return result_list

    def get_results_for_current_config(self) -> list[str]:
        result_list = ["from file"]

        if os.path.exists(self.get_current_config_directory()):
            directory_contents = os.listdir(self.get_current_config_directory())

            for item in directory_contents:
                if "ind_result" in item:
                    result_list.append(item.replace(".json", ""))

        return result_list
    
    def get_network_folder_names(self) -> list[str]:
        # Get the current working directory
        current_directory = os.getcwd() + "/datastore"
        # List all files and directories in the current directory
        directory_contents = os.listdir(current_directory)

        # Filter and show only the directory names
        folder_names: list[str] = ["from file"]
        for item in directory_contents:
            if os.path.isdir(os.path.join(current_directory, item)):
                folder_names.append(item)
        return folder_names
    
    ######## apply configs, network files, moving files, parameter configs
    def load_param_config_from_file(self, file_path: str):
        # load param config
        with open(file_path, 'r', encoding="utf-8") as openfile:
                # Reading params from json file
                loaded_params = json.load(openfile)
                self.config_params = default_simulation_params | loaded_params
                # set current config name
                index = file_path.rfind("/")
                self.current_config_name = file_path[index+1:-5]
                # apply network params
                self.apply_current_network_params_to_graph()
    
    def load_result_ind_from_file(self, name: str):
        if name != "from file":
            self.son.load_graph_from_json_adjacency_file(self.get_current_config_directory() + name + ".json", True)
    
    def apply_current_network_params_to_graph(self):
        if bool(self.config_params):
            self.son.apply_network_node_attributes(self.config_params)
        else:
            self.son.apply_network_node_attributes(default_simulation_params)
   
    def load_current_adjacencies(self):
        # List all files and directories in the current directory
        if self.get_current_network_directory() == "":
            self.son = Son()
            return
        
        directory_contents = os.listdir(self.get_current_network_directory())
        adjacencies_file_name = ""
        for item in directory_contents:
            if "adjacencies" in item:
                adjacencies_file_name = item

        layout_file_path = os.getcwd() + "/datastore/" + self.current_network_name + "/" + adjacencies_file_name
        self.son.load_graph_from_json_adjacency_file(layout_file_path, True)
    
    def load_current_moving_selection(self):
        if self.config_params["moving_selection_name"] != "from file" and self.config_params["moving_selection_name"] != "" and self.get_current_moving_selection_directory() != "":
            with open(self.get_current_moving_selection_directory() + self.config_params["moving_selection_name"] + ".json", 'r', encoding="utf-8") as openfile:
                self.moving_users = json.load(openfile)
        else:
            self.moving_users = {}
    
    ########### methods related to saving stuff to files
    def save_new_moving_selection(self, name: str):
        if name == "":
            return ErrorEnum.NAME_MISSING.value
        if not bool(self.moving_users):
            return ErrorEnum.MOVING_SELECTION_EMPTY.value
        if  os.path.isfile(self.get_current_moving_selection_directory() + name + ".json" ):
            return ErrorEnum.NAME_ALREADY_EXISTS.value
        
        # save current selection
        with open(self.get_current_moving_selection_directory() + name + ".json", "w+", encoding="utf-8") as outfile:
            json.dump(self.moving_users, outfile)

        # update params
        self.config_params["moving_selection_name"] = name

        return ""
    
    def save_new_network(self, new_network_name: str):
        if new_network_name == "" or new_network_name == None:
            return ErrorEnum.NAME_MISSING.value
        
        if new_network_name in self.get_network_folder_names():
            return ErrorEnum.NAME_ALREADY_EXISTS.value
        
        # Get the current working directory
        current_directory = os.getcwd()
        # Path of the new folder
        new_folder_path = os.path.join(current_directory, "datastore/" + new_network_name)
        # Create the new folders
        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)
            os.mkdir(new_folder_path + "/moving_selections")
        # save adjacencies
        self.son.save_json_adjacency_graph_to_file(new_folder_path + "/" + new_network_name + "_adjacencies.json")

        self.current_network_name = new_network_name
        
        ## load default param config for new network
        self.config_params = default_simulation_params
        self.apply_current_network_params_to_graph()
        
        return ""
    
    def save_new_config_to_file(self, new_config_name: str):
        if new_config_name == "" or new_config_name == "from file":
            return ErrorEnum.NAME_MISSING.value
        if new_config_name in self.get_config_names_for_current_network():
            return ErrorEnum.NAME_ALREADY_EXISTS.value
        # create cconfig folder
        os.mkdir(self.get_current_network_directory() + new_config_name + "/")
        with open(self.get_current_network_directory() + new_config_name + "/" + new_config_name + ".json", "w+", encoding="utf-8") as outfile:
            json.dump(self.config_params, outfile)
        
        self.current_config_name = new_config_name
        return ""
    
    def save_objective_history_to_file(self):
        json_data = json.dumps(self.objective_history)
        # Save objective JSON data to a file

        with open(f"{self.get_current_config_directory()}objectives_result_{self.iterations}.json", 'w', encoding="utf-8") as file:
            file.write(json_data)
        return ""
    
    ########### methods related to moving users
    def initialize_moving_users(self):

        percentage = self.config_params["moving_selection_percent"]
        user_nodes = [x[0]
                      for x in list(
                          filter(self.son.filter_user_nodes, self.son.graph.nodes.data()))]

        if len(user_nodes) <= 0:
            return False
        self.moving_users = {}
        # randomly select x % of the useres for movement and dict with ids and moving vector
        moving_user_count = round(percentage/100 * len(user_nodes))
        selection_list = np.random.choice(
            user_nodes,
            size=moving_user_count,
            replace=False)

        # initialize moving directions randomly
        deg = np.random.randint(0, 360, size=len(selection_list))
        for index, deg_value in enumerate(deg):
            v = self.rotate_vector_by_deg(np.array([1, 0]), deg_value)
            self.moving_users[selection_list[index]] = (v[0], v[1])
        
        return True
    
    def generate_user_nodes(self, percentage: float):

        user_node_id_list = [x[0] for x in self.son.graph.nodes.data() if x[1]["type"] == "cell"]
        self.son.graph.remove_nodes_from(user_node_id_list)
        
        for _, bs_node in enumerate(list(filter(self.son.filter_bs_nodes, self.son.graph.nodes.data()))):
            bs_user_count = round(self.son.network_node_params[bs_node[1]["type"]]["antennas"] * percentage / 100)
            space_in_deg = 360 / bs_user_count
            radius = self.son.get_bs_type_range(bs_node[1]["type"]) / 2
            radius_vec = np.array([0, radius])
           
            for counter in range(0, bs_user_count):
                rotated_radius_vec = self.rotate_vector_by_deg(radius_vec, counter * space_in_deg, False)
                x_pos = bs_node[1]["pos_x"] + rotated_radius_vec[0]
                y_pos = bs_node[1]["pos_y"] + rotated_radius_vec[1]

                if x_pos > 10000:
                    x_pos = 10000
                if x_pos < 0:
                    x_pos = 0
                if y_pos > 10000:
                    y_pos = 10000
                if y_pos < 0:
                    y_pos = 0

                self.son.add_user_node((x_pos, y_pos),update_network=False)

    def rotate_vector_by_deg(self, vec: np.ndarray, deg: float, normalize=True) -> np.ndarray:
        # rotate vector
        theta = np.deg2rad(deg)
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        v2 = np.dot(rot, vec)

        # normalize vector
        if normalize:
            v2 = v2 / np.linalg.norm(v2)
        
        return v2

    def move_some_users(self):
        if self.config_params["moving_speed"] == 0:

            self.topology_changed = False
            return
        else:
            self.topology_changed = True

        for _, user_id in enumerate(self.moving_users):
            self.move_one_user(user_id)

        self.son.initialize_edges()

    def move_one_user(self, user_node_id: str):
        # TODO make it performant
        self.update_direction_for_user(user_node_id)

        self.son.move_node_by_vec(
            user_node_id,
            (self.moving_users[user_node_id][0] * self.config_params["moving_speed"] / self.fps,
             self.moving_users[user_node_id][1] * self.config_params["moving_speed"] / self.fps),
            update_network=False, initialize_edges=False)

    def update_direction_for_user(self, user_node_id: str):

        moving_vector = (
            self.moving_users[user_node_id][0] * self.config_params["moving_speed"] / self.fps,
            self.moving_users[user_node_id][1] * self.config_params["moving_speed"] / self.fps)
        (next_x_pos, next_y_pos) = self.son.graph.nodes.data()[
            user_node_id]["pos_x"] + moving_vector[0], self.son.graph.nodes.data()[user_node_id]["pos_y"] + moving_vector[1]

        if not (next_x_pos <= 10000 and next_x_pos >= 0 and next_y_pos >= 0 and next_y_pos <= 10000):
            new_direction_numpy = self.rotate_vector_by_deg(
                np.array(self.moving_users[user_node_id]), 180)
            self.moving_users[user_node_id] = (new_direction_numpy[0], new_direction_numpy[1])

        while (self.check_direction_valid(user_node_id, moving_vector) is False):

            new_direction_numpy = self.rotate_vector_by_deg(
                np.array(self.moving_users[user_node_id]), 90)

            self.moving_users[user_node_id] = (new_direction_numpy[0], new_direction_numpy[1])
            moving_vector = (
                self.moving_users[user_node_id][0] * self.config_params["moving_speed"] / self.fps,
                self.moving_users[user_node_id][1] * self.config_params["moving_speed"] /self.fps)

    def check_direction_valid(self, user_node_id: str, moving_vector):

        for _, edge in enumerate(self.son.graph[user_node_id].items()):
            next_rssi = self.son.get_rssi_cell(
                user_node_id, (user_node_id, edge[0]),
                moving_vector=moving_vector)

            if next_rssi > self.son.min_rssi:
                return True

        return False
    
    ######## live running methods
    ### TODO make it dynamic accoridng to objective selection 
    def update_objective_history(self):
        measurements_object = []

        current_avg_user_degree = self.son.get_average_userNode_degree()
        measurements_object.append(round(self.running_time_in_s, 2))
        measurements_object.append(self.running_ticks)
        measurements_object.append(current_avg_user_degree)
        measurements_object.append(self.ngen_total)

        if ObjectiveEnum.AVG_DL_RATE.value in self.config_params["objectives"]:
            measurements_object.append(self.son.get_average_dl_datarate())
        if ObjectiveEnum.POWER_CONSUMPTION.value in self.config_params["objectives"]:
            measurements_object.append(self.son.get_total_energy_consumption())
        if ObjectiveEnum.AVG_ENERGY_EFFICENCY.value in self.config_params["objectives"]:
            measurements_object.append(self.son.get_avg_energy_efficiency())
        if ObjectiveEnum.TOTAL_ENERGY_EFFICIENCY.value in self.config_params["objectives"]:
            measurements_object.append(self.son.get_total_energy_consumption())
        if ObjectiveEnum.AVG_LOAD.value in self.config_params["objectives"]:
            measurements_object.append(self.son.get_average_network_load())
        if ObjectiveEnum.AVG_RSSI.value in self.config_params["objectives"]:
            measurements_object.append(self.son.get_average_rssi())
        if ObjectiveEnum.AVG_SINR.value in self.config_params["objectives"]:
            measurements_object.append(self.son.get_average_sinr())
        if ObjectiveEnum.OVERLOAD.value in self.config_params["objectives"]:
            measurements_object.append(self.son.get_avg_overlad())
        
        self.objective_history.append(measurements_object)
        self.dt_since_last_history_update = 0
        self.ticks_since_last_history_update = 0

    def trigger_evo_reset_invalid_activation_profile(self):
        # invoke evo_reset if threshhold is met
        correction_factor = 1 if self.fps == 30 else 2

        if self.ngen_since_last_evo_reset >= self.config_params["reset_rate_in_ngen"]-correction_factor and self.pymoo_is_reset_ready:
            # check if current activation profile is valid
            # or someone has moved since last call
            if self.topology_changed:
                self.dt_since_last_evo_reset = 0
                self.ngen_since_last_evo_reset = 0
                self.topology_changed = False

                edge_list_with_attributes = deepcopy(self.son.graph.edges)
                node_dic_with_attributes = {}

                for _, node in enumerate(self.son.graph.nodes.data()):
                    node_dic_with_attributes[node[0]] = node[1]

                self.editor_message_queue.put(
                    {"terminate": False,
                     "graph":
                     {"edge_list_with_attributes": edge_list_with_attributes,
                      "node_dic_with_attributes": node_dic_with_attributes
                      },
                     "reset": True})
                self.pymoo_is_reset_ready = False
    
    def stop_evo(self):
        self.editor_message_queue.put(
            {"terminate": True, "graph": False, "reset": False})
        if self.config_params["use_greedy_assign"]:
            self.finished = True

    def force_stop_evo(self):
        self.iterations = self.config_params["iterations"]
        self.editor_message_queue.put(
            {"terminate": True, "graph": False, "reset": False})
        if self.config_params["use_greedy_assign"]:
            self.finished = True
    

    def start_evo(self, new_config_name):

        if not os.path.exists(self.get_current_network_directory()):
            return ErrorEnum.NETWORK_MISSING.value
        
        if self.iterations == - 1:
            # in case of script mode self.iterations initially is -1
            self.iterations = 0
            
        if self.config_params["running_mode"] == RunningMode.LIVE.value and self.config_params["moving_selection_name"] == "":
            return ErrorEnum.NO_MOVING_SELECTION.value
        
        if self.iterations == 0:
            #try to save new config if its the first iteration of the experiment
            save_response_value = self.save_new_config_to_file(new_config_name=new_config_name)
            if save_response_value != "":
                return save_response_value
        
        # start optimization
        if not self.config_params["use_greedy_assign"]:
            self.optimization_process = multiprocessing.Process(target=start_optimization, args=(
                int(self.config_params["pop_size"]),
                int(self.config_params["n_offsprings"]),
                int(self.config_params["n_generations"]),
                "",
                self.config_params["sampling"],
                self.config_params["crossover"],
                self.config_params["mutation"],
                self.config_params["eliminate_duplicates"],
                self.config_params["objectives"],
                self.config_params["algorithm"],
                self.son,
                self.get_current_config_directory(),
                self.pymoo_message_queue,
                self.editor_message_queue,
                self.config_params["running_mode"],
                0.3,
                0.3
            ))
            self.optimization_process.start()

        self.optimization_running = True
        
        return ""

    def on_optimization_run_has_finished(self):
        self.optimization_running = False
        self.iterations += 1
        if self.config_params["running_mode"] == RunningMode.LIVE.value:
            self.save_objective_history_to_file()

        print("finished " + str(self.iterations) + " iteraiton")
                  
        self.reset_all_after_run()
        if self.iterations < self.config_params["iterations"] and self.config_params["running_mode"] != RunningMode.STATIC.value:
            self.start_evo(self.current_config_name)
        else:
            if not self.script_mode:
                self.iterations = 0
                self.onFinishedEvent("message")

    def reset_queue_flags(self):
        self.queue_flags = {"activation_dict": False, "objective_space": False,
                            "just_resetted": False,
                            "n_gen_since_last_fetch": 0, "n_gen": 0,
                            "n_gen_since_last_reset": 0}

    def reset_all_after_run(self):
        self.reset_queue_flags()
        self.activation = {}
        self.topology_changed = False
        self.dt_since_last_history_update = 0
        self.ticks_since_last_history_update = 0
        self.optimization_running = False
        self.dt_since_last_evo_reset = 0
        self.objective_history = []
        self.ngen_since_last_evo_reset = 0
        self.running_time_in_s = 0
        self.running_ticks = 0
        self.n_gen_since_last_fetch = 0
        self.ngen_total = 0
        self.pymoo_is_reset_ready = True
        # reset moving users vecors from file:
        self.load_current_moving_selection()
        self.finished = False
        # reset topology -> load graph from file
        self.load_current_adjacencies()
        # apply current  param config
        self.apply_current_network_params_to_graph()
        # empty message queues
        while not self.editor_message_queue.empty():
            self.editor_message_queue.get()
        while not self.pymoo_message_queue.empty():
            self.pymoo_message_queue.get()

    def step_one_tick(self, dt):
        if self.script_mode:
            # start evo
            if self.iterations == -1:
                self.start_evo(self.current_config_name)

        if self.optimization_running:
            if self.config_params["running_mode"] == RunningMode.LIVE.value:
                # if self.running_ticks % self.fps == 0 and self.running_ticks <= self.config_params["running_time_in_s"] * self.fps:
                self.move_some_users()
                # trigger evo_reset if current activation profile violates son topology and to adjust to movement changes
                self.trigger_evo_reset_invalid_activation_profile()

            if not self.config_params["use_greedy_assign"]:

                # read message queue
                while self.pymoo_message_queue.empty() is False:
                    callback_obj = self.pymoo_message_queue.get()

                    if callback_obj["finished"] == True:
                        # update dropdowns after normal completion and static mode
                        self.finished = True
                    
                    if callback_obj["activation_dict"] is not False and callback_obj["objective_space"] is not False:
                        self.queue_flags["activation_dict"] = callback_obj["activation_dict"]
                        self.queue_flags["objective_space"] = callback_obj["objective_space"]

                    self.queue_flags["n_gen_since_last_fetch"] = callback_obj["n_gen_since_last_fetch"]

                    self.queue_flags["n_gen"] = callback_obj["n_gen"]
                    self.ngen_total = callback_obj["n_gen"]

                    self.queue_flags["n_gen_since_last_reset"] = callback_obj["n_gen_since_last_reset"]
                    self.queue_flags["just_resetted"] = callback_obj["just_resetted"]

                # react to queue messages
                if self.queue_flags["activation_dict"] is not False and self.queue_flags["objective_space"] is not False:

                    self.activation = self.queue_flags["activation_dict"]
                    self.dt_since_last_activation_profile_fetch = 0

                if self.queue_flags["n_gen_since_last_fetch"] is not False:
                    self.n_gen_since_last_fetch = self.queue_flags["n_gen_since_last_fetch"]

                if self.queue_flags["just_resetted"]:
                    self.pymoo_is_reset_ready = True


                self.ngen_since_last_evo_reset = self.queue_flags["n_gen_since_last_reset"]
                # reset queue_flags
                self.reset_queue_flags()
               
                # apply current activation and repair it -> greedy assign for assignemnts which dont match the proposal
                if len(self.activation) > 0:
                    if self.config_params["greedy_to_moving"]:
                        self.activation = self.son.apply_activation_dict(
                            self.activation, update_network_attributes=True,
                            greedy_assign_list=self.moving_users)
                    else:
                        self.activation = self.son.apply_activation_dict(
                            self.activation, update_network_attributes=True)

                else:
                    # if no activation profile is present -> use greedy approach for all useres
                    self.son.apply_activation_profile_greedy_user(update_attributes=True)

            if self.config_params["use_greedy_assign"]:
                self.son.apply_activation_profile_greedy_user(update_attributes=True)

            
            if self.config_params["running_mode"] == RunningMode.LIVE.value:
                # update objectives every second
                if self.running_ticks % self.fps == 0 and self.running_ticks <= self.config_params["running_time_in_s"] * self.fps:
                    self.update_objective_history()

                # stop if running time is over but time is translated into frames
                if self.running_ticks == self.config_params["running_time_in_s"] * self.fps:
                    self.stop_evo()
            
            self.dt_since_last_history_update += dt
            self.ticks_since_last_history_update += 1
            self.dt_since_last_evo_reset += dt
            self.dt_since_last_activation_profile_fetch += dt
            self.running_time_in_s += dt
            self.running_ticks += 1
            
            if self.finished:
                self.on_optimization_run_has_finished()

if __name__ == "__main__":
    network_name = sys.argv[1]
    config_file_path = sys.argv[2]

    son = Son()
    fps = 1
    
    simulation = Network_Simulation_State(son, script_mode=True, network_name=network_name, config_file_path=config_file_path, fps=fps)
    
    dt = 1
    while(simulation.iterations < simulation.config_params["iterations"]):
        start_time = time.time()
        simulation.step_one_tick(dt)
        time.sleep(max(1./fps - (time.time() - start_time), 0))
        dt = time.time() - start_time

    # cProfile.run("main.run()", sort="tottime")