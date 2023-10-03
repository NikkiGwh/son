
from copy import deepcopy
import math
from warnings import catch_warnings
from pygame_gui._constants import UI_BUTTON_PRESSED
import pygame
import sys
from pygame_settings import *
import networkx as nx
import json
from son_main_script import NodeType, Son, default_node_params
import pygame_gui
import os
import multiprocessing
import numpy as np
from math import cos, sin
from son_pymoo import AlgorithmEnum, CrossoverEnum, MutationEnum, ObjectiveEnum, RunningMode, SamplingEnum, start_optimization
import cProfile
import re
from scipy.constants import speed_of_light


class CustomConfirmationDialog(pygame_gui.windows.UIConfirmationDialog):
    def __init__(
            self, rect, manager, window_title, action_long_desc, action_short_name, visible,
            blocking):
        super().__init__(
            rect=rect, manager=manager, window_title=window_title,
            action_long_desc=action_long_desc, action_short_name=action_short_name, visible=visible,
            blocking=blocking)

    def on_close_window_button_pressed(self):
        # Add your custom implementation here
        self.hide()

    def process_event(self, event: pygame.event.Event):
        # consumed_event = super().process_event(event)

        if event.type == UI_BUTTON_PRESSED and event.ui_element == self.cancel_button:
            self.hide()

        if event.type == UI_BUTTON_PRESSED and event.ui_element == self.confirm_button:
            self.hide()

        return True


dropdown_menue_options_list = ["macro", "micro", "femto", "pico", "cell", "remove"]
text_input_float_number_type_characters = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "."]
text_input_integer_number_type_characters = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

default_algorithm_param_config = {
    "pop_size": 200,
    "n_offsprings": 40,
    "n_generations": 10000000,
    "termination": "",
    "sampling": SamplingEnum.RANDOM_SAMPLING.value,
    "crossover": CrossoverEnum.UNIFORM_CROSSOVER.value,
    "mutation": MutationEnum.PM_MUTATION.value,
    "eliminate_duplicates": "True",
    "objectives": [ObjectiveEnum.AVG_DL_RATE.value, ObjectiveEnum.ENERGY_EFFICIENCY.value],
    "algorithm": AlgorithmEnum.NSGA2.value,
    # moving_speed = 1 refers to 30 m/s (because of 30 tickes per second)
    "moving_speed": 28.0,
    "reset_rate_in_ngen": 5,
    "moving_selection_percent": 30,
    "running_time_in_s": 120,
    "iterations": 10,
    "greedy_to_moving": False,
    "use_greedy_assign": False,
    "moving_selection_name": ""
} | default_node_params


def get_network_folder_names() -> list[str]:
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


class Main():
    def __init__(self, graph: Son, script_mode=False, network_name="", config_name="") -> None:
        pygame.init()
        self.script_mode = script_mode
        self.running_mode = RunningMode.LIVE.value
        self.finished = False
        self.moving_speed_in_m_per_second = 0
        self.topology_changed = False
        self.dt_since_last_history_update = 0
        self.ticks_since_last_history_update = 0
        self.optimization_running = False
        self.dt_since_last_activation_profile_fetch = 0
        self.n_gen_since_last_fetch = 0
        self.ngen_total = 0
        self.dt_since_last_evo_reset = 0
        self.objective_history = []
        self.ngen_since_last_evo_reset = 0
        self.running_time_in_s = 0
        self.running_ticks = 0
        self.iterations = 0
        self.selected_node_id = None
        self.moving_users = {}
        self.show_moving_users = False
        self.queue_flags = {"activation_dict": False, "objective_space": False,
                            "n_gen_since_last_fetch": 0, "n_gen": 0,
                            "n_gen_since_last_reset": 0}
        self.current_save_result_directory = ""
        self.pymoo_message_queue = multiprocessing.Queue()
        self.editor_message_queue = multiprocessing.Queue()
        self.right_mouse_action = dropdown_menue_options_list[0]
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.background = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.background.fill(pygame.colordict.THECOLORS["white"])
        self.clock = pygame.time.Clock()
        self.son = graph
        self.activation = {}

        # networkx 1 = 1m I want max 10 km -> 0.1 on screen equals 1 m
        self.unit_size_x, self.unit_size_y = (
            GAME_WIDTH / 10000,
            GAME_HEIGHT / 10000)

        # parameter config
        self.config_params = default_algorithm_param_config
        # GUI
        self.manager = pygame_gui.UIManager((WINDOW_WIDTH, WINDOW_HEIGHT), './theme.json')
        self.popup = CustomConfirmationDialog(
            rect=pygame.Rect(GAME_WIDTH + 20, 20, 100, 100),
            blocking=True,
            manager=self.manager,
            window_title="network already exist",
            action_short_name="ok",
            action_long_desc="this network name already exist",
            visible=False,
        )
        self.info_text_box = pygame_gui.elements.UITextBox(
            "hallo", pygame.Rect(0, 0, -1, -1), self.manager, wrap_to_height=True, visible=False)

        self.info_text_box_objectives = pygame_gui.elements.UITextBox(
            "hallo", pygame.Rect(500, 500, -1, -1),
            manager=self.manager, wrap_to_height=True, visible=False)

        # algorithm params
        self.create_algo_param_ui_elements()

        # live config elements
        self.create_live_param_ui_elements()
        self.ui_container_live_config.disable()

        # auto mode
        if script_mode:
            # load network

            self.dropdown_menu_pick_network.selected_option = network_name
            self.input_algo_config_name.set_text(config_name)
            self.dropdown_pick_algo_config.selected_option = config_name

            event: pygame.event.Event = pygame.event.Event(0)
            event.text = network_name
            self.on_dropdown_pick_network_changed(event)

            # load param config
            with open("predefined_configs/" + config_name + ".json", 'r', encoding="utf-8") as openfile:
                # Reading params from json file
                self.config_params = json.load(openfile)
                # apply network params
                self.apply_current_network_params_to_ui_and_graph(initial_network_name=network_name, initial_config_name=config_name)

            # set iterations to -1 so that optimiaztion starts itself
            self.iterations = -1
            # start optimization -> happens in running_method
            # terminate all processes -> in optimization_finished

    def disable_ui(self):
        self.ui_container.disable()
        self.ui_container_live_config.disable()

    def enable_ui(self):
        self.ui_container.enable()
        self.ui_container_live_config.enable()

    def get_configs_for_current_network(self) -> list[str]:
        result_list = ["from file"]

        if self.dropdown_menu_pick_network.selected_option != "from file":
            directory_path = "datastore/" + self.dropdown_menu_pick_network.selected_option
            directory_contents = os.listdir(directory_path)
            for item in directory_contents:
                if os.path.isdir(
                        os.path.join(directory_path, item)) and item != "moving_selections":
                    result_list.append(item)

        return result_list

    def get_moving_selections_for_current_network(self) -> list[str]:
        result_list = ["from file"]

        if self.dropdown_menu_pick_network.selected_option != "from file":
            directory_contents = os.listdir(
                "datastore/" + self.dropdown_menu_pick_network.selected_option + "/moving_selections")

            for item in directory_contents:
                result_list.append(item.replace(".json", ""))

        return result_list

    def get_results_for_current_config(self) -> list[str]:
        result_list = ["from file"]

        if os.path.exists("datastore/" + self.dropdown_menu_pick_network.selected_option + "/" + self.dropdown_pick_algo_config.selected_option):
            directory_contents = os.listdir(
                "datastore/" + self.dropdown_menu_pick_network.selected_option + "/" + self.dropdown_pick_algo_config.selected_option)

            for item in directory_contents:
                if "ind_result" in item:
                    result_list.append(item.replace(".json", ""))

        return result_list

    def get_max_x_y(self, graph: nx.Graph) -> tuple[float, float]:
        max_x = 100
        max_y = 100
        for _, node in enumerate(graph.nodes.data()):
            if node[1]["pos_x"] > max_x:
                max_x = node[1]["pos_x"]
            if node[1]["pos_y"] > max_y:
                max_y = node[1]["pos_y"]

        return (max_x, max_y)

    def draw_network(self):
        self.display_surface.blit(self.background, (0, 0))
        self.draw_edges()
        self.draw_nodes()

    def draw_edges(self):
        for _, edge in enumerate(self.son.graph.edges.data()):
            activation = "active" if edge[2]["active"] == True else "inactive"

            if self.show_edges_checkbox_active or activation == "active":
                pygame.draw.line(
                    self.display_surface, pygame.colordict.THECOLORS
                    [style["edges"]["colors"][activation]],
                    (self.son.graph.nodes[edge[0]]["pos_x"] * self.unit_size_x, self.son.graph.nodes
                     [edge[0]]["pos_y"] * self.unit_size_y),
                    (self.son.graph.nodes[edge[1]]["pos_x"] * self.unit_size_x, self.son.graph.nodes
                     [edge[1]]["pos_y"] * self.unit_size_y),
                    style["edges"]["sizes"]["edge_width"][activation])

    def draw_nodes(self):

        for _, node in enumerate(self.son.graph.nodes.data()):
            node_type = node[1]["type"]
            activation = False if node[1]["type"] != "cell" and node[1]["active"] == False else True

            if self.show_moving_users:
                node_color_key = "inactive" if not activation else node_type if node[
                    0] not in self.moving_users else "moving"
            else:
                node_color_key = "inactive" if not activation else node_type

            pygame.draw.circle(
                self.display_surface, style["nodes"]["colors"]["fill"][node_color_key],
                (node[1]["pos_x"] * self.unit_size_x, node[1]["pos_y"] * self.unit_size_y),
                style["nodes"]["sizes"]["radius"][node_type])

            if (node_type != "cell" and node[1]["load"] > 1):
                pygame.draw.circle(
                    self.display_surface, pygame.colordict.THECOLORS
                    [style["nodes"]["colors"]["fill"]["overload"]],
                    (node[1]["pos_x"] * self.unit_size_x, node[1]["pos_y"] * self.unit_size_y),
                    style["nodes"]["sizes"]["radius"][node_type], width=1)

    def node_clicked(self, pos: tuple[int, int]) -> str | None:
        id = None

        for _, node in enumerate(self.son.graph.nodes.data()):
            node_type = node[1]["type"]
            y = node[1]["pos_y"] * self.unit_size_x
            x = node[1]["pos_x"] * self.unit_size_y
            r = style["nodes"]["sizes"]["radius"][node_type]
            if pygame.Vector2(x, y).distance_to(pos) <= r:
                id = node[0]

        return id

    def node_drag(self, node_id, target_pos: tuple[int, int]):
        target_x = round(target_pos[0] / self.unit_size_x, 2)
        target_y = round(target_pos[1] / self.unit_size_y, 2)
        self.son.move_node_by_pos(
            node_id, (target_x, target_y),
            update_network=False)
        self.topology_changed = True

    def initialize_moving_users(self):

        percentage = self.config_params["moving_selection_percent"]

        user_nodes = [x[0]
                      for x in list(
                          filter(self.son.filter_user_nodes, self.son.graph.nodes.data()))]

        if len(user_nodes) <= 0:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 100, 100),
                blocking=True,
                manager=self.manager,
                window_title="No users",
                action_short_name="ok",
                action_long_desc="add some users",
                visible=True,

            )
            return
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

    def rotate_vector_by_deg(self, vec: np.ndarray, deg: int) -> np.ndarray:
        # rotate vector
        theta = np.deg2rad(deg)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        v2 = np.dot(rot, vec)

        # normalize vector
        vector_norm = v2 / np.linalg.norm(v2)
        return vector_norm

    def move_some_users(self):
        for _, user_id in enumerate(self.moving_users):

            self.move_one_user(user_id)

        self.son.initialize_edges()

    def move_one_user(self, user_node_id: str):

        # TODO make it performant
        self.update_direction_for_user(user_node_id)

        self.son.move_node_by_vec(
            user_node_id,
            (self.moving_users[user_node_id][0] * self.moving_speed_in_m_per_second,
             self.moving_users[user_node_id][1] * self.moving_speed_in_m_per_second),
            update_network=False, initialize_edges=False)

        if self.config_params["moving_speed"] != 0:
            self.topology_changed = True
        else:
            self.topology_changed = False

    def update_direction_for_user(self, user_node_id: str):

        moving_vector = (
            self.moving_users[user_node_id][0] * self.config_params["moving_speed"],
            self.moving_users[user_node_id][1] * self.config_params["moving_speed"])
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
                self.moving_users[user_node_id][0] * self.config_params["moving_speed"],
                self.moving_users[user_node_id][1] * self.config_params["moving_speed"])

    def check_direction_valid(self, user_node_id: str, moving_vector):

        for _, edge in enumerate(self.son.graph[user_node_id].items()):
            next_rssi = self.son.get_rssi_cell(
                user_node_id, (user_node_id, edge[0]),
                moving_vector=moving_vector)

            if next_rssi > self.son.min_rssi:
                return True

        return False

    def update_objective_history(self):
        current_energy_efficiency = self.son.get_energy_efficiency()
        # current_energy_efficiency = self.son.get_total_energy_consumption()
        current_avg_dl_datarate = self.son.get_average_dl_datarate()
        current_avg_user_degree = self.son.get_average_userNode_degree()

        self.objective_history.append(
            (round(self.running_time_in_s, 2),
             self.running_ticks, current_energy_efficiency, current_avg_dl_datarate, current_avg_user_degree, self.ngen_total))

        self.dt_since_last_history_update = 0
        self.ticks_since_last_history_update = 0

    def trigger_evo_reset_invalid_activation_profile(self):
        # invoke evo_reset if threshhold is met
        if self.ngen_since_last_evo_reset == self.config_params["reset_rate_in_ngen"]-1:

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

    def on_left_click(self, target_pos: tuple[int, int]):
        node_id = self.selected_node_id
        if node_id is not None and self.dropdown_drag_inspect_menue.selected_option == "inspect":
            self.info_text_box.set_relative_position(target_pos)
            new_text = str(target_pos) + "<br>"
            new_text += str(node_id) + "<br>"
            for _, node_attribute in enumerate(self.son.graph.nodes[node_id]):
                new_text += node_attribute + ": " + str(
                    self.son.graph.nodes[node_id][node_attribute]) + "<br>"

            self.info_text_box.set_text(new_text)
            self.info_text_box.show()
        else:
            self.info_text_box.hide()

    def show_objectives_info_box(self):
        if self.objectives_checkbox_active:
            self.objectives_checkbox_active = False
            self.info_text_box_objectives.hide()
            self.objectives_infobox_toggle.unselect()
            return
        else:
            self.objectives_checkbox_active = True

        if len(self.son.graph.nodes) == 0 or len(self.son.graph.edges) == 0:
            self.info_text_box_objectives.set_text("There is no valid network yet.")
            self.info_text_box_objectives.show()
            self.objectives_infobox_toggle.select()
            return

        objective_text = ""

        avg_sinr: float = self.son.get_average_sinr()
        avg_rssi: float = self.son.get_average_rssi()
        avg_dl_rate: float = self.son.get_average_dl_datarate()
        avg_load: float = self.son.get_average_network_load()
        energy_efficiency: float = self.son.get_energy_efficiency()
        network_energy_consumption: float = self.son.get_total_energy_consumption()
        objective_text = "user devices<br>avg_rssi: " + str(avg_rssi) + "<br>avg_sinr: " + str(avg_sinr) + "<br>avg_dl_rate: " + str(avg_dl_rate) + "<br><br>base stations<br>avg_load %: " + str(
            avg_load) + "<br>total energy consumption: " + str(network_energy_consumption) + "<br>energy_efficiency: " + str(energy_efficiency)

        self.info_text_box_objectives.set_text(objective_text)
        self.info_text_box_objectives.show()

    def on_right_click(self, target_pos: tuple[int, int]):
        target_x = round(target_pos[0] / self.unit_size_x, 2)
        target_y = round(target_pos[1] / self.unit_size_y, 2)

        if self.right_mouse_action != "remove" and self.right_mouse_action != "cell":
            self.son.add_bs_node(
                (target_x, target_y),
                update_network=False, bs_type=self.right_mouse_action)
        if self.right_mouse_action == "cell":
            self.son.add_user_node(
                (target_x, target_y),
                update_network=False)
        elif self.right_mouse_action == "remove":
            node_id = self.node_clicked(target_pos)
            if node_id:
                self.son.remove_node(node_id=node_id, update_network=False)
        self.update_network_info_lables()

    def update_network_info_lables(self):
        beams = 0
        bs_nodes_list = list(filter(lambda x: x[1]["type"] != "cell", self.son.graph.nodes.data()))
        for _, bs_node in enumerate(bs_nodes_list):
            beams += self.son.network_node_params[bs_node[1]["type"]]["antennas"]

        user_count = len(
            list(
                filter(
                    lambda x: x[1]["type"] == "cell", self.son.graph.nodes.data())))
        self.capacity_label.set_text(f"network capacity: {beams}")
        self.user_count_label.set_text(f"user count: {user_count}")

    def onclick_show_edges_checkbox(self):
        self.show_edges_checkbox_active = not self.show_edges_checkbox_active
        self.show_edges_checkbox.select() if self.show_edges_checkbox_active else self.show_edges_checkbox.unselect()


    def ontoggle_greedy_to_moving(self):
        self.config_params["greedy_to_moving"] = not self.config_params["greedy_to_moving"]
        self.greedy_to_moving_toggle.select() if self.config_params["greedy_to_moving"] else self.greedy_to_moving_toggle.unselect()

    def onshow_moving_selection_checkbox(self):
        self.show_moving_users = not self.show_moving_users

        self.show_moving_selection_toggle.select() if self.show_moving_users else self.show_moving_selection_toggle.unselect()


    def apply_current_network_params_to_graph(self):
        self.son.apply_network_node_attributes(self.config_params)

    def apply_current_network_params_to_ui_and_graph(self, initial_config_name="", initial_network_name=""):

        # apply params on network
        self.son.apply_network_node_attributes(self.config_params)
        # load moving selections
        if str(self.config_params["moving_selection_name"]) != "":
            net_name = initial_network_name if initial_network_name != "" else self.dropdown_menu_pick_network.selected_option
            with open("datastore/" + net_name + "/moving_selections/" + str(self.config_params["moving_selection_name"]) + ".json", 'r', encoding="utf-8") as openfile:
                # Reading from json file
                self.moving_users = json.load(openfile)
                self.dropdown_moving_selection.selected_option = str(
                    self.config_params["moving_selection_name"])
        
        self.create_algo_param_ui_elements(
                    creata_dropdown_pick_algo_config=True,
                    initial_config_name=initial_config_name,
                    initial_network_name=self.dropdown_menu_pick_network.selected_option if initial_network_name == "" else initial_network_name)

        self.create_live_param_ui_elements()

    def dbm_to_watts(self, dbm: float):

        return math.pow(10, dbm/10) / 1000

    def watts_to_dbm(self, watts: float):

        return 10 * math.log(watts * 1000, 10)

    def ghz_to_wave_length_m(self, frequency: float):

        return speed_of_light / frequency / 1000000000

    def apply_adjacencies_from_json_file(self, file_name):
        layout_file_path = os.getcwd() + "/datastore/" + self.dropdown_menu_pick_network.selected_option + "/" + file_name
        self.son.load_graph_from_json_adjacency_file(layout_file_path, True)
        self.update_network_info_lables()

    def on_entry_changed(self, event: pygame.event.Event):

        pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)$'
        if not re.match(pattern, event.text):
            event.text = 0
        if event.ui_element == self.input_transmission_power:
            tx_in_watts = self.dbm_to_watts(float(event.text))
            self.config_params[self.right_mouse_action]["tx_power"] = tx_in_watts
        if event.ui_element == self.input_antennas:
            self.config_params[self.right_mouse_action]["antennas"] = float(event.text)
        if event.ui_element == self.input_channel_bandwidth:
            self.config_params[self.right_mouse_action]["bandwidth"] = float(event.text)
        if event.ui_element == self.input_frequency:
            self.config_params[self.right_mouse_action]["frequency"] = float(event.text)
            if float(event.text) > 0:
                wave_length = self.ghz_to_wave_length_m(float(event.text))
            else:
                wave_length = 0
            self.config_params[self.right_mouse_action]["wave_length"] = wave_length
        if event.ui_element == self.input_standby_power:
            self.config_params[self.right_mouse_action]["standby_power"] = float(event.text)
        if event.ui_element == self.input_static_power:
            self.config_params[self.right_mouse_action]["static_power"] = float(event.text)
        if event.ui_element == self.input_n_generations:
            self.config_params["n_generations"] = float(event.text)
        if event.ui_element == self.input_n_offsprings:
            self.config_params["n_offsprings"] = float(event.text)
        if event.ui_element == self.input_pop_size:
            self.config_params["pop_size"] = float(event.text)
        if event.ui_element == self.input_resetting_rate:
            self.config_params["reset_rate_in_ngen"] = float(event.text)
        if event.ui_element == self.input_running_time:
            self.config_params["running_time_in_s"] = float(event.text)
        if event.ui_element == self.input_velocity:
            self.config_params["moving_speed"] = float(event.text)
            # convert simulation moving speed to m/s -> 30 ticks per second, 1 x = 1 m
            self.moving_speed_in_m_per_second = round(float(event.text) / 30, 4)
        if event.ui_element == self.input_create_movement_selection_percentage:
            self.config_params["moving_selection_percent"] = float(event.text)
        if event.ui_element == self.input_iterations:
            self.config_params["iterations"] = int(event.text)

    def on_dropdown_pick_network_changed(self, event: pygame.event.Event):
        # reset config params to default
        self.config_params = default_algorithm_param_config
        self.create_algo_param_ui_elements(creata_dropdown_pick_algo_config=True, initial_network_name=event.text)
        self.create_live_param_ui_elements()

        if event.text != "from file":
            
            self.reload_current_network_graph()
            self.ui_container_live_config.enable()
        else:
            self.dropdown_pick_algo_config.disable()
            self.dropdown_pick_result.disable()
            self.ui_container_live_config.disable()
            self.son = Son()

    def reload_current_network_graph(self):
        # Get the current working directory
        current_directory = os.getcwd() + "/datastore/" + self.dropdown_menu_pick_network.selected_option
        # List all files and directories in the current directory
        directory_contents = os.listdir(current_directory)
        adjacencies_file_name = ""
        for item in directory_contents:
            if "adjacencies" in item:
                adjacencies_file_name = item
        # load graph layout
        self.apply_adjacencies_from_json_file(adjacencies_file_name)

        # reapply current config for graph
        self.apply_current_network_params_to_graph()

    def reload_current_moving_selection(self):
        if self.dropdown_moving_selection.selected_option != "from file" and self.dropdown_moving_selection.selected_option != "" and self.dropdown_menu_pick_network.selected_option != "from file":
            with open("datastore/" + self.dropdown_menu_pick_network.selected_option + "/moving_selections/" + self.dropdown_moving_selection.selected_option + ".json", 'r', encoding="utf-8") as openfile:
                # Reading from json file
                self.moving_users = json.load(openfile)
                # update all algorithm config inputs
        else:
            self.moving_users = {}

    def on_dropdown_input_algorithm_config_changed(self, event: pygame.event.Event, param_key: str):
        self.config_params[param_key] = event.text

    def on_selectionlist_input_objectives_changed(self, event: pygame.event.Event):
        self.config_params["objectives"] = self.input_objectives.get_multi_selection()

        if (len(self.config_params["objectives"]) == 0):
            self.evo_start_button.disable()
        else:
            self.evo_start_button.enable()

    def on_dropdown_pick_algo_config_changed(self, event: pygame.event.Event):
        if event.text == "from file":
            self.dropdown_pick_result.disable()
        else:
            with open("datastore/" + self.dropdown_menu_pick_network.selected_option + "/" + event.text + "/" + event.text + ".json", 'r', encoding="utf-8") as openfile:
                # Reading params from json file
                self.config_params = json.load(openfile)
                # apply network params to ui and graph
                self.apply_current_network_params_to_ui_and_graph(initial_config_name=event.text)


    def on_dropdown_moving_selection_changed(self, event: pygame.event.Event):
        if event.text == "from file":
            self.moving_users = {}
            self.config_params["moving_selection_name"] = ""
        else:
            with open("datastore/" + self.dropdown_menu_pick_network.selected_option + "/moving_selections/" + event.text + ".json", 'r', encoding="utf-8") as openfile:
                # Reading from json file
                self.moving_users = json.load(openfile)
                self.config_params["moving_selection_name"] = event.text

    def on_dropdown_pick_result_changed(self, event: pygame.event.Event):
        # Opening json file
        if event.text != "from file":
            self.son.load_graph_from_json_adjacency_file(
                "datastore/" + self.dropdown_menu_pick_network.selected_option + "/" + self.
                dropdown_pick_algo_config.selected_option + "/" + event.text, True)
        self.update_network_info_lables()
        # TODO disable some other elements ?

    def on_dropdown_mode_changed(self, event: pygame.event.Event):

        self.running_mode = event.text

    def create_live_param_ui_elements(self):


        if hasattr(self, "ui_container_live_config"):
            self.ui_container_live_config.kill()


        self.ui_container_live_config = pygame_gui.elements.UIPanel(
            pygame.Rect((GAME_WIDTH, 680), (WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT - 680)),
            object_id="#ui_container_live_config")

        self.input_resetting_rate_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 0), (-1, 30)), "resetting rate in nGen", self.manager, self.ui_container_live_config)
        self.input_resetting_rate = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 0),
                        (100, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="resetting rate",
            initial_text=str(self.config_params["reset_rate_in_ngen"]))
        self.input_resetting_rate.set_allowed_characters(text_input_integer_number_type_characters)

        self.input_running_time_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 30), (-1, 30)), "running time in s", self.manager, self.ui_container_live_config)
        self.input_running_time = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 30),
                        (100, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="running time",
            initial_text=str(self.config_params["running_time_in_s"]))
        self.input_running_time.set_allowed_characters(text_input_integer_number_type_characters)

        self.input_velocity_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 60), (-1, 30)), "velocity", self.manager, self.ui_container_live_config)
        self.input_velocity = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 60),
                        (100, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="velocity",
            initial_text=str(self.config_params["moving_speed"]))
        self.input_velocity.set_allowed_characters(text_input_float_number_type_characters)
        # set simulation moving_speed from current_param_config
        self.moving_speed_in_m_per_second = round(float(self.config_params["moving_speed"]) / 30, 4)

        self.create_movement_selection_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, 90),
                                      (-1, 30)),
            text='create selection %', manager=self.manager,
            container=self.ui_container_live_config)
        self.input_create_movement_selection_percentage = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 90),
                        (100, 30)),
            self.manager, self.ui_container_live_config,
            initial_text=str(self.config_params["moving_selection_percent"]))
        self.input_create_movement_selection_percentage.set_allowed_characters(
            text_input_float_number_type_characters)
        self.show_moving_selection_toggle = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (330, 90), (-1, 30)), text="show moving users", manager=self.manager, container=self.ui_container_live_config, object_id="toggle")

        self.save_moving_selection_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, 120),
                                      (-1, 30)),
            text='save current selection', manager=self.manager,
            container=self.ui_container_live_config)
        self.input_moving_selection_name = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 120),
                        (150, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="selection name",
            initial_text=str(self.config_params["moving_selection_name"]))

        self.select_moving_selection_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 150), (-1, 30)), "moving selections", self.manager, self.ui_container_live_config)

        self.dropdown_moving_selection = pygame_gui.elements.UIDropDownMenu(
            options_list=self.get_moving_selections_for_current_network(),
            starting_option=str(self.config_params["moving_selection_name"])
            if str(self.config_params["moving_selection_name"]) != "" else "from file",
            relative_rect=pygame.Rect((220, 150),
                                      (200, 30)),
            manager=self.manager, container=self.ui_container_live_config,)
        self.input_iterations_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 180), (-1, 30)), "iterations", self.manager, self.ui_container_live_config)
        self.input_iterations = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 180),
                        (100, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="iterations",
            initial_text=str(self.config_params["iterations"]))
        self.input_iterations.set_allowed_characters(text_input_integer_number_type_characters)

    def create_algo_param_ui_elements(
            self, creata_dropdown_pick_algo_config=True, initial_network_name="", initial_config_name =""):
        
        
        if hasattr(self, "ui_container"):
            self.ui_container.kill()

        initial_nodeType = self.right_mouse_action if self.right_mouse_action != NodeType.CELL.value else NodeType.MACRO.value
        self.ui_container = pygame_gui.elements.UIPanel(
            pygame.Rect((GAME_WIDTH, 0), (WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT)),
            object_id="#ui_container")

        self.dropdown_menu = pygame_gui.elements.UIDropDownMenu(
            dropdown_menue_options_list,
            dropdown_menue_options_list[0],
            pygame.Rect((20, 20), (100, 30)), self.manager,
            container=self.ui_container)

        self.dropdown_menu_pick_network = pygame_gui.elements.UIDropDownMenu(
            options_list=get_network_folder_names(),
            starting_option=get_network_folder_names()[0]
            if initial_network_name == "" else initial_network_name, relative_rect=pygame.Rect(
                (120, 20),
                (100, 30)),
            manager=self.manager, container=self.ui_container)
        self.current_save_result_directory = "datastore/" + self.dropdown_menu_pick_network.selected_option

        self.dropdown_drag_inspect_menue = pygame_gui.elements.UIDropDownMenu(
            options_list=["inspect", "drag"],
            starting_option="drag",
            relative_rect=pygame.Rect(
                220, 20, 100, 30),
            manager=self.manager,
            container=self.ui_container
        )

        self.show_edges_checkbox = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (320, 20),
            (50, 30)),
            object_id="toggle",
            text="show edges", manager=self.manager,
            container=self.ui_container)
        self.show_edges_checkbox_active = True

        self.objectives_infobox_toggle = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (370, 20),
            (50, 30)),
            object_id="toggle",
            text="obj", manager=self.manager,
            container=self.ui_container)
        self.objectives_checkbox_active = False

        self.apply_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (320, 230),
            (-1, 30)),
            object_id="#apply_button",
            text="apply params", manager=self.manager,
            container=self.ui_container)

        # network params

        self.input_antennas_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 50), (-1, 30)), "maximum amount of beams", self.manager, self.ui_container)
        self.input_antennas = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 50),
                        (100, 30)),
            self.manager, self.ui_container, placeholder_text="max beams",
            initial_text=str(self.config_params[initial_nodeType]["antennas"]))
        self.input_antennas.set_allowed_characters(text_input_float_number_type_characters)

        self.input_network_name = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((320, 50),
                        (100, 30)),
            self.manager, self.ui_container, placeholder_text="network name")

        self.save_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((320, 80), (100, 30)),
            object_id="#save_button", text='save', manager=self.manager, container=self.
            ui_container)

        self.capacity_label = pygame_gui.elements.UILabel(pygame.Rect(
            (320, 110), (-1, 30)), f"user capacity: 0", self.manager, self.ui_container)
        self.user_count_label = pygame_gui.elements.UILabel(pygame.Rect(
            (320, 140), (-1, 30)), "user count: 0", self.manager, self.ui_container)
        self.update_network_info_lables()

        self.input_channel_bandwidth_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 110), (-1, 30)), "channel bandwidth in HZ", self.manager, self.ui_container)
        self.input_channel_bandwidth = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 110), (100, 30)), self.manager, self.ui_container, placeholder_text="bandwidth",
            initial_text=str(self.config_params[initial_nodeType]["channel_bandwidth"]))
        self.input_channel_bandwidth.set_allowed_characters(text_input_float_number_type_characters)

        self.input_transmission_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 140), (-1, 30)), "transmission power in dbm", self.manager, self.ui_container)
       
        # transform tx_power in watts from file to dbm for ui
        tx_power_dbm = self.watts_to_dbm(self.config_params[initial_nodeType]["tx_power"])
        self.input_transmission_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 140), (100, 30)), self.manager, self.ui_container, placeholder_text="tx power",
            initial_text=str(tx_power_dbm))
        self.input_transmission_power.set_allowed_characters(
            text_input_float_number_type_characters)

        self.input_static_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 170), (-1, 30)), "static power in Watt", self.manager, self.ui_container)
        self.input_static_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 170), (100, 30)), self.manager, self.ui_container, placeholder_text="static power",
            initial_text=str(self.config_params[initial_nodeType]["static_power"]))
        self.input_static_power.set_allowed_characters(text_input_float_number_type_characters)

        self.input_standby_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 200), (-1, 30)), "standby power in Watt", self.manager, self.ui_container)
        self.input_standby_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 200), (100, 30)), self.manager, self.ui_container, placeholder_text="standby power",
            initial_text=str(self.config_params[initial_nodeType]["standby_power"]))
        self.input_standby_power.set_allowed_characters(text_input_float_number_type_characters)

        self.input_frequency_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 230), (-1, 30)), "frequency in GHZ", self.manager, self.ui_container)
        self.input_frequency = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 230), (100, 30)), self.manager, self.ui_container, placeholder_text="frequency",
            initial_text=str(self.config_params[initial_nodeType]["frequency"]))
        self.input_frequency.set_allowed_characters(text_input_float_number_type_characters)

        self.input_pop_size_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 260), (-1, 30)), "population size", self.manager, self.ui_container)
        self.input_pop_size = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 260), (100, 30)), self.manager, self.ui_container, placeholder_text="pop_size",
            initial_text=str(self.config_params["pop_size"]))
        self.input_pop_size.set_allowed_characters(text_input_float_number_type_characters)

        self.input_n_offsprings_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 290), (-1, 30)), "number of offsprings", self.manager, self.ui_container)
        self.input_n_offsprings = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 290), (100, 30)), self.manager, self.ui_container, placeholder_text="n_offsprings",
            initial_text=str(self.config_params["n_offsprings"]))
        self.input_n_offsprings.set_allowed_characters(text_input_float_number_type_characters)

        self.input_n_generations_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 320), (-1, 30)), "number of generations", self.manager, self.ui_container)
        self.input_n_generations = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 320), (100, 30)), self.manager, self.ui_container, placeholder_text="n_generations",
            initial_text=str(self.config_params["n_generations"]))
        self.input_n_generations.set_allowed_characters(text_input_float_number_type_characters)

        self.input_sampling_dropdown_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 350), (-1, 30)), "sampling operation", self.manager, self.ui_container)
        self.input_sampling_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in SamplingEnum],
            starting_option=str(self.config_params["sampling"]),
            relative_rect=pygame.Rect(
                220, 350, 250, 30),
            manager=self.manager,
            container=self.ui_container

        )

        self.input_crossover_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 380), (-1, 30)), "crossover operation", self.manager, self.ui_container)
        self.input_crossover = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in CrossoverEnum],
            starting_option=str(self.config_params["crossover"]),
            relative_rect=pygame.Rect(
                220, 380, 250, 30),
            manager=self.manager,
            container=self.ui_container
        )

        self.input_mutation_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 410), (-1, 30)), "mutation operation", self.manager, self.ui_container)
        self.input_mutation = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in MutationEnum],
            starting_option=str(self.config_params["mutation"]),
            relative_rect=pygame.Rect(
                220, 410, 250, 30),
            manager=self.manager,
            container=self.ui_container
        )

        self.input_algorithm_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 440), (-1, 30)), "algorithm", self.manager, self.ui_container)
        self.input_algorithm = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in AlgorithmEnum],
            starting_option=str(self.config_params["algorithm"]),
            relative_rect=pygame.Rect(
                220, 440, 250, 30),
            manager=self.manager,
            container=self.ui_container
        )

        self.input_eliminate_duplicates_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 470), (-1, 30)), "eliminate duplicates", self.manager, self.ui_container)
        self.input_eliminate_duplicates = pygame_gui.elements.UIDropDownMenu(
            options_list=["True", "False"],
            starting_option=str(self.config_params["eliminate_duplicates"]),
            relative_rect=pygame.Rect(
                220, 470, 250, 30),
            manager=self.manager,
            container=self.ui_container
        )

        self.input_objectves_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 500), (-1, 30)), "objectives", self.manager, self.ui_container)
        self.input_objectives = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect(220, 500, 250, 100),
            item_list=[item.value for item in ObjectiveEnum],
            allow_multi_select=True,
            container=self.ui_container,
            manager=self.manager,
            default_selection=self.config_params["objectives"]
        )

        self.switch_algorithm_mode_toggle = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 530), (-1, 30)), text='greedy mode',
            manager=self.manager, container=self.ui_container, object_id="toggle")
        self.switch_algorithm_mode_toggle.select() if self.config_params["use_greedy_assign"] else self.switch_algorithm_mode_toggle.unselect()

        self.evo_start_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 560), (-1, 30)), text='start',
            manager=self.manager, container=self.ui_container)

        self.input_algo_config_name = pygame_gui.elements.UITextEntryLine(
            container=self.ui_container, relative_rect=pygame.Rect((85, 560), (120, 30)),
            manager=self.manager, placeholder_text="config_name")
        if initial_config_name != "":
            self.input_algo_config_name.set_text(initial_config_name)

        self.evo_stop_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 590), (-1, 30)), text='stop',
            manager=self.manager, container=self.ui_container)
        self.evo_stop_button.disable()

        self.dropdown_pick_running_mode = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in RunningMode],
            starting_option=self.running_mode,
            relative_rect=pygame.Rect((20, 620), (200, 30)),
            manager=self.manager,
            container=self.ui_container,
        )
        self.greedy_to_moving_toggle = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (220, 620),
            (-1, 30)),
            text='greedy to moving', 
            manager=self.manager,
            object_id="toggle",
            container=self.ui_container)
        self.greedy_to_moving_toggle.select() if self.config_params["greedy_to_moving"] else self.greedy_to_moving_toggle.unselect()

        if creata_dropdown_pick_algo_config:
            self.dropdown_pick_algo_config = pygame_gui.elements.UIDropDownMenu(
                options_list=self.get_configs_for_current_network(),
                starting_option=self.get_configs_for_current_network()[0] if initial_config_name == "" else initial_config_name,
                relative_rect=pygame.Rect((20, 650), (280, 30)),
                manager=self.manager,
                container=self.ui_container,
            )
            
        self.dropdown_pick_result = pygame_gui.elements.UIDropDownMenu(
            options_list=self.get_results_for_current_config(),
            starting_option=self.get_results_for_current_config()[0],
            relative_rect=pygame.Rect((300, 650), (200, 30)),
            manager=self.manager,
            container=self.ui_container,
        )

    def on_dropdown_canvas_action_changed(self, event: pygame.event.Event):
        self.right_mouse_action = event.text
        if event.text == "remove" or event.text == "cell":
            self.apply_button.disable()
            self.input_antennas.disable()
            self.input_channel_bandwidth.disable()
            self.input_frequency.disable()
            self.input_standby_power.disable()
            self.input_static_power.disable()
            self.input_transmission_power.disable()
            self.dropdown_menu_pick_network.disable()
        else:
            self.apply_button.enable()
            self.dropdown_menu_pick_network.enable()
            self.input_antennas.enable()
            self.input_antennas.set_text(
                str(self.config_params[self.right_mouse_action]["antennas"]))

            self.input_channel_bandwidth.enable()
            self.input_channel_bandwidth.set_text(
                str(self.config_params[self.right_mouse_action]["channel_bandwidth"]))

            self.input_frequency.enable()
            self.input_frequency.set_text(
                str(self.config_params[self.right_mouse_action]["frequency"]))

            self.input_standby_power.enable()
            self.input_standby_power.set_text(
                str(self.config_params[self.right_mouse_action]["standby_power"]))

            self.input_static_power.enable()
            self.input_static_power.set_text(
                str(self.config_params[self.right_mouse_action]["static_power"]))

            self.input_transmission_power.enable()
            tx_power_dbm = self.watts_to_dbm(
                self.config_params[self.right_mouse_action]["tx_power"])
            self.input_transmission_power.set_text(str(tx_power_dbm))

    def switch_algorithm_mode(self):
        self.config_params["use_greedy_assign"] = not self.config_params["use_greedy_assign"]
        self.switch_algorithm_mode_toggle.select() if self.config_params["use_greedy_assign"] else self.switch_algorithm_mode_toggle.unselect()

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

    def save_current_moving_selection(self):
        name = self.input_moving_selection_name.get_text()
        if name == "":
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 100, 100),
                blocking=True,
                manager=self.manager,
                window_title="Missing file name",
                action_short_name="ok",
                action_long_desc="enter file name",
                visible=True,
            )
            return
        if not bool(self.moving_users):
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 100, 100),
                blocking=True,
                manager=self.manager,
                window_title="Missing user selection",
                action_short_name="ok",
                action_long_desc="press create moving selection first.",
                visible=True,

            )
            return

        if os.path.exists(
                "datastore/" + self.dropdown_menu_pick_network.selected_option + "/moving_selections"):
            directory_contents = os.listdir(
                "datastore/" + self.dropdown_menu_pick_network.selected_option + "/moving_selections")
            for item in directory_contents:
                if name + ".json" == item:
                    self.popup.kill()
                    self.popup = CustomConfirmationDialog(
                        rect=pygame.Rect(GAME_WIDTH + 20, 20, 100, 100),
                        blocking=True,
                        manager=self.manager,
                        window_title="file already exists ",
                        action_short_name="ok",
                        action_long_desc="choose different file name",
                        visible=True)
                    return

            # save current selection
            with open(self.current_save_result_directory + "/moving_selections/" + name + ".json", "w+", encoding="utf-8") as outfile:
                json.dump(self.moving_users, outfile)

            # update params
            self.config_params["moving_selection_name"] = name
            # update moving_selections dropdown
            self.dropdown_moving_selection.kill()
            self.dropdown_moving_selection = pygame_gui.elements.UIDropDownMenu(
                options_list=self.get_moving_selections_for_current_network(),
                starting_option=name,
                relative_rect=pygame.Rect((220, 150), (200, 30)),
                manager=self.manager,
                container=self.ui_container_live_config,
            )

    def start_evo(self):
        if self.running_mode == RunningMode.LIVE.value and self.dropdown_moving_selection.selected_option == "from file":
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 100, 100),
                blocking=True,
                manager=self.manager,
                window_title="missing selection name",
                action_short_name="ok",
                action_long_desc="save current selection with name or select one from file",
                visible=True,

            )
            return

        if os.path.exists("datastore/" + self.dropdown_menu_pick_network.selected_option):
            network_confg_name = self.input_algo_config_name.get_text()
            self.dropdown_pick_algo_config.selected_option = network_confg_name
            if self.iterations == - 1:
                self.iterations = 0
            if self.iterations == 0:
                # create new config folder
                # get all alread used config names

                directory_contents = os.listdir(
                    "datastore/" + self.dropdown_menu_pick_network.selected_option)

                if network_confg_name == "" or network_confg_name in directory_contents:
                    self.popup.kill()
                    self.popup = CustomConfirmationDialog(
                        rect=pygame.Rect(GAME_WIDTH + 20, 20, 100, 100),
                        blocking=True,
                        manager=self.manager,
                        window_title="config name invalid",
                        action_short_name="ok",
                        action_long_desc="no config name or name already exists",
                        visible=True)
                    return

                os.mkdir(
                    "datastore/" + self.dropdown_menu_pick_network.selected_option + "/" +
                    network_confg_name)

                # save current algorithm config
                with open(self.current_save_result_directory + "/" + network_confg_name + "/" + network_confg_name + ".json", "w+", encoding="utf-8") as outfile:
                    json.dump(self.config_params, outfile)

            # start optimization
            if not self.config_params["use_greedy_assign"]:
                optimization_process = multiprocessing.Process(target=start_optimization, args=(
                    int(self.config_params["pop_size"]),
                    int(self.config_params["n_offsprings"]),
                    int(self.config_params["n_generations"]),
                    "",
                    self.config_params["sampling"],
                    self.config_params["crossover"],
                    self.config_params["mutation"],
                    self.config_params["eliminate_duplicates"] == "True",
                    self.config_params["objectives"],
                    self.config_params["algorithm"],
                    self.son,
                    self.current_save_result_directory + "/" + network_confg_name + "/",
                    self.pymoo_message_queue,
                    self.editor_message_queue,
                    self.running_mode,
                    0.3,
                    0.3
                ))
                optimization_process.start()

            self.optimization_running = True
            self.disable_ui()
            
            if not self.script_mode:
                self.evo_stop_button.enable()
        else:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 100, 100),
                blocking=True,
                manager=self.manager,
                window_title="Missing network file",
                action_short_name="ok",
                action_long_desc="press save first to create required target folder",
                visible=True
            )

    def on_optimization_finished(self):
        # update algo param config dropdown
        self.optimization_running = False
        self.iterations += 1
        if self.running_mode == RunningMode.LIVE.value:

            json_data = json.dumps(self.objective_history)
            # Save objective JSON data to a file
            config_name = self.dropdown_pick_algo_config.selected_option
            file_path = f"{self.current_save_result_directory}/{config_name}/objectives_result_{self.iterations}.json"
            with open(file_path, 'w', encoding="utf-8") as file:
                file.write(json_data)

        print("finished " + str(self.iterations) + " iteraiton")

        self.reset_all_after_run()

        self.create_algo_param_ui_elements(creata_dropdown_pick_algo_config=True, initial_network_name=self.dropdown_menu_pick_network.selected_option, initial_config_name=config_name)
        self.enable_ui()

        if self.iterations < self.config_params["iterations"]:
            self.start_evo()
        else:
            # kill process when finished
            if self.script_mode:
                self.iterations = 0
                quit()
            else:
                self.iterations = 0

    def save_current_network(self):

        network_name = self.input_network_name.get_text()

        if network_name == "" or network_name == None:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 100, 100),
                blocking=True,
                manager=self.manager,
                window_title="Missing file name",
                action_short_name="ok",
                action_long_desc="sepecify network name first",
                visible=True,
            )
            return
        if network_name in get_network_folder_names():
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 100, 100),
                blocking=True,
                manager=self.manager,
                window_title="file name already exists",
                action_short_name="ok",
                action_long_desc="choose different network name",
                visible=True,)
            return
        # Get the current working directory
        current_directory = os.getcwd()
        # Path of the new folder
        new_folder_path = os.path.join(current_directory, "datastore/" + network_name)
        # Create the new folders
        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)
            os.mkdir(new_folder_path + "/moving_selections")
        # save adjacencies
        self.son.save_json_adjacency_graph_to_file(
            new_folder_path + "/" + network_name + "_adjacencies.json")

        self.config_params = default_algorithm_param_config
        self.apply_current_network_params_to_ui_and_graph(initial_network_name=network_name)

        self.ui_container_live_config.enable()

    def reset_queue_flags(self):
        self.queue_flags = {"activation_dict": False, "objective_space": False,
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
        # reset moving users vecors from file:
        self.reload_current_moving_selection()
        self.finished = False
        # reset topology -> load graph from file
        self.reload_current_network_graph()
        # empty message queues
        while not self.editor_message_queue.empty():
            self.editor_message_queue.get()
        while not self.pymoo_message_queue.empty():
            self.pymoo_message_queue.get()

    def run(self):
        while True:
            # set time per tick
            dt = self.clock.tick(30)/1000

            if self.script_mode:
                # start evo
                if self.iterations == -1:
                    self.start_evo()

            if self.running_mode == RunningMode.LIVE.value and self.optimization_running:
                self.dt_since_last_history_update += dt
                self.ticks_since_last_history_update += 1
                self.dt_since_last_evo_reset += dt
                self.dt_since_last_activation_profile_fetch += dt
                self.running_time_in_s += dt
                self.running_ticks += 1

                if not self.config_params["use_greedy_assign"]:

                    # trigger evo_reset if current activation profile violates son topology
                    self.trigger_evo_reset_invalid_activation_profile()

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

                    # react to queue messages
                    if self.queue_flags["activation_dict"] is not False and self.queue_flags["objective_space"] is not False:

                        self.activation = self.queue_flags["activation_dict"]
                        self.dt_since_last_activation_profile_fetch = 0

                    if self.queue_flags["n_gen_since_last_fetch"] is not False:
                        self.n_gen_since_last_fetch = self.queue_flags["n_gen_since_last_fetch"]

                    self.ngen_since_last_evo_reset = self.queue_flags["n_gen_since_last_reset"]

                    if self.ticks_since_last_history_update >= 30 and self.running_ticks <= self.config_params["running_time_in_s"] * 30:

                        self.update_objective_history()

                    # move users
                    self.move_some_users()
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
                        self.son.find_activation_profile_greedy_user(update_attributes=True)
                    if self.finished:
                        self.on_optimization_finished()

                if self.config_params["use_greedy_assign"]:

                    if self.ticks_since_last_history_update >= 30 and self.running_ticks <= self.config_params["running_time_in_s"] * 30:
                        self.update_objective_history()
                    # move users
                    self.move_some_users()
                    self.son.find_activation_profile_greedy_user(update_attributes=True)
                    if self.finished:
                        self.on_optimization_finished()

                # stop if running time is exceeded
                # if self.running_time_in_s >= self.config_params["running_time_in_s"]:
                # stop if running time is over but time is translated into frames
                if self.running_ticks == self.config_params["running_time_in_s"] * 30:
                    self.stop_evo()

                # reset queue_flags
                self.reset_queue_flags()

            # handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.background.get_rect().collidepoint(event.pos):
                        if event.button == 1:
                            self.selected_node_id = self.node_clicked(event.pos)
                            self.on_left_click(event.pos)
                        if event.button == 3:
                            self.on_right_click(event.pos)

                if event.type == pygame.MOUSEMOTION:
                    if self.dropdown_drag_inspect_menue.selected_option == "drag" and self.background.get_rect().collidepoint(event.pos):
                        if event.buttons[0] and self.selected_node_id:
                            self.node_drag(self.selected_node_id, event.pos)

                if event.type == pygame.MOUSEBUTTONUP:
                    if self.background.get_rect().collidepoint(event.pos):
                        self.selected_node_id = None

                # GIU events
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.switch_algorithm_mode_toggle:
                        self.switch_algorithm_mode()
                    if event.ui_element == self.apply_button:
                        self.apply_current_network_params_to_graph()
                    if event.ui_element == self.save_button:
                        self.save_current_network()
                    if event.ui_element == self.show_edges_checkbox:
                        self.onclick_show_edges_checkbox()
                    if event.ui_element == self.objectives_infobox_toggle:
                        self.show_objectives_info_box()
                    if event.ui_element == self.evo_start_button:
                        self.start_evo()
                    if event.ui_element == self.evo_stop_button:
                        self.force_stop_evo()
                    if event.ui_element == self.create_movement_selection_button:
                        self.initialize_moving_users()
                    if event.ui_element == self.save_moving_selection_button:
                        self.save_current_moving_selection()
                    if event.ui_element == self.show_moving_selection_toggle:
                        self.onshow_moving_selection_checkbox()
                    if event.ui_element == self.greedy_to_moving_toggle:
                        self.ontoggle_greedy_to_moving()
                if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    if event.ui_element == self.dropdown_menu:
                        self.on_dropdown_canvas_action_changed(event)
                    if event.ui_element == self.dropdown_menu_pick_network:
                        self.on_dropdown_pick_network_changed(event)
                    if event.ui_element == self.dropdown_moving_selection:
                        self.on_dropdown_moving_selection_changed(event)
                    if event.ui_element == self.input_crossover:
                        self.on_dropdown_input_algorithm_config_changed(event, "crossover")
                    if event.ui_element == self.input_mutation:
                        self.on_dropdown_input_algorithm_config_changed(event, "mutation")
                    if event.ui_element == self.input_sampling_dropdown:
                        self.on_dropdown_input_algorithm_config_changed(event, "sampling")
                    if event.ui_element == self.input_algorithm:
                        self.on_dropdown_input_algorithm_config_changed(event, "algorithm")
                    if event.ui_element == self.input_eliminate_duplicates:
                        self.on_dropdown_input_algorithm_config_changed(
                            event, "eliminate_duplicates")
                    if event.ui_element == self.dropdown_pick_algo_config:
                        self.on_dropdown_pick_algo_config_changed(event)
                    if event.ui_element == self.dropdown_pick_result:
                        self.on_dropdown_pick_result_changed(event)
                    if event.ui_element == self.dropdown_pick_running_mode:
                        self.on_dropdown_mode_changed(event)
                if event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION or event.type == pygame_gui.UI_SELECTION_LIST_DROPPED_SELECTION:
                    if event.ui_element == self.input_objectives:
                        self.on_selectionlist_input_objectives_changed(event)
                if event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
                    self.on_entry_changed(event)
                self.manager.process_events(event)

            self.draw_network()
            self.manager.update(dt)
            self.manager.draw_ui(self.display_surface)

            pygame.display.update()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        network_name = sys.argv[1]
        config_name = sys.argv[2]
        son = Son()
        main = Main(son, script_mode=True, network_name=network_name, config_name=config_name)
        main.run()
    else:
        son = Son()
        main = Main(son)
        main.run()
        # cProfile.run("main.run()", sort="tottime")
