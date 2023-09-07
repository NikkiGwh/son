
from copy import deepcopy
from gc import disable
from genericpath import isdir
import math
from pygame_gui._constants import UI_BUTTON_PRESSED
import pygame
import sys
from pygame_settings import *
import networkx as nx
import json
from son_main_script import BaseStationOrder, BinPackingType, CellOrderTwo, Son
import pygame_gui
import re
import os
import multiprocessing
import numpy as np
from math import cos, sin
from son_pymoo import AlgorithmEnum, CrossoverEnum, MutationEnum, ObjectiveEnum, RunningMode, SamplingEnum, start_optimization

dropdown_menue_options_list = ["macro", "micro", "femto", "pico", "cell", "remove"]
text_input_float_number_type_characters = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "."]
text_input_integer_number_type_characters = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

default_algorithm_param_config = {
    "pop_size": 100,
    "n_offsprings": 20,
    "n_generations": 10,
    "termination": "",
    "sampling": SamplingEnum.RANDOM_SAMPLING.value,
    "crossover": CrossoverEnum.UNIFORM_CROSSOVER.value,
    "mutation": MutationEnum.PM_MUTATION.value,
    "eliminate_duplicates": "True",
    "objectives": [ObjectiveEnum.AVG_DL_RATE.value, ObjectiveEnum.ENERGY_EFFICIENCY.value],
    "algorithm": AlgorithmEnum.NSGA2.value,
    "pick_rate_in_n_gen": 5,
    "pick_rate_in_s": 20,
    "moving_speed": 0.1,
    "reset_rate_in_s": 20,
    "reset_rate_in_ngen": 5,
    "reset_delay_in_s": 10,
    "moving_selection_percent": 30,
    "running_time_in_s": 600,
}


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
    def __init__(self, graph: Son) -> None:
        pygame.init()
        self.running_mode = RunningMode.LIVE.value
        self.use_greedy_assign = False
        self.topology_changed = False
        self.dt_since_last_history_update = 0
        self.optimization_running = False
        self.dt_since_last_activation_profile_fetch = 0
        self.n_gen_since_last_fetch = 0
        self.dt_since_last_evo_reset = 0
        self.objective_history = []
        self.ngen_since_last_evo_reset = 0
        self.running_time_in_s = 0
        self.selected_node_id = None
        self.moving_users = {}
        self.show_moving_users = False
        self.queue_flags = {"finished": False, "activation_dict": False,
                            "objective_space": False, "n_gen_since_last_fetch": False, "n_gen": False}
        self.current_save_result_directory = ""
        self.pymoo_message_queue = multiprocessing.Queue()
        self.editor_message_queue = multiprocessing.Queue()
        self.right_mouse_action = dropdown_menue_options_list[0]
        self.network_folder_name_list = get_network_folder_names()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.background = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.background.fill(pygame.colordict.THECOLORS["white"])
        self.clock = pygame.time.Clock()
        self.son = graph
        self.activation = {}
        max_x, max_y = self.get_max_x_y(son.graph)
        max_value = max(max_x, max_y)
        self.unit_size_x, self.unit_size_y = (
            math.trunc(GAME_WIDTH / max_value),
            math.trunc(GAME_HEIGHT / max_value))

        self.network_params_dic = son.network_node_params
        self.algorithm_param_dic = default_algorithm_param_config
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

        self.ui_container = pygame_gui.elements.UIPanel(
            pygame.Rect((GAME_WIDTH, 0), (WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT)),
            object_id="#ui_container")

        self.dropdown_menu = pygame_gui.elements.UIDropDownMenu(
            dropdown_menue_options_list,
            dropdown_menue_options_list[0],
            pygame.Rect((20, 20), (100, 30)), self.manager,
            container=self.ui_container)

        self.dropdown_menu_pick_network = pygame_gui.elements.UIDropDownMenu(
            options_list=self.network_folder_name_list,
            starting_option=self.network_folder_name_list[0],
            relative_rect=pygame.Rect((120, 20), (100, 30)),
            manager=self.manager,
            container=self.ui_container
        )

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
            object_id="#apply_button",
            text="hide", manager=self.manager,
            container=self.ui_container)
        self.show_edges_checkbox_active = True

        self.objectives_checkbox = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (370, 20),
            (50, 30)),
            object_id="#apply_button",
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
            initial_text=str(self.network_params_dic[self.right_mouse_action]["antennas"]))
        self.input_antennas.set_allowed_characters(text_input_float_number_type_characters)

        self.input_network_name = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((320, 50),
                        (100, 30)),
            self.manager, self.ui_container, placeholder_text="network name")

        self.input_wave_length_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 80), (-1, 30)), "wave length in m", self.manager, self.ui_container)
        self.input_wave_length = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 80),
                        (100, 30)),
            self.manager, self.ui_container, placeholder_text="wave length",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["wave_length"]))
        self.input_wave_length.set_allowed_characters(text_input_float_number_type_characters)

        self.save_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((320, 80), (100, 30)),
            object_id="#save_button", text='save', manager=self.manager, container=self.
            ui_container)

        self.input_channel_bandwidth_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 110), (-1, 30)), "channel bandwidth in HZ", self.manager, self.ui_container)
        self.input_channel_bandwidth = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 110), (100, 30)), self.manager, self.ui_container, placeholder_text="bandwidth",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["channel_bandwidth"]))
        self.input_channel_bandwidth.set_allowed_characters(text_input_float_number_type_characters)

        self.input_transmission_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 140), (-1, 30)), "transmission power in Watt", self.manager, self.ui_container)
        self.input_transmission_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 140), (100, 30)), self.manager, self.ui_container, placeholder_text="tx power",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["tx_power"]))
        self.input_transmission_power.set_allowed_characters(
            text_input_float_number_type_characters)

        self.input_static_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 170), (-1, 30)), "static power in Watt", self.manager, self.ui_container)
        self.input_static_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 170), (100, 30)), self.manager, self.ui_container, placeholder_text="static power",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["static_power"]))
        self.input_static_power.set_allowed_characters(text_input_float_number_type_characters)

        self.input_standby_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 200), (-1, 30)), "standby power in Watt", self.manager, self.ui_container)
        self.input_standby_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 200), (100, 30)), self.manager, self.ui_container, placeholder_text="standby power",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["standby_power"]))
        self.input_standby_power.set_allowed_characters(text_input_float_number_type_characters)

        self.input_frequency_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 230), (-1, 30)), "frequency in HZ", self.manager, self.ui_container)
        self.input_frequency = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 230), (100, 30)), self.manager, self.ui_container, placeholder_text="frequency",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["frequency"]))
        self.input_frequency.set_allowed_characters(text_input_float_number_type_characters)

        # algorithm params
        self.create_algo_param_ui_elements()

        # bottom UI
        self.bin_packing_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 530), (-1, 30)), object_id="#bin_packing_button", text='bin packing',
            manager=self.manager, container=self.ui_container)

        self.evo_start_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 560), (-1, 30)), text='start evo',
            manager=self.manager, container=self.ui_container)

        self.evo_stop_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 590), (-1, 30)), text='stop evo',
            manager=self.manager, container=self.ui_container)
        self.evo_stop_button.disable()

        self.dropdown_pick_mode = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in RunningMode],
            starting_option=self.running_mode,
            relative_rect=pygame.Rect((20, 620), (200, 30)),
            manager=self.manager,
            container=self.ui_container,
        )

        # live config elements
        self.create_live_param_ui_elements()
        self.ui_container_live_config.disable()

    def disable_ui(self):
        self.ui_container.disable()
        self.ui_container_live_config.disable()

    def enable_ui(self):
        self.ui_container.enable()
        self.ui_container_live_config.enable()

    def get_configs_for_current_network(self) -> list[str]:
        result_list = ["from file"]

        if self.dropdown_menu_pick_network.selected_option != "from file":
            directory_contents = os.listdir(
                "datastore/" + self.dropdown_menu_pick_network.selected_option)

            for item in directory_contents:
                if "algorithm_config" in item:
                    result_list.append(item)

        return result_list

    def get_moving_selections_for_current_network(self) -> list[str]:
        result_list = ["from file"]

        if self.dropdown_menu_pick_network.selected_option != "from file":
            directory_contents = os.listdir(
                "datastore/" + self.dropdown_menu_pick_network.selected_option + "/moving_selections")

            for item in directory_contents:
                result_list.append(item)

        return result_list

    def get_results_for_current_config(self) -> list[str]:
        result_list = ["from file"]

        if self.dropdown_menu_pick_network.selected_option != "from file" and self.dropdown_pick_algo_config.selected_option != "from file":
            directory_contents = os.listdir(
                "datastore/" + self.dropdown_menu_pick_network.selected_option + "/" + self.dropdown_pick_algo_config.selected_option)

            for item in directory_contents:
                if "ind_result" in item:
                    result_list.append(item)

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

        percentage = self.algorithm_param_dic["moving_selection_percent"]

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
        selection_list = np.random.choice(
            [1, 0],
            size=len(user_nodes),
            p=[percentage / 100, 1 - percentage / 100])
        selection_id_list = [x for index, x in enumerate(user_nodes) if selection_list[index] == 1]

        # initialize moving directions randomly
        deg = np.random.randint(0, 360, size=len(selection_id_list))
        for index, deg_value in enumerate(deg):
            v = self.rotate_vector_by_deg(np.array([1, 0]), deg_value)
            self.moving_users[selection_id_list[index]] = (v[0], v[1])

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
            (self.moving_users[user_node_id][0] * self.algorithm_param_dic["moving_speed"],
             self.moving_users[user_node_id][1] * self.algorithm_param_dic["moving_speed"]),
            update_network=False, initialize_edges=False)

        self.topology_changed = True

    def update_direction_for_user(self, user_node_id: str):

        while (self.check_direction_valid(user_node_id) is False):
            new_direction_numpy = self.rotate_vector_by_deg(
                np.array(self.moving_users[user_node_id]), 30)

            self.moving_users[user_node_id] = (new_direction_numpy[0], new_direction_numpy[1])

    def check_direction_valid(self, user_node_id: str):

        for _, edge in enumerate(self.son.graph[user_node_id].items()):

            next_rssi = self.son.get_rssi_cell(
                user_node_id, (user_node_id, edge[0]),
                moving_vector=(self.moving_users[user_node_id][0] * self.algorithm_param_dic
                               ["moving_speed"],
                               self.moving_users[user_node_id][1] * self.algorithm_param_dic
                               ["moving_speed"]))

            if next_rssi > self.son.min_rssi:
                return True

        return False

    def update_objective_history(self):
        current_energy_efficiency = self.son.get_energy_efficiency()
        current_avg_dl_datarate = self.son.get_average_dl_datarate()
        self.objective_history.append(
            (round(self.running_time_in_s, 2), current_energy_efficiency, current_avg_dl_datarate))
        self.dt_since_last_history_update = 0

    def trigger_evo_reset_invalid_activation_profile(self):
        # invoke evo_reset if threshhold is met
        if self.dt_since_last_evo_reset >= self.algorithm_param_dic["reset_rate_in_s"] or self.ngen_since_last_evo_reset >= self.algorithm_param_dic["reset_rate_in_ngen"] and self.dt_since_last_evo_reset >= self.algorithm_param_dic["reset_delay_in_s"]:
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
                     "reset": True, "send_results": False})

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
            return
        else:
            self.objectives_checkbox_active = True

        if len(self.son.graph.nodes) == 0 or len(self.son.graph.edges) == 0:
            self.info_text_box_objectives.set_text("There is no valid network yet.")
            self.info_text_box_objectives.show()
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

    def onclick_show_edges_checkbox(self):
        self.show_edges_checkbox_active = not self.show_edges_checkbox_active
        self.show_edges_checkbox.set_text("hide" if self.show_edges_checkbox_active else "show")

    def show_moving_selection_checkbox(self):
        self.show_moving_users = not self.show_moving_users
        self.show_moving_selection_button.set_text("hide" if self.show_moving_users else "show")

    def apply_params_from_text_inputs(self):
        self.son.apply_network_node_attributes(self.network_params_dic)

    def apply_params_from_json_file(self, file_name: str):
        config_file_path = os.getcwd() + "/datastore/" + self.dropdown_menu_pick_network.selected_option + "/" + file_name
        # Opening JSON file
        with open(config_file_path, 'r', encoding="utf-8") as openfile:
            # Reading from json file
            json_string_params = json.load(openfile)
            self.network_params_dic = json_string_params
            # apply params on network
            self.son.apply_network_node_attributes(self.network_params_dic)
            # update text inputs ui
            self.input_antennas.set_text(
                str(self.network_params_dic[self.right_mouse_action]["antennas"]))
            self.input_channel_bandwidth.set_text(
                str(self.network_params_dic[self.right_mouse_action]["channel_bandwidth"]))
            self.input_frequency.set_text(
                str(self.network_params_dic[self.right_mouse_action]["frequency"]))
            self.input_frequency.disable()
            self.input_standby_power.set_text(
                str(self.network_params_dic[self.right_mouse_action]["standby_power"]))
            self.input_static_power.set_text(
                str(self.network_params_dic[self.right_mouse_action]["static_power"]))
            self.input_transmission_power.set_text(
                str(self.network_params_dic[self.right_mouse_action]["tx_power"]))
            self.input_wave_length.set_text(
                str(self.network_params_dic[self.right_mouse_action]["wave_length"]))

    def apply_adjacencies_from_json_file(self, file_name):
        layout_file_path = os.getcwd() + "/datastore/" + self.dropdown_menu_pick_network.selected_option + "/" + file_name
        self.son.load_graph_from_json_adjacency_file(layout_file_path, True)

    def on_entry_changed(self, event: pygame.Event):

        pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)$'
        if not re.match(pattern, event.text):
            event.text = 0
        if event.ui_element == self.input_transmission_power:
            self.network_params_dic[self.right_mouse_action]["tx_power"] = float(event.text)
        if event.ui_element == self.input_antennas:
            self.network_params_dic[self.right_mouse_action]["antennas"] = float(event.text)
        if event.ui_element == self.input_channel_bandwidth:
            self.network_params_dic[self.right_mouse_action]["bandwidth"] = float(event.text)
        if event.ui_element == self.input_frequency:
            self.network_params_dic[self.right_mouse_action]["frequency"] = float(event.text)
        if event.ui_element == self.input_standby_power:
            self.network_params_dic[self.right_mouse_action]["standby_power"] = float(event.text)
        if event.ui_element == self.input_static_power:
            self.network_params_dic[self.right_mouse_action]["static_power"] = float(event.text)
        if event.ui_element == self.input_wave_length:
            self.network_params_dic[self.right_mouse_action]["wave_length"] = float(event.text)
        if event.ui_element == self.input_n_generations:
            self.algorithm_param_dic["n_generations"] = float(event.text)
        if event.ui_element == self.input_n_offsprings:
            self.algorithm_param_dic["n_offsprings"] = float(event.text)
        if event.ui_element == self.input_pop_size:
            self.algorithm_param_dic["pop_size"] = float(event.text)
        if event.ui_element == self.input_resetting_rate:
            self.algorithm_param_dic["reset_rate_in_ngen"] = float(event.text)
        if event.ui_element == self.input_picking_rate:
            self.algorithm_param_dic["pick_rate_in_n_gen"] = float(event.text)
        if event.ui_element == self.input_velocity:
            self.algorithm_param_dic["moving_speed"] = float(event.text)
        if event.ui_element == self.input_create_movement_selection_percentage:
            self.algorithm_param_dic["moving_selection_percent"] = float(event.text)

    def on_dropdown_pick_network_changed(self, event: pygame.Event):
        if event.text != "from file":
            # Get the current working directory
            current_directory = os.getcwd() + "/datastore/" + event.text
            # List all files and directories in the current directory
            directory_contents = os.listdir(current_directory)
            config_file_name = ""
            adjacencies_file_name = ""
            for item in directory_contents:
                if "network_config" in item:
                    config_file_name = item
                if "adjacencies" in item:
                    adjacencies_file_name = item

            # apply parameter config
            self.apply_params_from_json_file(config_file_name)
            # load graph layout
            self.apply_adjacencies_from_json_file(adjacencies_file_name)
            # update algorithm config dropdown
            self.dropdown_pick_algo_config.kill()
            self.dropdown_pick_algo_config = pygame_gui.elements.UIDropDownMenu(
                options_list=self.get_configs_for_current_network(),
                starting_option=self.get_configs_for_current_network()[0],
                relative_rect=pygame.Rect((320, 290), (200, 30)),
                manager=self.manager,
                container=self.ui_container,
            )
            # update result dropdown
            self.dropdown_pick_result.kill()
            self.dropdown_pick_result = pygame_gui.elements.UIDropDownMenu(
                options_list=self.get_results_for_current_config(),
                starting_option=self.get_results_for_current_config()[0],
                relative_rect=pygame.Rect((320, 320), (200, 30)),
                manager=self.manager,
                container=self.ui_container,
            )
            # update moving_selections dropdown
            self.dropdown_moving_selection.kill()
            self.dropdown_moving_selection = pygame_gui.elements.UIDropDownMenu(
                options_list=self.get_moving_selections_for_current_network(),
                starting_option=self.get_moving_selections_for_current_network()[0],
                relative_rect=pygame.Rect((220, 150), (200, 30)),
                manager=self.manager,
                container=self.ui_container_live_config,
            )
            self.ui_container_live_config.enable()
        else:
            self.dropdown_pick_algo_config.disable()
            self.dropdown_pick_result.disable()
            self.ui_container_live_config.disable()
            self.son = Son()
        self.moving_users = {}

    def on_dropdown_input_algorithm_config_changed(self, event: pygame.Event, param_key: str):
        self.algorithm_param_dic[param_key] = event.text

    def on_selectionlist_input_changed(self, event: pygame.Event):
        self.algorithm_param_dic["objectives"] = self.input_objectives.get_multi_selection()

        if (len(self.algorithm_param_dic["objectives"]) == 0):
            self.evo_start_button.disable()
        else:
            self.evo_start_button.enable()

    def on_dropdown_pick_algo_config_changed(self, event: pygame.Event):
        if event.text == "from file":
            self.dropdown_pick_result.disable()
        else:
            with open("datastore/" + self.dropdown_menu_pick_network.selected_option + "/" + event.text + "/" + event.text + ".json", 'r', encoding="utf-8") as openfile:
                # Reading from json file
                self.algorithm_param_dic = json.load(openfile)
                # update all algorithm config inputs
                self.input_algorithm.kill()
                self.input_crossover.kill()
                self.input_objectives.kill()
                self.input_sampling_dropdown.kill(),
                self.input_mutation.kill()
                self.input_n_generations.kill()
                self.input_n_offsprings.kill()
                self.input_pop_size.kill()
                self.input_eliminate_duplicates.kill()
                self.dropdown_pick_result.kill()
                self.create_algo_param_ui_elements(creata_dropdown_pick_algo_config=False)

                self.ui_container_live_config.kill()
                self.create_live_param_ui_elements()

    def on_dropdown_moving_selection_changed(self, event: pygame.Event):
        if event.text == "from file":
            self.moving_users = {}
        else:
            with open("datastore/" + self.dropdown_menu_pick_network.selected_option + "/moving_selections/" + event.text, 'r', encoding="utf-8") as openfile:
                # Reading from json file
                self.moving_users = json.load(openfile)
                # update all algorithm config inputs

    def on_dropdown_pick_result_changed(self, event: pygame.Event):
        # Opening json file
        if event.text != "from file":
            self.son.load_graph_from_json_adjacency_file(
                "datastore/" + self.dropdown_menu_pick_network.selected_option + "/" + self.
                dropdown_pick_algo_config.selected_option + "/" + event.text, True)
        # TODO disable some other elements ?

    def on_dropdown_mode_changed(self, event: pygame.Event):

        self.running_mode = event.text

    def create_live_param_ui_elements(self):
        self.ui_container_live_config = pygame_gui.elements.UIPanel(
            pygame.Rect((GAME_WIDTH, 650), (WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT - 650)),
            object_id="#ui_container_live_config")

        self.input_resetting_rate_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 0), (-1, 30)), "resetting rate in nGen", self.manager, self.ui_container_live_config)
        self.input_resetting_rate = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 0),
                        (100, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="resetting rate",
            initial_text=str(self.algorithm_param_dic["reset_rate_in_ngen"]))
        self.input_resetting_rate.set_allowed_characters(text_input_integer_number_type_characters)

        self.input_picking_rate_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 30), (-1, 30)), "picking rate in nGen", self.manager, self.ui_container_live_config)
        self.input_picking_rate = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 30),
                        (100, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="picking rate",
            initial_text=str(self.algorithm_param_dic["pick_rate_in_n_gen"]))
        self.input_picking_rate.set_allowed_characters(text_input_integer_number_type_characters)

        self.input_velocity_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 60), (-1, 30)), "velocity", self.manager, self.ui_container_live_config)
        self.input_velocity = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 60),
                        (100, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="velocity",
            initial_text=str(self.algorithm_param_dic["moving_speed"]))
        self.input_velocity.set_allowed_characters(text_input_float_number_type_characters)

        self.create_movement_selection_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, 90),
                                      (-1, 30)),
            text='create selection %', manager=self.manager,
            container=self.ui_container_live_config)
        self.input_create_movement_selection_percentage = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 90),
                        (100, 30)),
            self.manager, self.ui_container_live_config,
            initial_text=str(self.algorithm_param_dic["moving_selection_percent"]))
        self.input_create_movement_selection_percentage.set_allowed_characters(
            text_input_float_number_type_characters)
        self.show_moving_selection_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (330, 90), (-1, 30)), text="show", manager=self.manager, container=self.ui_container_live_config)

        self.save_moving_selection_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, 120),
                                      (-1, 30)),
            text='save current selection', manager=self.manager,
            container=self.ui_container_live_config)
        self.input_moving_selection_name = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 120),
                        (150, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="selection name")

        self.select_moving_selection_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 150), (-1, 30)), "moving selections", self.manager, self.ui_container_live_config)

        self.dropdown_moving_selection = pygame_gui.elements.UIDropDownMenu(
            options_list=self.get_moving_selections_for_current_network(),
            starting_option=self.get_moving_selections_for_current_network()[0],
            relative_rect=pygame.Rect((220, 150), (200, 30)),
            manager=self.manager,
            container=self.ui_container_live_config,
        )

    def create_algo_param_ui_elements(self, creata_dropdown_pick_algo_config=True):
        self.input_pop_size_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 260), (-1, 30)), "population size", self.manager, self.ui_container)
        self.input_pop_size = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 260), (100, 30)), self.manager, self.ui_container, placeholder_text="pop_size",
            initial_text=str(self.algorithm_param_dic["pop_size"]))
        self.input_pop_size.set_allowed_characters(text_input_float_number_type_characters)

        self.input_n_offsprings_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 290), (-1, 30)), "number of offsprings", self.manager, self.ui_container)
        self.input_n_offsprings = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 290), (100, 30)), self.manager, self.ui_container, placeholder_text="n_offsprings",
            initial_text=str(self.algorithm_param_dic["n_offsprings"]))
        self.input_n_offsprings.set_allowed_characters(text_input_float_number_type_characters)

        if creata_dropdown_pick_algo_config:
            self.dropdown_pick_algo_config = pygame_gui.elements.UIDropDownMenu(
                options_list=self.get_configs_for_current_network(),
                starting_option=self.get_configs_for_current_network()[0],
                relative_rect=pygame.Rect((320, 290), (200, 30)),
                manager=self.manager,
                container=self.ui_container,
            )

        self.input_n_generations_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 320), (-1, 30)), "number of generations", self.manager, self.ui_container)
        self.input_n_generations = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 320), (100, 30)), self.manager, self.ui_container, placeholder_text="n_generations",
            initial_text=str(self.algorithm_param_dic["n_generations"]))
        self.input_n_generations.set_allowed_characters(text_input_float_number_type_characters)

        self.dropdown_pick_result = pygame_gui.elements.UIDropDownMenu(
            options_list=self.get_results_for_current_config(),
            starting_option=self.get_results_for_current_config()[0],
            relative_rect=pygame.Rect((320, 320), (200, 30)),
            manager=self.manager,
            container=self.ui_container,
        )

        self.input_sampling_dropdown_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 350), (-1, 30)), "sampling operation", self.manager, self.ui_container)
        self.input_sampling_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in SamplingEnum],
            starting_option=str(self.algorithm_param_dic["sampling"]),
            relative_rect=pygame.Rect(
                220, 350, 250, 30),
            manager=self.manager,
            container=self.ui_container

        )

        self.input_crossover_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 380), (-1, 30)), "crossover operation", self.manager, self.ui_container)
        self.input_crossover = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in CrossoverEnum],
            starting_option=str(self.algorithm_param_dic["crossover"]),
            relative_rect=pygame.Rect(
                220, 380, 250, 30),
            manager=self.manager,
            container=self.ui_container
        )

        self.input_mutation_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 410), (-1, 30)), "mutation operation", self.manager, self.ui_container)
        self.input_mutation = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in MutationEnum],
            starting_option=str(self.algorithm_param_dic["mutation"]),
            relative_rect=pygame.Rect(
                220, 410, 250, 30),
            manager=self.manager,
            container=self.ui_container
        )

        self.input_algorithm_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 440), (-1, 30)), "algorithm", self.manager, self.ui_container)
        self.input_algorithm = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in AlgorithmEnum],
            starting_option=str(self.algorithm_param_dic["algorithm"]),
            relative_rect=pygame.Rect(
                220, 440, 250, 30),
            manager=self.manager,
            container=self.ui_container
        )

        self.input_eliminate_duplicates_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 470), (-1, 30)), "eliminate duplicates", self.manager, self.ui_container)
        self.input_eliminate_duplicates = pygame_gui.elements.UIDropDownMenu(
            options_list=["True", "False"],
            starting_option=str(self.algorithm_param_dic["eliminate_duplicates"]),
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
            default_selection=self.algorithm_param_dic["objectives"]
        )

    def on_dropdown_changed(self, event: pygame.Event):
        self.right_mouse_action = event.text
        if event.text == "remove" or event.text == "cell":
            self.apply_button.disable()
            self.input_antennas.disable()
            self.input_channel_bandwidth.disable()
            self.input_frequency.disable()
            self.input_standby_power.disable()
            self.input_static_power.disable()
            self.input_wave_length.disable()
            self.input_transmission_power.disable()
            self.dropdown_menu_pick_network.disable()
        else:
            self.apply_button.enable()
            self.dropdown_menu_pick_network.enable()
            self.input_antennas.enable()
            self.input_antennas.set_text(
                str(self.network_params_dic[self.right_mouse_action]["antennas"]))

            self.input_channel_bandwidth.enable()
            self.input_channel_bandwidth.set_text(
                str(self.network_params_dic[self.right_mouse_action]["channel_bandwidth"]))

            self.input_frequency.enable()
            self.input_frequency.set_text(
                str(self.network_params_dic[self.right_mouse_action]["frequency"]))

            self.input_standby_power.enable()
            self.input_standby_power.set_text(
                str(self.network_params_dic[self.right_mouse_action]["standby_power"]))

            self.input_static_power.enable()
            self.input_static_power.set_text(
                str(self.network_params_dic[self.right_mouse_action]["static_power"]))

            self.input_transmission_power.enable()
            self.input_transmission_power.set_text(
                str(self.network_params_dic[self.right_mouse_action]["tx_power"]))

            self.input_wave_length.enable()
            self.input_wave_length.set_text(
                str(self.network_params_dic[self.right_mouse_action]["wave_length"]))

    def start_bin_packing(self):
        # self.son.find_activation_profile_greedy_user(update_attributes=True)
        self.use_greedy_assign = True
        self.start_evo()

    def stop_evo(self):
        self.editor_message_queue.put(
            {"terminate": True, "graph": False, "reset": False, "send_results": False})

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

            self.current_save_result_directory = "datastore/" + self.dropdown_menu_pick_network.selected_option + "/moving_selections/"
            # save current selection
            with open(self.current_save_result_directory + name + ".json", "w+", encoding="utf-8") as outfile:
                json.dump(self.moving_users, outfile)

            # update moving_selections dropdown
            self.dropdown_moving_selection.kill()
            self.dropdown_moving_selection = pygame_gui.elements.UIDropDownMenu(
                options_list=self.get_moving_selections_for_current_network(),
                starting_option=name + ".json",
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
            result_directory_count = 1
            directory_contents = os.listdir(
                "datastore/" + self.dropdown_menu_pick_network.selected_option)
            for item in directory_contents:
                if "algorithm_config" in item:
                    result_directory_count += 1
            network_confg_name = "algorithm_config_" + str(
                result_directory_count) + "_" + self.running_mode
            if self.use_greedy_assign:
                network_confg_name += "_greedy"

            os.mkdir(
                "datastore/" + self.dropdown_menu_pick_network.selected_option + "/" +
                network_confg_name)

            self.current_save_result_directory = "datastore/" + self.dropdown_menu_pick_network.selected_option + \
                "/" + network_confg_name
            # save current algorithm config
            with open(self.current_save_result_directory + "/" + network_confg_name + ".json", "w+", encoding="utf-8") as outfile:
                if self.running_mode == RunningMode.LIVE.value:
                    self.algorithm_param_dic["moving_selection_name"] = self.dropdown_moving_selection.selected_option
                    json.dump(self.algorithm_param_dic, outfile)
                else:
                    json.dump(self.algorithm_param_dic, outfile)

            if not self.use_greedy_assign:
                optimization_process = multiprocessing.Process(target=start_optimization, args=(
                    int(self.algorithm_param_dic["pop_size"]),
                    int(self.algorithm_param_dic["n_offsprings"]),
                    int(self.algorithm_param_dic["n_generations"]),
                    "",
                    self.algorithm_param_dic["sampling"],
                    self.algorithm_param_dic["crossover"],
                    self.algorithm_param_dic["mutation"],
                    self.algorithm_param_dic["eliminate_duplicates"] == "True",
                    self.algorithm_param_dic["objectives"],
                    self.algorithm_param_dic["algorithm"],
                    self.son,
                    self.current_save_result_directory + "/",
                    self.pymoo_message_queue,
                    self.editor_message_queue,
                    self.running_mode,
                    0.3,
                    0.3
                ))
                optimization_process.start()

            self.optimization_running = True
            self.disable_ui()
            self.evo_stop_button.enable()
        else:
            self.use_greedy_assign = False
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
        self.use_greedy_assign = False
        if self.running_mode == RunningMode.LIVE.value:

            json_data = json.dumps(self.objective_history)
            # Save JSON data to a file
            file_path = self.current_save_result_directory + "/objectives_result.json"
            with open(file_path, 'w', encoding="utf-8") as file:
                file.write(json_data)

        self.evo_stop_button.disable()
        self.dropdown_pick_algo_config.kill()
        self.dropdown_pick_algo_config = pygame_gui.elements.UIDropDownMenu(
            options_list=self.get_configs_for_current_network(),
            starting_option=self.get_configs_for_current_network()[-1],
            relative_rect=pygame.Rect((320, 290), (200, 30)),
            manager=self.manager,
            container=self.ui_container,
        )
        # update reults dropdown dropdown
        self.dropdown_pick_result.kill()
        self.dropdown_pick_result = pygame_gui.elements.UIDropDownMenu(
            options_list=self.get_results_for_current_config(),
            starting_option=self.get_results_for_current_config()[0],
            relative_rect=pygame.Rect((320, 320), (200, 30)),
            manager=self.manager,
            container=self.ui_container,
        )
        self.reset_all_after_run()
        self.enable_ui()

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
            self.popup.show()
            return
        # Get the current working directory
        current_directory = os.getcwd()
        # Path of the new folder
        new_folder_path = os.path.join(current_directory, "datastore/" + network_name)
        # Create the new folder
        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)
            os.mkdir(new_folder_path + "/moving_selections")
        # save adjacencies
        self.son.save_json_adjacency_graph_to_file(
            new_folder_path + "/" + network_name + "_adjacencies.json")
        # save current network parameter config
        with open(new_folder_path + "/" + network_name + "_network_config.json", "w+", encoding="utf-8") as outfile:
            json.dump(self.network_params_dic, outfile)
            # apply current network params to son object
            self.son.apply_network_node_attributes(self.network_params_dic)

        # update network pick dropdown
        self.dropdown_menu_pick_network.kill()
        self.network_folder_name_list = get_network_folder_names()
        self.dropdown_menu_pick_network = pygame_gui.elements.UIDropDownMenu(
            options_list=self.network_folder_name_list,
            starting_option=network_name,
            relative_rect=pygame.Rect((120, 20), (100, 30)),
            manager=self.manager,
            container=self.ui_container
        )
        # activate ui_live container
        self.ui_container_live_config.enable()

    def reset_queue_flags(self):
        self.queue_flags = {"finished": False, "activation_dict": False,
                            "objective_space": False, "n_gen_since_last_fetch": False, "n_gen": False}

    def reset_all_after_run(self):
        self.reset_queue_flags()
        self.activation = {}
        self.moving_users = []
        self.use_greedy_assign = False
        self.topology_changed = False
        self.dt_since_last_history_update = 0
        self.optimization_running = False
        self.dt_since_last_activation_profile_fetch = 0
        self.n_gen_since_last_fetch = 0
        self.dt_since_last_evo_reset = 0
        self.objective_history = []
        self.ngen_since_last_evo_reset = 0
        self.running_time_in_s = 0
        self.selected_node_id = None
        self.moving_users = {}
        self.show_moving_users = False

    def run(self):
        while True:
            # set time per tick
            dt = self.clock.tick(30)/1000

            if self.running_mode == RunningMode.LIVE.value and self.optimization_running:
                self.dt_since_last_history_update += dt
                self.dt_since_last_evo_reset += dt
                self.dt_since_last_activation_profile_fetch += dt
                self.running_time_in_s += dt

                if not self.use_greedy_assign:
                    # send fetch data fetch request to process queue
                    if self.n_gen_since_last_fetch >= self.algorithm_param_dic["pick_rate_in_n_gen"] or self.dt_since_last_activation_profile_fetch >= self.algorithm_param_dic["pick_rate_in_s"]:
                        self.editor_message_queue.put(
                            {"terminate": False, "graph": False, "reset": False, "send_results": True})

                    # trigger evo_reset if current activation profile violates son topology
                    self.trigger_evo_reset_invalid_activation_profile()

                    # read message queue
                    while self.pymoo_message_queue.empty() is False:
                        callback_obj = self.pymoo_message_queue.get()
                        if callback_obj["finished"] == True:
                            # update dropdowns after normal completion and static mode
                            self.queue_flags["finished"] = True

                        if self.running_mode == RunningMode.LIVE.value:
                            if callback_obj["activation_dict"] is not False and callback_obj["objective_space"] is not False:
                                self.queue_flags["activation_dict"] = callback_obj["activation_dict"]
                                self.queue_flags["objective_space"] = callback_obj["objective_space"]

                            if callback_obj["n_gen_since_last_fetch"] is not False:
                                self.queue_flags["n_gen_since_last_fetch"] = callback_obj["n_gen_since_last_fetch"]

                            if callback_obj["n_gen"] is not False:
                                self.queue_flags["n_gen"] = callback_obj["n_gen"]

                        # react to queue messages
                        if self.queue_flags["activation_dict"] is not False and self.queue_flags["objective_space"] is not False:

                            self.activation = self.queue_flags["activation_dict"]
                            self.dt_since_last_activation_profile_fetch = 0

                        if self.queue_flags["n_gen_since_last_fetch"] is not False:
                            self.n_gen_since_last_fetch = self.queue_flags["n_gen_since_last_fetch"]

                        if self.queue_flags["n_gen"] is not False:
                            self.ngen_since_last_evo_reset = self.queue_flags["n_gen"]

                    if self.dt_since_last_history_update >= 1:
                        self.update_objective_history()
                    # move users
                    self.move_some_users()
                    # apply current activation
                    if len(self.activation) > 0:
                        self.activation = self.son.apply_activation_dict(
                            self.activation, update_network_attributes=True)
                    else:
                        # if no activation profile is present -> use greedy approach for all useres
                        self.son.find_activation_profile_greedy_user(update_attributes=True)
                    if self.queue_flags["finished"]:
                        self.on_optimization_finished()

                    # reset queue flags
                    self.reset_queue_flags()

                if self.use_greedy_assign:
                    if self.dt_since_last_history_update >= 1:
                        self.update_objective_history()
                    # move users
                    self.move_some_users()
                    self.son.find_activation_profile_greedy_user(update_attributes=True)
                    if self.queue_flags["finished"]:
                        self.on_optimization_finished()

                # stop if running time is exceeded
                if self.running_time_in_s >= self.algorithm_param_dic["running_time_in_s"]:
                    self.stop_evo()

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
                    if event.ui_element == self.bin_packing_button:
                        self.start_bin_packing()
                    if event.ui_element == self.apply_button:
                        self.apply_params_from_text_inputs()
                    if event.ui_element == self.save_button:
                        self.save_current_network()
                    if event.ui_element == self.show_edges_checkbox:
                        self.onclick_show_edges_checkbox()
                    if event.ui_element == self.objectives_checkbox:
                        self.show_objectives_info_box()
                    if event.ui_element == self.evo_start_button:
                        self.start_evo()
                    if event.ui_element == self.evo_stop_button:
                        self.stop_evo()
                    if event.ui_element == self.create_movement_selection_button:
                        self.initialize_moving_users()
                    if event.ui_element == self.save_moving_selection_button:
                        self.save_current_moving_selection()
                    if event.ui_element == self.show_moving_selection_button:
                        self.show_moving_selection_checkbox()
                if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    if event.ui_element == self.dropdown_menu:
                        self.on_dropdown_changed(event)
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
                    if event.ui_element == self.dropdown_pick_mode:
                        self.on_dropdown_mode_changed(event)
                if event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION or event.type == pygame_gui.UI_SELECTION_LIST_DROPPED_SELECTION:
                    if event.ui_element == self.input_objectives:
                        self.on_selectionlist_input_changed(event)
                if event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
                    self.on_entry_changed(event)
                self.manager.process_events(event)

            self.draw_network()
            self.manager.update(dt)
            self.manager.draw_ui(self.display_surface)

            pygame.display.update()


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


if __name__ == "__main__":
    son = Son()
    main = Main(son)
    main.run()
