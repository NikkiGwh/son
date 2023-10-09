from pygame_gui._constants import UI_BUTTON_PRESSED
import pygame
import sys
import pygame_gui
import os
import cProfile
import re
from network_simulation_script import ErrorEnum, Network_Simulation_State, default_algorithm_param_config
from pygame_settings import *
from son_main_script import NodeType, Son
from son_pymoo import AlgorithmEnum, CrossoverEnum, MutationEnum, ObjectiveEnum, RunningMode, SamplingEnum


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
class Editor():
    def __init__(self, net_sim: Network_Simulation_State) -> None:
        pygame.init()
        self.gui_ticks = 0
        self.gui_fps = 30
        self.selected_node_id = None
        self.show_moving_users = False
        self.objectives_checkbox_active = False
        self.right_mouse_action = dropdown_menue_options_list[0]
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.background = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.background.fill(pygame.colordict.THECOLORS["white"])
        self.clock = pygame.time.Clock()
        self.net_sim = net_sim
        self.net_sim.AddSubscribersForFinishedEvent(self.on_optimization_finished_callback)
        self.one_fps_sim_mode = True

        # networkx 1 = 1m I want max 10 km -> 0.1 on screen equals 1 m
        self.unit_size_x, self.unit_size_y = (
            GAME_WIDTH / 10000,
            GAME_HEIGHT / 10000)
        # GUI
        self.manager = pygame_gui.UIManager((WINDOW_WIDTH, WINDOW_HEIGHT), './theme.json')
        self.ui_container = pygame_gui.elements.UIPanel(
            pygame.Rect((GAME_WIDTH, 0), (WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT)),
            object_id="#ui_container")
        self.ui_container_live_config = pygame_gui.elements.UIPanel(
            pygame.Rect((GAME_WIDTH, 680), (WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT - 680)),
            object_id="#ui_container_live_config")
        self.popup = CustomConfirmationDialog(
            rect=pygame.Rect(GAME_WIDTH + 20, 20, 260, 200),
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

    def disable_ui(self):
        self.ui_container.disable()
        self.ui_container_live_config.disable()

    def enable_ui(self):
        self.ui_container.enable()
        self.ui_container_live_config.enable()

    def draw_network(self):
        self.display_surface.blit(self.background, (0, 0))
        self.draw_edges()
        self.draw_nodes()

    def draw_edges(self):
        for _, edge in enumerate(self.net_sim.son.graph.edges.data()):
            activation = "active" if edge[2]["active"] == True else "inactive"

            if self.show_edges_checkbox_active or activation == "active":
                pygame.draw.line(
                    self.display_surface, pygame.colordict.THECOLORS
                    [style["edges"]["colors"][activation]],
                    (self.net_sim.son.graph.nodes[edge[0]]["pos_x"] * self.unit_size_x, self.net_sim.son.graph.nodes
                     [edge[0]]["pos_y"] * self.unit_size_y),
                    (self.net_sim.son.graph.nodes[edge[1]]["pos_x"] * self.unit_size_x, self.net_sim.son.graph.nodes
                     [edge[1]]["pos_y"] * self.unit_size_y),
                    style["edges"]["sizes"]["edge_width"][activation])

    def draw_nodes(self):

        for _, node in enumerate(self.net_sim.son.graph.nodes.data()):
            node_type = node[1]["type"]
            activation = False if node[1]["type"] != "cell" and node[1]["active"] == False else True

            if self.show_moving_users:
                node_color_key = "inactive" if not activation else node_type if node[
                    0] not in self.net_sim.moving_users else "moving"
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
        for _, node in enumerate(self.net_sim.son.graph.nodes.data()):
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
        self.net_sim.son.move_node_by_pos(
            node_id, (target_x, target_y),
            update_network=False)
        self.net_sim.topology_changed = True

    def onclick_create_moving_users(self):
        if not self.net_sim.initialize_moving_users():
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 260, 200),
                blocking=True,
                manager=self.manager,
                window_title="No users",
                action_short_name="ok",
                action_long_desc="add some users",
                visible=True,
            )

    def on_left_click(self, target_pos: tuple[int, int]):
        node_id = self.selected_node_id
        if node_id is not None and self.dropdown_drag_inspect_menue.selected_option == "inspect":
            self.info_text_box.set_relative_position(target_pos)
            new_text = str(target_pos) + "<br>"
            new_text += str(node_id) + "<br>"
            for _, node_attribute in enumerate(self.net_sim.son.graph.nodes[node_id]):
                new_text += node_attribute + ": " + str(
                    self.net_sim.son.graph.nodes[node_id][node_attribute]) + "<br>"

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
            self.objectives_infobox_toggle.select()

        if len(self.net_sim.son.graph.nodes) == 0 or len(self.net_sim.son.graph.edges) == 0:
            self.info_text_box_objectives.set_text("There is no valid network yet.")
            self.info_text_box_objectives.show()
            return

        objective_text = ""

        avg_sinr: float = self.net_sim.son.get_average_sinr()
        avg_rssi: float = self.net_sim.son.get_average_rssi()
        avg_dl_rate: float = self.net_sim.son.get_average_dl_datarate()
        avg_load: float = self.net_sim.son.get_average_network_load()
        total_energy_efficiency: float = self.net_sim.son.get_total_energy_efficiency()
        avg_energy_efficiency = self.net_sim.son.get_avg_energy_efficiency()
        network_energy_consumption: float = self.net_sim.son.get_total_energy_consumption()
        objective_text = "user devices<br>avg_rssi: " + str(avg_rssi) + "<br>avg_sinr: " + str(avg_sinr) + "<br>avg_dl_rate: " + str(avg_dl_rate) + "<br><br>base stations<br>avg_load %: " + str(
            avg_load) + "<br>total_energy_consumption: " + str(network_energy_consumption) + "<br>total_energy_efficiency: " + str(total_energy_efficiency) + "<br>avg_energy_efficiency: " + str(avg_energy_efficiency)

        self.info_text_box_objectives.set_text(objective_text)
        self.info_text_box_objectives.show()

    def on_right_click(self, target_pos: tuple[int, int]):
        target_x = round(target_pos[0] / self.unit_size_x, 2)
        target_y = round(target_pos[1] / self.unit_size_y, 2)

        if self.right_mouse_action != "remove" and self.right_mouse_action != "cell":
            self.net_sim.son.add_bs_node(
                (target_x, target_y),
                update_network=False, bs_type=self.right_mouse_action)
        if self.right_mouse_action == "cell":
            self.net_sim.son.add_user_node(
                (target_x, target_y),
                update_network=False)
        elif self.right_mouse_action == "remove":
            node_id = self.node_clicked(target_pos)
            if node_id:
                self.net_sim.son.remove_node(node_id=node_id, update_network=False)
        self.update_network_info_lables()

    def update_network_info_lables(self):
        beams = 0
        bs_nodes_list = list(filter(lambda x: x[1]["type"] != "cell", self.net_sim.son.graph.nodes.data()))
        for _, bs_node in enumerate(bs_nodes_list):
            beams += self.net_sim.son.network_node_params[bs_node[1]["type"]]["antennas"]

        user_count = len(
            list(
                filter(
                    lambda x: x[1]["type"] == "cell", self.net_sim.son.graph.nodes.data())))
        self.capacity_label.set_text(f"network capacity: {beams}")
        self.user_count_label.set_text(f"user count: {user_count}")

    def onclick_show_edges_checkbox(self):
        self.show_edges_checkbox_active = not self.show_edges_checkbox_active
        self.show_edges_toggle.select() if self.show_edges_checkbox_active else self.show_edges_toggle.unselect()


    def ontoggle_greedy_to_moving(self):
        self.net_sim.config_params["greedy_to_moving"] = not self.net_sim.config_params["greedy_to_moving"]
        self.greedy_to_moving_toggle.select() if self.net_sim.config_params["greedy_to_moving"] else self.greedy_to_moving_toggle.unselect()

    def onshow_moving_selection_checkbox(self):
        self.show_moving_users = not self.show_moving_users

        self.show_moving_selection_toggle.select() if self.show_moving_users else self.show_moving_selection_toggle.unselect()

    def onpress_fps_toggle(self):
        self.one_fps_sim_mode = not self.one_fps_sim_mode
        self.net_sim.fps = 1 if self.one_fps_sim_mode else self.gui_fps
        self.sim_fps_toggle.select() if self.one_fps_sim_mode else self.sim_fps_toggle.unselect()

    def apply_current_network_params_to_ui_and_graph(self):
        # apply params on network
        self.net_sim.apply_current_network_params_to_graph()
        # load moving selections
        self.net_sim.load_current_moving_selection()

        self.create_algo_param_ui_elements()

        self.create_live_param_ui_elements()
   
    def onpress_generate_user_nodes(self):
        percentage = float(self.input_generate_user_nodes.get_text())
        self.net_sim.generate_user_nodes(percentage)
        self.update_network_info_lables()

    def on_entry_changed(self, event: pygame.event.Event):

        pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)$'
        if not re.match(pattern, event.text):
            event.text = 0
        if event.ui_element == self.input_transmission_power:
            tx_in_watts = self.net_sim.dbm_to_watts(float(event.text))
            self.net_sim.config_params[self.right_mouse_action]["tx_power"] = tx_in_watts
        if event.ui_element == self.input_antennas:
            self.net_sim.config_params[self.right_mouse_action]["antennas"] = float(event.text)
        if event.ui_element == self.input_channel_bandwidth:
            self.net_sim.config_params[self.right_mouse_action]["bandwidth"] = float(event.text)
        if event.ui_element == self.input_frequency:
            self.net_sim.config_params[self.right_mouse_action]["frequency"] = float(event.text)
            if float(event.text) > 0:
                wave_length = self.net_sim.ghz_to_wave_length_m(float(event.text))
            else:
                wave_length = 0
            self.net_sim.config_params[self.right_mouse_action]["wave_length"] = wave_length
        if event.ui_element == self.input_standby_power:
            self.net_sim.config_params[self.right_mouse_action]["standby_power"] = float(event.text)
        if event.ui_element == self.input_static_power:
            self.net_sim.config_params[self.right_mouse_action]["static_power"] = float(event.text)
        if event.ui_element == self.input_n_generations:
            self.net_sim.config_params["n_generations"] = float(event.text)
        if event.ui_element == self.input_n_offsprings:
            self.net_sim.config_params["n_offsprings"] = float(event.text)
        if event.ui_element == self.input_pop_size:
            self.net_sim.config_params["pop_size"] = float(event.text)
        if event.ui_element == self.input_resetting_rate:
            self.net_sim.config_params["reset_rate_in_ngen"] = float(event.text)
        if event.ui_element == self.input_running_time:
            self.net_sim.config_params["running_time_in_s"] = float(event.text)
        if event.ui_element == self.input_velocity:
            self.net_sim.config_params["moving_speed"] = float(event.text)
        if event.ui_element == self.input_create_movement_selection_percentage:
            self.net_sim.config_params["moving_selection_percent"] = float(event.text)
        if event.ui_element == self.input_iterations:
            self.net_sim.config_params["iterations"] = int(event.text)

    def on_dropdown_pick_network_changed(self, event: pygame.event.Event):
        # reset config params to default
        self.net_sim.config_params = default_algorithm_param_config
        self.net_sim.current_config_name = ""
       
        if event.text != "from file":
            self.net_sim.current_network_name = event.text
            self.ui_container_live_config.enable()
        else:
            self.net_sim.current_network_name = ""
           
            self.dropdown_pick_algo_config.disable()
            self.dropdown_pick_result.disable()
            self.ui_container_live_config.disable()
           
        self.reload_current_network_graph()
        self.create_algo_param_ui_elements()
        self.create_live_param_ui_elements()

    def reload_current_network_graph(self):
        # reload graph layout
        self.net_sim.load_current_adjacencies()
        # reapply current config for graph
        self.net_sim.apply_current_network_params_to_graph()

        self.update_network_info_lables()

    def on_dropdown_input_algorithm_config_changed(self, event: pygame.event.Event, param_key: str):
        self.net_sim.config_params[param_key] = event.text

    def on_selectionlist_input_objectives_changed(self, event: pygame.event.Event):
        self.net_sim.config_params["objectives"] = self.input_objectives.get_multi_selection()

        if (len(self.net_sim.config_params["objectives"]) == 0):
            self.evo_start_button.disable()
        else:
            self.evo_start_button.enable()

    def on_dropdown_pick_algo_config_changed(self, event: pygame.event.Event):
        if event.text == "from file":
            self.dropdown_pick_result.disable()
            
            self.net_sim.current_config_name = ""
            self.net_sim.config_params = default_algorithm_param_config
        else:
            self.net_sim.current_config_name = event.text
            self.net_sim.load_param_config_from_file(self.net_sim.get_current_config_directory() + event.text + ".json")
            
            self.apply_current_network_params_to_ui_and_graph()


    def on_dropdown_moving_selection_changed(self, event: pygame.event.Event):
        if event.text == "from file":
            self.net_sim.config_params["moving_selection_name"] = ""
        else:
            self.net_sim.config_params["moving_selection_name"] = event.text
        
        self.net_sim.load_current_moving_selection()

    def on_dropdown_pick_result_changed(self, event: pygame.event.Event):
        # Opening json file
        self.net_sim.load_result_ind_from_file(event.text)
        self.update_network_info_lables()

    def on_dropdown_mode_changed(self, event: pygame.event.Event):
        self.net_sim.running_mode = event.text

    def create_live_param_ui_elements(self):
        
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
            initial_text=str(self.net_sim.config_params["reset_rate_in_ngen"]))
        self.input_resetting_rate.set_allowed_characters(text_input_integer_number_type_characters)

        self.input_running_time_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 30), (-1, 30)), "running time in s", self.manager, self.ui_container_live_config)
        self.input_running_time = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 30),
                        (100, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="running time",
            initial_text=str(self.net_sim.config_params["running_time_in_s"]))
        self.input_running_time.set_allowed_characters(text_input_integer_number_type_characters)

        self.input_velocity_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 60), (-1, 30)), "velocity", self.manager, self.ui_container_live_config)
        self.input_velocity = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 60),
                        (100, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="velocity",
            initial_text=str(self.net_sim.config_params["moving_speed"]))
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
            initial_text=str(self.net_sim.config_params["moving_selection_percent"]))
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
            initial_text=str(self.net_sim.config_params["moving_selection_name"]))

        self.select_moving_selection_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 150), (-1, 30)), "moving selections", self.manager, self.ui_container_live_config)

        self.dropdown_moving_selection = pygame_gui.elements.UIDropDownMenu(
            options_list=self.net_sim.get_moving_selections_for_current_network(),
            starting_option=str(self.net_sim.config_params["moving_selection_name"])
            if str(self.net_sim.config_params["moving_selection_name"]) != "" else "from file",
            relative_rect=pygame.Rect((220, 150),
                                      (200, 30)),
            manager=self.manager, container=self.ui_container_live_config,)
        self.input_iterations_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 180), (-1, 30)), "iterations", self.manager, self.ui_container_live_config)
        self.input_iterations = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 180),
                        (100, 30)),
            self.manager, self.ui_container_live_config, placeholder_text="iterations",
            initial_text=str(self.net_sim.config_params["iterations"]))
        self.input_iterations.set_allowed_characters(text_input_integer_number_type_characters)


    def create_algo_param_ui_elements(self):
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
            options_list=self.net_sim.get_network_folder_names(),
            starting_option=self.net_sim.get_network_folder_names()[0]
            if self.net_sim.current_network_name == "" else self.net_sim.current_network_name, relative_rect=pygame.Rect(
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

        self.show_edges_toggle = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (320, 20),
            (50, 30)),
            object_id="toggle",
            text="show edges", manager=self.manager,
            container=self.ui_container)
        self.show_edges_checkbox_active = True
        self.show_edges_toggle.select()

        self.objectives_infobox_toggle = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (370, 20),
            (50, 30)),
            object_id="toggle",
            text="obj", manager=self.manager,
            container=self.ui_container)
        self.objectives_infobox_toggle.select() if self.objectives_checkbox_active else self.objectives_infobox_toggle.unselect()

        self.apply_ui_params_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (320, 230),
            (-1, 30)),
            object_id="#apply_button",
            text="apply params", manager=self.manager,
            container=self.ui_container)
       
        self.generate_user_nodes_butotn = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (320, 260),
            (-1, 30)),
            text="generate user nodes %", manager=self.manager,
            container=self.ui_container)
        self.input_generate_user_nodes = pygame_gui.elements.UITextEntryLine( pygame.Rect((520, 260), (40, 30)), self.manager, self.ui_container,initial_text= "100")
        self.input_generate_user_nodes.set_allowed_characters(text_input_float_number_type_characters)
        # network params

        self.input_antennas_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 50), (-1, 30)), "maximum amount of beams", self.manager, self.ui_container)

        self.input_antennas = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 50),
                        (100, 30)),
            self.manager, self.ui_container, placeholder_text="max beams",
            initial_text=str(self.net_sim.config_params[initial_nodeType]["antennas"]))
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
            initial_text=str(self.net_sim.config_params[initial_nodeType]["channel_bandwidth"]))
        self.input_channel_bandwidth.set_allowed_characters(text_input_float_number_type_characters)

        self.input_transmission_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 140), (-1, 30)), "transmission power in dbm", self.manager, self.ui_container)
       
        # transform tx_power in watts from file to dbm for ui
        tx_power_dbm = self.net_sim.watts_to_dbm(self.net_sim.config_params[initial_nodeType]["tx_power"])
        self.input_transmission_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 140), (100, 30)), self.manager, self.ui_container, placeholder_text="tx power",
            initial_text=str(tx_power_dbm))
        self.input_transmission_power.set_allowed_characters(
            text_input_float_number_type_characters)

        self.input_static_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 170), (-1, 30)), "static power in Watt", self.manager, self.ui_container)
        self.input_static_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 170), (100, 30)), self.manager, self.ui_container, placeholder_text="static power",
            initial_text=str(self.net_sim.config_params[initial_nodeType]["static_power"]))
        self.input_static_power.set_allowed_characters(text_input_float_number_type_characters)

        self.input_standby_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 200), (-1, 30)), "standby power in Watt", self.manager, self.ui_container)
        self.input_standby_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 200), (100, 30)), self.manager, self.ui_container, placeholder_text="standby power",
            initial_text=str(self.net_sim.config_params[initial_nodeType]["standby_power"]))
        self.input_standby_power.set_allowed_characters(text_input_float_number_type_characters)

        self.input_frequency_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 230), (-1, 30)), "frequency in GHZ", self.manager, self.ui_container)
        self.input_frequency = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 230), (100, 30)), self.manager, self.ui_container, placeholder_text="frequency",
            initial_text=str(self.net_sim.config_params[initial_nodeType]["frequency"]))
        self.input_frequency.set_allowed_characters(text_input_float_number_type_characters)

        self.input_pop_size_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 260), (-1, 30)), "population size", self.manager, self.ui_container)
        self.input_pop_size = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 260), (100, 30)), self.manager, self.ui_container, placeholder_text="pop_size",
            initial_text=str(self.net_sim.config_params["pop_size"]))
        self.input_pop_size.set_allowed_characters(text_input_float_number_type_characters)

        self.input_n_offsprings_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 290), (-1, 30)), "number of offsprings", self.manager, self.ui_container)
        self.input_n_offsprings = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 290), (100, 30)), self.manager, self.ui_container, placeholder_text="n_offsprings",
            initial_text=str(self.net_sim.config_params["n_offsprings"]))
        self.input_n_offsprings.set_allowed_characters(text_input_float_number_type_characters)

        self.input_n_generations_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 320), (-1, 30)), "number of generations", self.manager, self.ui_container)
        self.input_n_generations = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 320), (100, 30)), self.manager, self.ui_container, placeholder_text="n_generations",
            initial_text=str(self.net_sim.config_params["n_generations"]))
        self.input_n_generations.set_allowed_characters(text_input_float_number_type_characters)

        self.input_sampling_dropdown_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 350), (-1, 30)), "sampling operation", self.manager, self.ui_container)
        self.input_sampling_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in SamplingEnum],
            starting_option=str(self.net_sim.config_params["sampling"]),
            relative_rect=pygame.Rect(
                220, 350, 250, 30),
            manager=self.manager,
            container=self.ui_container

        )

        self.input_crossover_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 380), (-1, 30)), "crossover operation", self.manager, self.ui_container)
        self.input_crossover = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in CrossoverEnum],
            starting_option=str(self.net_sim.config_params["crossover"]),
            relative_rect=pygame.Rect(
                220, 380, 250, 30),
            manager=self.manager,
            container=self.ui_container
        )

        self.input_mutation_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 410), (-1, 30)), "mutation operation", self.manager, self.ui_container)
        self.input_mutation = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in MutationEnum],
            starting_option=str(self.net_sim.config_params["mutation"]),
            relative_rect=pygame.Rect(
                220, 410, 250, 30),
            manager=self.manager,
            container=self.ui_container
        )

        self.input_algorithm_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 440), (-1, 30)), "algorithm", self.manager, self.ui_container)
        self.input_algorithm = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in AlgorithmEnum],
            starting_option=str(self.net_sim.config_params["algorithm"]),
            relative_rect=pygame.Rect(
                220, 440, 250, 30),
            manager=self.manager,
            container=self.ui_container
        )

        self.input_eliminate_duplicates_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 470), (-1, 30)), "eliminate duplicates", self.manager, self.ui_container)
        self.input_eliminate_duplicates = pygame_gui.elements.UIDropDownMenu(
            options_list=["True", "False"],
            starting_option=str(self.net_sim.config_params["eliminate_duplicates"]),
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
            default_selection=self.net_sim.config_params["objectives"]
        )

        self.switch_algorithm_mode_toggle = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 530), (-1, 30)), text='greedy mode',
            manager=self.manager, container=self.ui_container, object_id="toggle")
        self.switch_algorithm_mode_toggle.select() if self.net_sim.config_params["use_greedy_assign"] else self.switch_algorithm_mode_toggle.unselect()

        self.evo_start_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 560), (-1, 30)), text='start',
            manager=self.manager, container=self.ui_container)

        self.input_algo_config_name = pygame_gui.elements.UITextEntryLine(
            container=self.ui_container, relative_rect=pygame.Rect((85, 560), (120, 30)),
            manager=self.manager, placeholder_text="config_name")

        self.evo_stop_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 590), (-1, 30)), text='stop',
            manager=self.manager, container=self.ui_container)
        self.evo_stop_button.disable()
        
        self.sim_fps_toggle = pygame_gui.elements.UIButton(
            container=self.ui_container, relative_rect=pygame.Rect((100, 590), (-1, 30)),
            manager=self.manager, text="1-fps-mode", object_id="toggle")
        self.sim_fps_toggle.select() if self.one_fps_sim_mode else self.sim_fps_toggle.unselect()

        self.dropdown_pick_running_mode = pygame_gui.elements.UIDropDownMenu(
            options_list=[item.value for item in RunningMode],
            starting_option=self.net_sim.running_mode,
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
        self.greedy_to_moving_toggle.select() if self.net_sim.config_params["greedy_to_moving"] else self.greedy_to_moving_toggle.unselect()


        self.dropdown_pick_algo_config = pygame_gui.elements.UIDropDownMenu(
            options_list=self.net_sim.get_config_names_for_current_network(),
            starting_option=self.net_sim.get_config_names_for_current_network()[0] if self.net_sim.current_config_name == "" else self.net_sim.current_config_name,
            relative_rect=pygame.Rect((20, 650), (280, 30)),
            manager=self.manager,
            container=self.ui_container,
        )
            
        self.dropdown_pick_result = pygame_gui.elements.UIDropDownMenu(
            options_list=self.net_sim.get_results_for_current_config(),
            starting_option=self.net_sim.get_results_for_current_config()[0],
            relative_rect=pygame.Rect((300, 650), (200, 30)),
            manager=self.manager,
            container=self.ui_container,
        )

    def on_dropdown_canvas_action_changed(self, event: pygame.event.Event):
        self.right_mouse_action = event.text
        if event.text == "remove" or event.text == "cell":
            self.apply_ui_params_button.disable()
            self.input_antennas.disable()
            self.input_channel_bandwidth.disable()
            self.input_frequency.disable()
            self.input_standby_power.disable()
            self.input_static_power.disable()
            self.input_transmission_power.disable()
            self.dropdown_menu_pick_network.disable()
        else:
            self.apply_ui_params_button.enable()
            self.dropdown_menu_pick_network.enable()
            self.input_antennas.enable()
            self.input_antennas.set_text(
                str(self.net_sim.config_params[self.right_mouse_action]["antennas"]))

            self.input_channel_bandwidth.enable()
            self.input_channel_bandwidth.set_text(
                str(self.net_sim.config_params[self.right_mouse_action]["channel_bandwidth"]))

            self.input_frequency.enable()
            self.input_frequency.set_text(
                str(self.net_sim.config_params[self.right_mouse_action]["frequency"]))

            self.input_standby_power.enable()
            self.input_standby_power.set_text(
                str(self.net_sim.config_params[self.right_mouse_action]["standby_power"]))

            self.input_static_power.enable()
            self.input_static_power.set_text(
                str(self.net_sim.config_params[self.right_mouse_action]["static_power"]))

            self.input_transmission_power.enable()
            tx_power_dbm = self.net_sim.watts_to_dbm(
                self.net_sim.config_params[self.right_mouse_action]["tx_power"])
            self.input_transmission_power.set_text(str(tx_power_dbm))

    def switch_algorithm_mode(self):
        self.net_sim.config_params["use_greedy_assign"] = not self.net_sim.config_params["use_greedy_assign"]
        self.switch_algorithm_mode_toggle.select() if self.net_sim.config_params["use_greedy_assign"] else self.switch_algorithm_mode_toggle.unselect()


    def onpress_save_moving_selection(self):
        name = self.input_moving_selection_name.get_text()
        response = self.net_sim.save_new_moving_selection(name)

        if response == ErrorEnum.MOVING_SELECTION_EMPTY.value:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 260, 200),
                blocking=True,
                manager=self.manager,
                window_title="Missing user selection",
                action_short_name="ok",
                action_long_desc="press create moving selection first.",
                visible=True)
            return
            
        if response == ErrorEnum.NAME_ALREADY_EXISTS.value:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 260, 200),
                blocking=True,
                manager=self.manager,
                window_title="File Name already exists",
                action_short_name="ok",
                action_long_desc="enter different file name.",
                visible=True)

        if response == ErrorEnum.NAME_MISSING.value:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 260, 200),
                blocking=True,
                manager=self.manager,
                window_title="Missing file name",
                action_short_name="ok",
                action_long_desc="enter file name.",
                visible=True)
            return

        # update moving_selections dropdown
        self.dropdown_moving_selection.kill()
        self.dropdown_moving_selection = pygame_gui.elements.UIDropDownMenu(
            options_list=self.net_sim.get_moving_selections_for_current_network(),
            starting_option=name,
            relative_rect=pygame.Rect((220, 150), (200, 30)),
            manager=self.manager,
            container=self.ui_container_live_config)
    
    def onpress_apply_params_from_ui(self):
        self.net_sim.apply_current_network_params_to_graph()
    
    def onpress_stop_evo(self):
        self.net_sim.force_stop_evo()

    def onpress_start_evo(self):

        response = self.net_sim.start_evo(self.input_algo_config_name.get_text())
        if response == ErrorEnum.NO_MOVING_SELECTION.value:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 260, 200),
                blocking=True,
                manager=self.manager,
                window_title="missing movement selection name",
                action_short_name="ok",
                action_long_desc="save current selection with name or select one from file",
                visible=True)
            return
        if response == ErrorEnum.NAME_ALREADY_EXISTS.value:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 260, 200),
                blocking=True,
                manager=self.manager,
                window_title="config name already exists",
                action_short_name="ok",
                action_long_desc="choose different name for config",
                visible=True)
            return
        if response == ErrorEnum.NAME_MISSING.value:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 260, 200),
                blocking=True,
                manager=self.manager,
                window_title="no config name",
                action_short_name="ok",
                action_long_desc="type name for this config first.",
                visible=True)
            return
        if response == ErrorEnum.NETWORK_MISSING.value:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 260, 200),
                blocking=True,
                manager=self.manager,
                window_title="Missing network file",
                action_short_name="ok",
                action_long_desc="press save first to create required target folder",
                visible=True)
            return

        self.disable_ui()
        self.evo_stop_button.enable()

    def on_optimization_finished_callback(self, message: str):
        # update algo param config dropdown
        self.gui_ticks = 0
        self.create_algo_param_ui_elements()
        self.enable_ui()


    def onpress_save_network(self):
        response = self.net_sim.save_new_network(self.input_network_name.get_text())

        if response == ErrorEnum.NAME_ALREADY_EXISTS.value:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 260, 200),
                blocking=True,
                manager=self.manager,
                window_title="file name already exists",
                action_short_name="ok",
                action_long_desc="choose different network name",
                visible=True,)
            return
        if response == ErrorEnum.NAME_MISSING.value:
            self.popup.kill()
            self.popup = CustomConfirmationDialog(
                rect=pygame.Rect(GAME_WIDTH + 20, 20, 260, 200),
                blocking=True,
                manager=self.manager,
                window_title="Missing file name",
                action_short_name="ok",
                action_long_desc="sepecify network name first",
                visible=True,
            )
            return
        
        self.ui_container_live_config.enable()


    def run(self):
        while True:
            # set time per tick
            dt = self.clock.tick(self.gui_fps)/1000
            if self.one_fps_sim_mode:
                if self.gui_ticks % self.gui_fps == 0:
                    self.net_sim.step_one_tick(dt*self.gui_fps)
            else:
                self.net_sim.step_one_tick(dt)

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
                    if event.ui_element == self.apply_ui_params_button:
                        self.onpress_apply_params_from_ui()
                    if event.ui_element == self.generate_user_nodes_butotn:
                        self.onpress_generate_user_nodes()
                    if event.ui_element == self.save_button:
                        self.onpress_save_network()
                    if event.ui_element == self.show_edges_toggle:
                        self.onclick_show_edges_checkbox()
                    if event.ui_element == self.objectives_infobox_toggle:
                        self.show_objectives_info_box()
                    if event.ui_element == self.sim_fps_toggle:
                        self.onpress_fps_toggle()
                    if event.ui_element == self.evo_start_button:
                        self.onpress_start_evo()
                    if event.ui_element == self.evo_stop_button:
                        self.onpress_stop_evo()
                    if event.ui_element == self.create_movement_selection_button:
                        self.onclick_create_moving_users()
                    if event.ui_element == self.save_moving_selection_button:
                        self.onpress_save_moving_selection()
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
            self.gui_ticks += 1


if __name__ == "__main__":
    son = Son()
    main = Editor(Network_Simulation_State(son,script_mode=False))
    main.run()
    # cProfile.run("main.run()", sort="tottime")
