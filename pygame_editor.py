import math
import pygame
import sys
from pygame_settings import *
import networkx as nx
import json
from networkx.readwrite import json_graph
from son_main_script import BaseStationOrder, BinPackingType, CellOrderTwo, Son
import pygame_gui
import re


dropdown_menue_options_list = ["macro", "micro", "femto", "pico", "cell", "remove"]
text_input_number_type_characters = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "."]
default_network_params = {
    "macro": {
        "type": "macro",
        "tx_power": 20,
        "static_power": 4.0,
        "standby_power": 0,
        "antennas": 4,
        "frequency": 0,
        "wave_length": 80,
        "channel_bandwidth": 10
    },
    "micro": {
        "type": "micro",
        "tx_power": 1,
        "static_power": 1.5,
        "standby_power": 0,
        "antennas": 1,
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


class Main():
    def __init__(self, graph: Son) -> None:
        pygame.init()
        self.selected_node_id = None
        self.right_mouse_action = dropdown_menue_options_list[0]
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.background = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.background.fill(pygame.colordict.THECOLORS["white"])
        self.clock = pygame.time.Clock()
        self.son = graph
        max_x, max_y = self.get_max_x_y(son.graph)
        max_value = max(max_x, max_y)
        self.unit_size_x, self.unit_size_y = (
            math.trunc(GAME_WIDTH / max_value),
            math.trunc(GAME_HEIGHT / max_value))
        self.network_params_dic = default_network_params
        # GUI
        self.manager = pygame_gui.UIManager((WINDOW_WIDTH, WINDOW_HEIGHT), './theme.json')
        self.info_text_box = pygame_gui.elements.UITextBox(
            "hallo", pygame.Rect(0, 0, -1, -1), self.manager, wrap_to_height=True, visible=False)

        self.ui_container = pygame_gui.elements.UIPanel(
            pygame.Rect((GAME_WIDTH, 0), (WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT)),
            object_id="#ui_container")

        self.dropdown_menu = pygame_gui.elements.UIDropDownMenu(
            dropdown_menue_options_list,
            dropdown_menue_options_list[0],
            pygame.Rect((20, 20), (100, 50)), self.manager,
            container=self.ui_container)

        self.selection_list_menue = pygame_gui.elements.UISelectionList(pygame.Rect(
            120, 20, 100, 50),
            ["inspect", "drag"],
            manager=self.manager,
            starting_height=0,
            container=self.ui_container,
            default_selection="drag")

        # network params
        self.network_params_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 70), (-1, -1)), "network parameters", self.manager, self.ui_container)

        self.input_antennas_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 120), (-1, -1)), "maximum amount of beams", self.manager, self.ui_container)
        self.input_antennas = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 120),
                        (100, -1)),
            self.manager, self.ui_container, placeholder_text="max beams",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["antennas"]))
        self.input_antennas.set_allowed_characters(text_input_number_type_characters)

        self.input_wave_length_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 170), (-1, -1)), "wave length in m", self.manager, self.ui_container)
        self.input_wave_length = pygame_gui.elements.UITextEntryLine(
            pygame.Rect((220, 170),
                        (100, -1)),
            self.manager, self.ui_container, placeholder_text="wave length",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["wave_length"]))

        self.input_channel_bandwidth_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 220), (-1, -1)), "channel bandwidth in HZ", self.manager, self.ui_container)
        self.input_channel_bandwidth = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 220), (100, -1)), self.manager, self.ui_container, placeholder_text="bandwidth",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["channel_bandwidth"]))

        self.input_transmission_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 270), (-1, -1)), "transmission power in Watt", self.manager, self.ui_container)
        self.input_transmission_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 270), (100, -1)), self.manager, self.ui_container, placeholder_text="tx power",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["tx_power"]))

        self.input_static_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 320), (-1, -1)), "static power in Watt", self.manager, self.ui_container)
        self.input_static_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 320), (100, -1)), self.manager, self.ui_container, placeholder_text="static power",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["static_power"]))

        self.input_standby_power_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 370), (-1, -1)), "standby power in Watt", self.manager, self.ui_container)
        self.input_standby_power = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 370), (100, -1)), self.manager, self.ui_container, placeholder_text="standby power",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["standby_power"]))

        self.input_frequency_label = pygame_gui.elements.UILabel(pygame.Rect(
            (20, 420), (-1, -1)), "frequency in HZ", self.manager, self.ui_container)
        self.input_frequency = pygame_gui.elements.UITextEntryLine(pygame.Rect(
            (220, 420), (100, -1)), self.manager, self.ui_container, placeholder_text="frequency",
            initial_text=str(self.network_params_dic[self.right_mouse_action]["frequency"]))

        # buttons
        self.apply_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 470),
            (-1, 50)),
            object_id="#apply_button",
            text="apply params", manager=self.manager,
            container=self.ui_container)
        self.bin_packing_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 520), (-1, 50)), object_id="#bin_packing_button", text='bin packing',
            manager=self.manager, container=self.ui_container)

        self.save_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((-120, -70),
                                      (100, 50)),
            object_id="#save_button", text='save', manager=self.manager, container=self.
            ui_container, anchors={"right": "right", "bottom": "bottom"})

    def get_max_x_y(self, graph: nx.Graph) -> tuple[float, float]:
        max_x = 100
        max_y = 100
        for _, node in enumerate(graph.nodes.data()):
            if node[1]["pos_x"] > max_x:
                max_x = node[1]["pos_x"]
            if node[1]["pos_y"] > max_y:
                max_y = node[1]["pos_y"]

        return (max_x, max_y)

    def draw_whiteboard(self):
        self.display_surface.blit(self.background, (0, 0))
        self.draw_edges()
        self.draw_nodes()

    def draw_edges(self):
        for _, edge in enumerate(self.son.graph.edges.data()):
            activation = "active" if edge[2]["active"] == True else "inactive"
            pygame.draw.line(
                self.display_surface, pygame.colordict.THECOLORS
                [style["edges"]["colors"][activation]],
                (self.son.graph.nodes[edge[0]]["pos_x"] * self.unit_size_x, self.son.graph.nodes
                 [edge[0]]["pos_y"] * self.unit_size_y),
                (self.son.graph.nodes[edge[1]]["pos_x"] * self.unit_size_x, self.son.graph.nodes
                 [edge[1]]["pos_y"] * self.unit_size_y),
                style["edges"]["sizes"]["edge_width"])

    def draw_nodes(self):
        for _, node in enumerate(self.son.graph.nodes.data()):
            node_type = node[1]["type"]
            activation = False if node[1]["type"] != "cell" and node[1]["active"] == False else True
            node_color_key = "inactive" if not activation else node_type
            pygame.draw.circle(
                self.display_surface, pygame.colordict.THECOLORS
                [style["nodes"]["colors"]["fill"][node_color_key]],
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
        self.son.move_node(node_id, (target_x, target_y), update_network=False)

    def on_left_click(self, target_pos: tuple[int, int]):
        id = self.selected_node_id
        if id is not None and self.selection_list_menue.get_single_selection() == "inspect":
            self.info_text_box.set_relative_position(target_pos)
            new_text = str(target_pos) + "<br>"
            for _, node_attribute in enumerate(self.son.graph.nodes[id]):
                new_text += node_attribute + ": " + str(
                    self.son.graph.nodes[id][node_attribute]) + "<br>"

            self.info_text_box.set_text(new_text)
            self.info_text_box.show()
        else:
            self.info_text_box.hide()

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

    def apply_params(self):
        self.son.apply_network_node_attributes(self.network_params_dic)

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
        else:
            self.apply_button.enable()

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
        self.son.find_activation_profile_bin_packing(
            CellOrderTwo.LOWEST_DEGREE_FIRST, bs_order=[BaseStationOrder.MACRO_FIRST],
            bin_packing=BinPackingType.BEST_FIT)

    def save_current_network(self, file_name: str):
        self.son.save_json_adjacency_graph_to_file(file_name)

    def run(self):
        while True:
            # set time per tick
            dt = self.clock.tick(60)/1000

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
                    if self.selection_list_menue.get_single_selection() == "drag" and self.background.get_rect().collidepoint(event.pos):
                        if event.buttons[0] and self.selected_node_id:
                            self.node_drag(self.selected_node_id, event.pos)

                if event.type == pygame.MOUSEBUTTONUP:
                    if self.background.get_rect().collidepoint(event.pos):
                        self.selected_node_id = None

                # if event.type == pygame.KEYDOWN:
                    # print(event.key)
                    # print(event.unicode)
                    # print(pygame.K_c)
                # print("----")

                # GIU events
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.bin_packing_button:
                        self.start_bin_packing()
                    if event.ui_element == self.apply_button:
                        self.apply_params()
                    if event.ui_element == self.save_button:
                        self.save_current_network("neu.json")
                if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    if event.ui_element == self.dropdown_menu:
                        self.on_dropdown_changed(event)
                if event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION:
                    if event.ui_element == self.selection_list_menue:
                        if event.text == "drag":
                            self.info_text_box.hide()
                if event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
                    self.on_entry_changed(event)
                self.manager.process_events(event)

            self.draw_whiteboard()
            self.manager.update(dt)
            self.manager.draw_ui(self.display_surface)

            pygame.display.update()


if __name__ == "__main__":
    son = Son()
    main = Main(son)
    main.run()
