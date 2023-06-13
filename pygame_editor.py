import math
import pygame
import sys
from pygame_settings import *
import networkx as nx
import json
from networkx.readwrite import json_graph
from son_main_script import Son
import pygame_gui


dropdown_menue_options_list = ["macro", "micro", "femto", "pico", "cell", "remove"]
default_network_params = {
    "macro": {
        "type": "macro",
                "tx_power": 4,
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
        "type": "macro",
                "tx_power": 1,
                "static_power": 4.0,
                "standby_power": 0,
                "antennas": 4,
                "frequency": 0,
                "wave_length": 60,
                "channel_bandwidth": 10
    },
    "pico": {
        "type": "macro",
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
        self.background = pygame.Surface((GAME_HEIGHT, GAME_WIDTH))
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
        self.ui_container = pygame_gui.elements.UIPanel(
            pygame.Rect((GAME_WIDTH, 0), (WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT)),
            object_id="#ui_container")
        self.hello_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(
            (20, 10),
            (100, 50)),
            object_id="#hello_button",
            text='Say Hello',
            manager=self.manager,
            container=self.ui_container)
        self.dropdown_menu = pygame_gui.elements.UIDropDownMenu(
            dropdown_menue_options_list,
            dropdown_menue_options_list[0],
            pygame.Rect((20, 60), (100, 50)), self.manager,
            container=self.ui_container)

    def get_max_x_y(self, graph: nx.Graph) -> tuple[float, float]:
        max_x = 100
        max_y = 100
        for _, node in enumerate(graph.nodes.data()):
            if node[1]["pos_x"] > max_x:
                max_x = node[1]["pos_x"]
            if node[1]["pos_y"] > max_y:
                max_y = node[1]["pos_y"]

        return (max_x, max_y)

    def draw(self):
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

    def right_click_action(self, target_pos: tuple[int, int]):
        target_x = round(target_pos[0] / self.unit_size_x, 2)
        target_y = round(target_pos[1] / self.unit_size_y, 2)

        if self.right_mouse_action != "remove" and self.right_mouse_action != "cell":
            self.son.add_bs_node(
                (target_x, target_y),
                update_network=False,
                type=self.right_mouse_action,
                tx_power=self.network_params_dic
                [self.right_mouse_action]["tx_power"],
                active=True, antennas=self.network_params_dic[self.right_mouse_action]
                ["antennas"],
                channel_bandwidth=self.network_params_dic[self.right_mouse_action]
                ["channel_bandwidth"],
                frequency=self.network_params_dic[self.right_mouse_action]["frequency"],
                wave_length=self.network_params_dic[self.right_mouse_action]["wave_length"],
                static_power=self.network_params_dic[self.right_mouse_action]["static_power"])
        if self.right_mouse_action == "cell":
            self.son.add_user_node(
                (target_x, target_y),
                noise=self.network_params_dic[self.right_mouse_action]["noise"],
                shadow_component=self.network_params_dic[self.right_mouse_action]
                ["shadow_component"],
                update_network=False)
        elif self.right_mouse_action == "remove":
            node_id = self.node_clicked(target_pos)
            if node_id:
                self.son.remove_node(node_id=node_id, update_network=True)

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
                        if event.button == 3:
                            self.right_click_action(event.pos)

                if event.type == pygame.MOUSEMOTION:
                    if self.background.get_rect().collidepoint(event.pos):
                        if event.buttons[0] and self.selected_node_id:
                            self.node_drag(self.selected_node_id, event.pos)

                if event.type == pygame.MOUSEBUTTONUP:
                    if self.background.get_rect().collidepoint(event.pos):
                        self.selected_node_id = None

                if event.type == pygame.KEYDOWN:
                    print("key down")
                    # print(event.key)
                    # print(event.unicode)
                    # print(pygame.K_c)
                # print("----")

                # GIU events
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.hello_button:
                        print('Hello World!')
                if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    if event.ui_element == self.dropdown_menu:
                        self.right_mouse_action = event.text
                self.manager.process_events(event)

            self.draw()
            self.manager.update(dt)
            self.manager.draw_ui(self.display_surface)

            pygame.display.update()


if __name__ == "__main__":
    son = Son(file_name="test.json")
    main = Main(son)
    main.run()
