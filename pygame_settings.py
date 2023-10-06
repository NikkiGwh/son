WINDOW_HEIGHT = 900
WINDOW_WIDTH = 1500
GAME_HEIGHT = 900
GAME_WIDTH = 900


style = {
    "nodes": {
        "colors": {
            "fill": {
                "macro": "orange",
                "micro": "blue",
                "femto": "purple",
                "pico": "lime",
                "cell": "black",
                "moving": (178, 6, 255, 1),
                "inactive": "grey",
                "overload": "red"
            }
        },
        "sizes": {
            "radius": {
                "macro": 20,
                "micro": 15,
                "femto": 10,
                "pico": 5,
                "cell": 3,
            }
        }
    },
    "edges": {
        "colors": {
            "inactive": "grey",
            "active": "green"
        },
        "sizes": {
            "edge_width": {
                "active": 2,
                "inactive": 1
            }
        }
    }
}
