"""
This module creates an A4-sized figure with all possible figure sizes.
"""


import plotly.graph_objects as go
from figure_sizes import dimensions_by_type

import plotly_templates


page_size = {
    "height": plotly_templates.convert_mm_to_px(29.7, dpi=96),
    "width": plotly_templates.convert_mm_to_px(21.0, dpi=96)
}

page_margins = {
    "l": 10,
    "r": 10,
    "t": 10,
    "b": 10,
}

miniplot_separator = 10


fig_handle = go.Figure(
    layout = go.Layout(
        height = page_size["height"],
        width = page_size["width"],
        margin = page_margins,
        showlegend = False,
        template = "plotly_white",
        )
    )

# Add h and v lines at the margins.

# fig_handle.add_shape(
#     type="line",
#     x0=page_margins["l"],
#     y0=page_margins["b"],
#     x1=page_size["width"] - page_margins["r"],
#     y1=page_margins["b"],
#     line=dict(
#         color="black",
#         width=1,
#         ),
#     )

# fig_handle.add_shape(
#     type="line",
#     x0=page_margins["l"],
#     y0=page_margins["b"],
#     x1=page_margins["l"],
#     y1=page_size["height"] - page_margins["t"],
#     line=dict(
#         color="black",
#         width=1,
#         ),
#     )

# fig_handle.add_shape(
#     type="line",
#     x0=page_margins["l"],
#     y0=page_size["height"] - page_margins["t"],
#     x1=page_size["width"] - page_margins["r"],
#     y1=page_size["height"] - page_margins["t"],
#     line=dict(
#         color="black",
#         width=1,
#         ),
#     )

# fig_handle.add_shape(
#     type="line",
#     x0=page_size["width"] - page_margins["r"],
#     y0=page_margins["b"],
#     x1=page_size["width"] - page_margins["r"],
#     y1=page_size["height"] - page_margins["t"],
#     line=dict(
#         color="black",
#         width=1,
#         ),
#     )

# fig_handle.show()

fig_handle.show()