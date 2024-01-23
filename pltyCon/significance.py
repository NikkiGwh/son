

color_lut = {
    'baseline': {'name': 'darkgray', 'hex': '#A9A9A9', 'label': 'REF'},
    'p_value_higher_005': {'name': 'lightgray', 'hex': '#D3D3D3', 'label': ''},
    'p_value_lower_005': {'name': 'darkviolet', 'hex': '#9400D3', 'label': '*'},
    'p_value_lower_001': {'name': 'darkmagenta', 'hex': '#8B008B', 'label': '**'},
    'p_value_lower_0001': {'name': 'indigo', 'hex': '#4B0082', 'label': '***'},
}


def get_color_by_p_statement(p_value):
    if p_value == 'REF':
        return color_lut['baseline']['hex']
    elif p_value < 0.001:
        return color_lut['p_value_lower_0001']['hex']
    elif p_value < 0.005:
        return color_lut['p_value_lower_001']['hex']
    elif p_value < 0.05:
        return color_lut['p_value_lower_005']['hex']
    else:
        return color_lut['p_value_higher_005']['hex']


def get_label_by_p_statement(p_value):
    if p_value == 'REF':
        return color_lut['baseline']['label']
    elif p_value < 0.001:
        return color_lut['p_value_lower_0001']['label']
    elif p_value < 0.005:
        return color_lut['p_value_lower_001']['label']
    elif p_value < 0.05:
        return color_lut['p_value_lower_005']['label']
    else:
        return color_lut['p_value_higher_005']['label']



# import plotly.graph_objects as go

# # Define the color scale
# color_scale = [
#     color_lut['baseline']['hex'],
#     color_lut['p_value_higher_005']['hex'],
#     color_lut['p_value_lower_005']['hex'],
#     color_lut['p_value_lower_001']['hex'],
#     color_lut['p_value_lower_0001']['hex']
# ]

# # Define the labels
# labels = [
#     color_lut['baseline']['label'],
#     color_lut['p_value_higher_005']['label'],
#     color_lut['p_value_lower_005']['label'],
#     color_lut['p_value_lower_001']['label'],
#     color_lut['p_value_lower_0001']['label']
# ]

# # Create the plotly figure
# fig = go.Figure()

# # Add rectangles with colors and labels
# for i, color in enumerate(color_scale):
#     fig.add_shape(
#         type="rect",
#         x0=i,
#         y0=0,
#         x1=i+1,
#         y1=1,
#         fillcolor=color,
#         line=dict(color='black'),
#         layer="below"
#     )
#     fig.add_annotation(
#         x=i+0.5,
#         y=0.5,
#         text=labels[i],
#         showarrow=False,
#         font=dict(color='white')
#     )

# # Set the layout
# fig.update_layout(
#     plot_bgcolor='white',
#     xaxis=dict(
#         showgrid=False,
#         zeroline=False,
#         showticklabels=False
#     ),
#     yaxis=dict(
#         showgrid=False,
#         zeroline=False,
#         showticklabels=False
#     ),
#     width=600,  # Increase the width value
#     height=400  # Increase the height value
# )

# # Show the plotly figure
# fig.show()
