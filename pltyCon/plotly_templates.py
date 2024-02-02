import plotly.graph_objects as go
import plotly.io as pio

from pltyCon.figure_sizes import dimensions_by_type, column_width

from copy import deepcopy
import time

def load_templates(dpi=None, base_width_in_px = None):
    """
    Loads the templates for standard figure sizes, a draft overlay and layout elements.
    "Oscilloscope" is the default template, styled like a scientific plot from an oscilloscope.

    
    """

    #region CALCULATION_BASE

    # calculate custom dpi in case that base width is given
    if base_width_in_px is not None:
        if dpi is not None:
            raise ValueError("You can only specify dpi or base_width_in_px, not both.")
        else:
            _dpi = base_width_in_px / (column_width / 25.4)

    # set dpi none if not specified
    _dpi = None if dpi is None else dpi
    
    #endregion CALCULATION_BASE

    #region FIGURE_SIZES

    for dimensions_name, dimensions in dimensions_by_type.items():
        pio.templates[dimensions_name] = go.layout.Template(
            layout = {
                "height": convert_mm_to_px(dimensions["height"], dpi = _dpi), 
                "width": convert_mm_to_px(dimensions["width"], dpi=_dpi)
                })

    #endregion FIGURE_SIZES

    #region DRAFT_OVERLAY

    pio.templates["draft_overlay"] = go.layout.Template(
        layout_annotations=[
            dict(
                name="draft watermark",
                text="DRAFT",
                textangle=-30,
                opacity=0.1,
                font=dict(color="black", size=100),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        ],
    )

    #endregion DRAFT_OVERLAY


    #region LAYOUT_ELEMENTS

    standard_font_format = {
        'family': 'Courier New, monospace',
        'size': 8,
        'color': 'black'
    }

    standard_axis_format = {
        "showgrid": True,
        "zeroline": False,
        "showline": True,
        "ticks": "",
        "showticklabels": True,
        "mirror": True,
        "linecolor": "darkgray",
        "linewidth": 2,
        "gridcolor": "lightgray",
        "gridwidth": 1,
        "tickfont": standard_font_format,
        "titlefont": standard_font_format,
    }

    standard_legend_format = {
        "font": standard_font_format,
        # "yanchor": "bottom",
        # "xanchor":"left",
        # "orientation":"h",
        # "y":-0.2,
        # "x":0,
        "grouptitlefont": standard_font_format
    }
    standard_title_format = {
        "font": {
            'family': 'Courier New, monospace',
            'size': 10,
            'color': 'black'
            }
    }

    standard_layout_format = {
        "xaxis": standard_axis_format,
        "yaxis": standard_axis_format,
        "paper_bgcolor": "white",
        'plot_bgcolor': '#FFFFFF',
        'margin': {'l': 50, 'r': 50, 't': 20, 'b': 40},
        'showlegend': True,
        'legend': standard_legend_format,
        'title': standard_title_format
    }

    #endregion LAYOUT_ELEMENTS

    #region OSCILLOSCOPE

    pio.templates["oscilloscope"] = go.layout.Template(
        layout = standard_layout_format
        )

    #endregion OSCILLOSCOPE

    return _dpi



def convert_mm_to_px(mm, dpi = None):

    default_dpi = {
        "windows": 96,
        "mac": 72,
        "print": 300
    }

    milliliters_per_inch = 25.4

    if dpi is None:
        dpi = default_dpi["print"]

    size_in_inch = mm / milliliters_per_inch

    return size_in_inch * dpi



def export_with_bugfix(fig_handle, filename):

    # TODO: Add scaling for pixel-based imports (e.g. png), use "scale" argument and the loaded_dpi/actual_dpi ratio

    fig_handle.write_image(filename)

    # Sleep for 5 seconds to make sure that the file is written



    time.sleep(5)

    # Run twice to hide the watermark
    # https://github.com/plotly/plotly.py/issues/3469#issuecomment-993486363

    fig_handle.write_image(filename)

