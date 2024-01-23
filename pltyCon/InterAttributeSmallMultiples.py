"""
InterAttributeSmallMultiples.py

Creates a m×n-matrix of small multiples for comparing two m, n dimensions of attributes.
Within each cell of the matrix, further dimensions can be added by using appropriate chart types.


"""

import json
from plotly_templates import load_templates, export_with_bugfix
from significance import get_color_by_p_statement, get_label_by_p_statement

loaded_dpi = load_templates(
    # base_width_in_px=400,
    dpi=96
)


from plotly.subplots import make_subplots
import plotly.graph_objects as go


class InterAttributeSmallMultiples():

    def __init__(
            self,
            entry_array_rows,
            entry_array_cols,
            entry_arrays_cell,
            row_dim_label=None,
            col_dim_label=None,
            row_dim_unit=None,
            col_dim_unit=None,
            cell_dim_label=None,
            cell_dim_unit=None,
            ) -> None:
        
        self.row_entries = entry_array_rows
        self.col_entries = entry_array_cols

        self.row_dim_label = row_dim_label
        self.row_dim_unit = row_dim_unit

        self.col_dim_label = col_dim_label
        self.col_dim_unit = col_dim_unit

        self.cell_dim_label = cell_dim_label
        self.cell_dim_unit = cell_dim_unit

        # Print the configuration
        print(f'row_dim_label: {self.row_dim_label}')
        print(f'row_dim_unit: {self.row_dim_unit}')
        print(f'col_dim_label: {self.col_dim_label}')
        print(f'col_dim_unit: {self.col_dim_unit}')
        print(f'cell_dim_label: {self.cell_dim_label}')
        print(f'cell_dim_unit: {self.cell_dim_unit}')



    def assemble_title_eqn(self, dim_label, config_val, dim_unit):
        "Creates a MathJax-String showing the current configuration"

        return fr'${dim_label} = {config_val} {dim_unit}$'




    def set_labels(self):
        """
        Adds labels to the matrix at the bottom and left side.
        """

        # Add column labels at the bottom of the matrix
        for idx, config_val in enumerate(self.row_entries):
            self.fig_handle.update_xaxes(
                title_text=self.assemble_title_eqn(
                    dim_label=self.row_dim_label,
                    config_val=config_val,
                    dim_unit=self.row_dim_unit
                ),
                row=len(self.col_entries),
                col=idx+1,
            )

        # Add row labels at the left of the matrix
        for idx, config_val in enumerate(self.col_entries):
            self.fig_handle.update_yaxes(
                title_text=self.assemble_title_eqn(
                    dim_label=self.col_dim_label,
                    config_val=config_val,
                    dim_unit=self.col_dim_unit
                ),
                row= idx+1,
                col=1,
            )

        # TODO: Add lookup-matrix for the cell labels



    def add_boxplot(self, data, row, col, config_val, p_statement=None):
        """
        Injects the trace of a 1D-vector as a boxplot into the given cell of the matrix.
        The boxplot is 
        - labeled via the given name
        - labeled also with significance stars if requested
        - colored according to the significance level if requested
        """

        if p_statement is not None:
            color = get_color_by_p_statement(p_statement)
            p_marker = get_label_by_p_statement(p_statement)

        config_label = self.assemble_title_eqn(
            dim_label=self.cell_dim_label,
            config_val=config_val,
            dim_unit=self.cell_dim_unit
        )

        self.fig_handle.add_trace(
            go.Box(
                y=data,
                boxpoints='outliers',
                marker_color=color,
                line_color=color,
                name=fr'{config_label}<br>{p_marker}' if p_statement is not None else config_val,
            ),
            row=row,
            col=col,
        )


    def add_boxplot_comparison(self, row, col, contenders, baseline=None, mark_significance=None):

        if baseline is not None:
            # plot the baseline
            self.add_boxplot(
                data=baseline['data'],
                row=row,
                col=col,
                config_val=baseline['label'],
                p_statement='REF' if mark_significance else None
            )


        # inject the boxplot of each contender
        for contender in contenders:

            self.add_boxplot(
                data=contender['data'],
                row=row,
                col=col,
                config_val=contender['label'],
                p_statement=contender['p_value'] if mark_significance else None
            )


    def add_reference_line(self, row, col, value, label, color):
        """
        TODO: replace with hline
        """
            
        self.fig_handle.add_hline(
            y=value, 
            row=row, 
            col=col, 
            line_color=color, 
            # line_dash='dash', 
            # annotation_text=label
            )

    def init_figure(self,template):

        self.fig_handle = make_subplots(
            rows=len(self.col_entries),
            cols=len(self.row_entries),
        )

        # Set the template of the figure
        self.fig_handle.update_layout(template=template)

        self.set_labels()

    def show(self):
        self.fig_handle.show()


if __name__ == '__main__':


    row_dim_label = r'\frac{n_{u,\tiny{\text{DYN}}}}{\left|\tiny{\unicode{x1D4B0}} \enspace \right|}'
    row_dim_unit = r'\%'
    col_dim_label = r'\frac{c_{\tiny{\unicode{x1D4B0}}}}{c_{\mathcal{B}}}'
    col_dim_unit = r'\%'
    cell_dim_label = r'v_{\scriptsize{\text{UE}}}'
    cell_dim_unit = r'\frac{\text{m}}{\text{s}}'

    config_entries_row_dim=['30', '70', '100']
    config_entries_col_dim=['50', '100', '150']
    entry_arrays_cell=[]


    # TODO: Create exposed interface for adding data

    small_multiple_plot = InterAttributeSmallMultiples(
        entry_array_rows=config_entries_row_dim,
        row_dim_label=row_dim_label,
        entry_array_cols=config_entries_col_dim,
        col_dim_label=col_dim_label,
        row_dim_unit=row_dim_unit,
        col_dim_unit=col_dim_unit,
        cell_dim_unit=cell_dim_unit,
        cell_dim_label=cell_dim_label,
        entry_arrays_cell=[],
    )

    
    # dummy_data = {
    #     'baseline': {
    #         'data': [1, 2, 3],
    #         'label': 'REF',
    #     },
    #     'contenders': [
    #         {
    #             'config': {
    #                 'moving_portion': 0.3,
    #                 'usage_ratio': 0.5,
    #             },
    #             'data': [1, 2, 3],
    #             'label': '1.2',
    #             'p_value': 0.01
    #         },
    #         {
    #             'config': {
    #                 'moving_portion': 0.1,
    #                 'usage_ratio': 0.5,
    #             },
    #             'data': [3, 6, 8],
    #             'label': '14',
    #             'p_value': 0.05
    #         },
    #     ]
    # }

    actual_data = {
        "50":{
            "30":{
                "contenders": []
            },
            "70":{
                "contenders": []
            },
            "100": {
                "contenders": []
            }
        },
        "100":{
             "30": {
                "contenders": []
             },
            "70": {
                "contenders": []
            },
            "100": {
                "contenders": []
            }
        },
        "150": {
             "30": {
                "contenders": []
             },
            "70": {
                "contenders": []
            },
            "100": {
                "contenders": []
            }
        },
    }

    with open("./avg_hypervolume_ratios.json", "r") as jsonFile:
        jsonObject = json.load(jsonFile)

        for network_index, network in enumerate(jsonObject):
            for config_index, config_name in enumerate(jsonObject[network]):
                if "var" in config_name:
                    continue

                network_key = "50"
                moving_portion_key ="30"
                speed_key = ""
                
                if "het_C50" in config_name:
                    network_key = "50"
                elif "het_C100" in config_name:
                    network_key = "100"
                elif "het_C150" in config_name:
                    network_key = "150"

                if "1,2ms" in config_name:
                    speed_key = "1.2"
                elif "14ms" in config_name:
                    speed_key = "14"
                
                if "MP30" in config_name:
                    moving_portion_key = "30"
                elif "MP70" in config_name:
                    moving_portion_key= "70"
                elif "MP100" in config_name:
                    moving_portion_key = "100"
               
                actual_data[network_key][moving_portion_key]["contenders"].append(
                     {
                        "data": jsonObject[network][config_name],
                        "p_value": 0.01,
                        "label": speed_key
                    }
                )


    small_multiple_plot.init_figure(
        template='draft_overlay+oscilloscope+HC--3-2nd-CW'
    )

    for idx_horz, label_horz in enumerate(config_entries_row_dim):
        for idx_vert, label_vert in enumerate(config_entries_col_dim):

            small_multiple_plot.add_boxplot_comparison(
                contenders=actual_data[label_vert][label_horz]['contenders'],
                # baseline=dummy_data['baseline'],
                row=idx_vert+1,
                col=idx_horz+1,
                mark_significance=True
            )

            small_multiple_plot.add_reference_line(
                row=idx_vert+1,
                col=idx_horz+1,
                value=1,
                label='Reference',
                color='black'
            )




    small_multiple_plot.show()


    # export_with_bugfix(
    #     fig_handle=small_multiple_plot.fig_handle, 
    #     filename="InterAttributeSmallMultiple-PLACEHOLDER.pdf"
    # )