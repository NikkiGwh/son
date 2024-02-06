
import plotly.graph_objects as go
from pymoo.indicators.hv import Hypervolume
from pymoo.decomposition.asf import ASF
import json
import os
import numpy as np
from plotly.subplots import make_subplots
import glob
from PyPDF2 import PdfWriter, PdfReader 

from pltyCon.plotly_templates import load_templates
template_one_diagram = "oscilloscope+TH--1-3-TH"
template_subplot_diagrams = "oscilloscope+TH--3-4-TH"
template_one_diagram_paretofront= template_one_diagram
opacity_runs = 0.2
greedy_color= "black"
evo_color= "blue"
runs_color = "#2cbcc4"
datastore_directory_name = "datastore_thesis_4"

loaded_dpi = load_templates(
    # base_width_in_px=400,
    dpi=96
)
lineWidth = 1

subplots4x3 = make_subplots(
            rows=4,
            cols=3
            )

subplots4x3.update_layout(
     template=template_subplot_diagrams,
     showlegend=False
)
subplots4x3.update_annotations(font= {
        'family': 'Courier New, monospace',
        'size': 8,
        'color': 'black'
    }
    )

subplots4x3.update_yaxes(
title_text="-AVG_DL", row=1, col=1
)
subplots4x3.update_yaxes(
title_text="PC", row=2, col=1
)
subplots4x3.update_yaxes(
title_text="HV", row=3, col=1
)
subplots4x3.update_yaxes(
title_text="HV_ratio", row=4, col=1
)
subplots4x3.update_xaxes(
    title_text="ticks", row=3, col=1
)
subplots4x3.update_xaxes(
    title_text="ticks", row=3, col=2
)
subplots4x3.update_xaxes(
    title_text="ticks", row=4, col=3
)


def create_objective_diagrams(
        network_name: str,
        moving_portion: str,
        config_name: str, 
        baseline_name: str, 
        show_diagram: bool = False, 
        export_diagrams: bool=False, 
        col_index= 0):
    shape = (0, 1)
    config_results_directory = "./{datastore_directory_name}/{network_name}/{configuration_name}/".format(network_name=network_name, configuration_name= config_name, datastore_directory_name = datastore_directory_name)
    with open("{results_directory}{configuration_name}.json".format(results_directory=config_results_directory, configuration_name =config_name)) as openfile:
        config_json_object = json.load(openfile)
        shape = (0, int(config_json_object["running_time_in_s"] + 1))


    config_results_directory = "./{datastore_directory_name}/{network_name}/{configuration_name}/".format(network_name=network_name, configuration_name= config_name, datastore_directory_name=datastore_directory_name)
    base_line_results_directory = "./{datastore_directory_name}/{network_name}/{configuration_name}/".format(network_name=network_name, configuration_name= baseline_name, datastore_directory_name=datastore_directory_name)
    ## create objective list
    time_and_frames_list = np.empty(shape)
    directory_contents = os.listdir(config_results_directory)
    base_line_directory_contents = os.listdir(base_line_results_directory)

    objective_1_list = np.empty(shape)
    objective_2_list = np.empty(shape)  
    base_line_objective_1_list = np.empty(shape)
    base_line_objective_2_list = np.empty(shape)  

    #  user_dl_datarate variance
    variance_list = np.empty(shape)
    # add user_dl_datarate interquartile range
    iqr_list = np.empty(shape)
    # add user_dl_datarate lower quartile median
    q25_list = np.empty(shape)
    # add user_dl_datarate total median
    q50_list = np.empty(shape)
    # add user_dl_datarate upper median
    q75_list = np.empty(shape)
    # add user_dl_datarate min value
    min_list = np.empty(shape)
    # add user_dl_datarate max value
    max_list = np.empty(shape)

    #  user_dl_datarate variance
    base_line_variance_list = np.empty(shape)
    # add user_dl_datarate interquartile range
    base_line_iqr_list = np.empty(shape)
    # add user_dl_datarate lower quartile median
    base_line_q25_list = np.empty(shape)
    # add user_dl_datarate total median
    base_line_q50_list = np.empty(shape)
    # add user_dl_datarate upper median
    base_line_q75_list = np.empty(shape)
    # add user_dl_datarate min value
    base_line_min_list = np.empty(shape)
    # add user_dl_datarate max value
    base_line_max_list = np.empty(shape)
    
    time_and_frames_list = np.empty(shape)
    for run in directory_contents:
        if "objectives_result" in run:
            with open(config_results_directory + "/" + run, mode="r",encoding="utf-8") as openfile:

                config_json_object = json.load(openfile)
                result_dict_object = np.array(config_json_object["objective_history"])
                
                objective_1_list = np.append(np.array([result_dict_object[:, -2]]), objective_1_list, axis=0)
                objective_2_list = np.append(np.array([result_dict_object[:, -1]]), objective_2_list, axis=0)
                
                # user_dl_datarate variance
                variance_list = np.append(np.array([result_dict_object[:, 4]]), variance_list, axis=0)
                # add user_dl_datarate interquartile range
                iqr_list  = np.append(np.array([result_dict_object[:, 5]]), iqr_list, axis=0)
                # add user_dl_datarate lower quartile median
                q25_list = np.append(np.array([result_dict_object[:, 6]]), q25_list, axis=0)
                # add user_dl_datarate total median
                q50_list = np.append(np.array([result_dict_object[:, 7]]), q50_list, axis=0)
                # add user_dl_datarate upper median
                q75_list = np.append(np.array([result_dict_object[:, 8]]), q75_list, axis=0)
                # add user_dl_datarate min value
                min_list = np.append(np.array([result_dict_object[:, 9]]), min_list, axis=0)
                # add user_dl_datarate max value
                max_list = np.append(np.array([result_dict_object[:, 10]]), max_list, axis=0)
                
                time_and_frames_list = result_dict_object[:, 0:2]
    
    for run in base_line_directory_contents:
        if "objectives_result" in run:
            with open(base_line_results_directory + "/" + run, mode="r",encoding="utf-8") as openfile:

                config_json_object = json.load(openfile)
                result_dict_object = np.array(config_json_object["objective_history"])
                
                base_line_objective_1_list = np.append(np.array([result_dict_object[:, -2]]), base_line_objective_1_list, axis=0)
                base_line_objective_2_list = np.append(np.array([result_dict_object[:, -1]]), base_line_objective_2_list, axis=0)
                
                # user_dl_datarate variance
                base_line_variance_list = np.append(np.array([result_dict_object[:, 4]]), base_line_variance_list, axis=0)
                # add user_dl_datarate interquartile range
                base_line_iqr_list  = np.append(np.array([result_dict_object[:, 5]]), base_line_iqr_list, axis=0)
                # add user_dl_datarate lower quartile median
                base_line_q25_list = np.append(np.array([result_dict_object[:, 6]]), base_line_q25_list, axis=0)
                # add user_dl_datarate total median
                base_line_q50_list = np.append(np.array([result_dict_object[:, 7]]), base_line_q50_list, axis=0)
                # add user_dl_datarate upper median
                base_line_q75_list = np.append(np.array([result_dict_object[:, 8]]), base_line_q75_list, axis=0)
                # add user_dl_datarate min value
                base_line_min_list = np.append(np.array([result_dict_object[:, 9]]), base_line_min_list, axis=0)
                # add user_dl_datarate max value
                base_line_max_list = np.append(np.array([result_dict_object[:, 10]]), base_line_max_list, axis=0)
                
    

    ## normalize objectives
    objective_1_max_value = np.append(objective_1_list, base_line_objective_1_list).max()
    objective_2_max_value = np.append(objective_2_list, base_line_objective_2_list).max()
    objective_1_min_value = np.append(objective_1_list, base_line_objective_1_list).min()
    objective_2_min_value = np.append(objective_2_list, base_line_objective_2_list).min()

    normalize_factor_1 = (objective_1_max_value - objective_1_min_value) if (objective_1_max_value - objective_1_min_value) != 0 else objective_1_max_value
    normalize_factor_2 = (objective_2_max_value - objective_2_min_value) if (objective_2_max_value - objective_2_min_value) != 0 else objective_2_max_value
    
    objective_1_normalized_list = np.array((objective_1_list - objective_1_min_value) / normalize_factor_1)
    objective_2_normalized_list = np.array((objective_2_list - objective_2_min_value) / normalize_factor_2)
    base_line_objective_1_normalized_list = np.array((base_line_objective_1_list - objective_1_min_value) / normalize_factor_1)
    base_line_objective_2_normalized_list = np.array((base_line_objective_2_list - objective_2_min_value) / normalize_factor_2)

    # create averaged objective lists over all runs

    objective_1_avrg_list =np.average(objective_1_list, axis=0)
    objective_2_avrg_list = np.average(objective_2_list, axis=0)

    objective_1_normalized_avrg_list = np.average(objective_1_normalized_list, axis=0)
    objective_2_normalized_avrg_list = np.average(objective_2_normalized_list, axis=0)

    #  user_dl_datarate variance
    variance_list_avrg_list = np.average(variance_list, axis=0)
    # add user_dl_datarate interquartile range
    iqr_list_avrg_list  = np.average(iqr_list, axis=0)
    # add user_dl_datarate lower quartile median
    q25_list_avrg_list = np.average(q25_list, axis=0)
    # add user_dl_datarate total median
    q50_list_avrg_list = np.average(q50_list, axis=0)
    # add user_dl_datarate upper median
    q75_list_avrg_list = np.average(q75_list, axis=0)
    # add user_dl_datarate min value
    min_list_avrg_list = np.average( min_list, axis=0)
    # add user_dl_datarate max value
    max_list_avrg_list = np.average(max_list, axis=0)
    
    base_line_objective_1_avrg_list =np.average(base_line_objective_1_list, axis=0)
    base_line_objective_2_avrg_list = np.average(base_line_objective_2_list, axis=0)

    base_line_objective_1_normalized_avrg_list = np.average(base_line_objective_1_normalized_list, axis=0)
    base_line_objective_2_normalized_avrg_list = np.average(base_line_objective_2_normalized_list, axis=0)

    #  user_dl_datarate variance
    base_line_variance_list_avrg_list = np.average(base_line_variance_list, axis=0)
    # add user_dl_datarate interquartile range
    base_line_iqr_list_avrg_list  = np.average(base_line_iqr_list, axis=0)
    # add user_dl_datarate lower quartile median
    base_line_q25_list_avrg_list = np.average(base_line_q25_list, axis=0)
    # add user_dl_datarate total median
    base_line_q50_list_avrg_list = np.average(base_line_q50_list, axis=0)
    # add user_dl_datarate upper median
    base_line_q75_list_avrg_list = np.average(base_line_q75_list, axis=0)
    # add user_dl_datarate min value
    base_line_min_list_avrg_list = np.average(base_line_min_list, axis=0)
    # add user_dl_datarate max value
    base_line_max_list_avrg_list = np.average(base_line_max_list, axis=0)

    ## save averaged objective set in current directory
    with open(config_results_directory + "/averaged_objectives.json", "w") as outfile:
        dump_array = np.array([time_and_frames_list[:,0], 
                            time_and_frames_list[:, 1], 
                            objective_1_avrg_list, 
                            objective_2_avrg_list,
                            variance_list_avrg_list,
                            iqr_list_avrg_list,
                            q25_list_avrg_list,
                            q50_list_avrg_list,
                            q75_list_avrg_list,
                            min_list_avrg_list,
                            max_list_avrg_list])
        
        dump_array= np.transpose(dump_array).tolist()
        json.dump(dump_array, outfile)
        
    ## save averaged objective set in current directory
    with open(base_line_results_directory + "/averaged_objectives.json", "w") as outfile:
        dump_array = np.array([time_and_frames_list[:,0], 
                            time_and_frames_list[:, 1], 
                            base_line_objective_1_avrg_list, 
                            base_line_objective_2_avrg_list,
                            base_line_variance_list_avrg_list,
                            base_line_iqr_list_avrg_list,
                            base_line_q25_list_avrg_list,
                            base_line_q50_list_avrg_list,
                            base_line_q75_list_avrg_list,
                            base_line_min_list_avrg_list,
                            base_line_max_list_avrg_list])
        
        dump_array= np.transpose(dump_array).tolist()
        json.dump(dump_array, outfile)

#############################    
# plots for all runs
#############################

    ## objectives plot
    fig2 = go.Figure()
    fig2.update_layout(
        title= config_name,
        xaxis_title="time in ticks",
        template= template_one_diagram,
        yaxis_title="-AVG_DL in bit/s"
        )
    
    fig3 = go.Figure()
    fig3.update_layout(
        title= config_name,
        xaxis_title="time in ticks",
        template= template_one_diagram,
        yaxis_title="PC in watts"
        )

    for index, _ in enumerate(objective_1_list):
        fig2.add_trace(
            go.Scatter(
                x=time_and_frames_list[:, 1],
                y=objective_1_list[index],
                name="run_i",
                showlegend= True if index == 0 else False,
                mode='lines',
                line={"color": runs_color,"width": lineWidth},
                opacity=opacity_runs,
            ))
        fig3.add_trace(
            go.Scatter(
                x=time_and_frames_list[:, 1],
                y=objective_2_list[index],
                name="run_i",
                showlegend=True if index == 0 else False,
                mode='lines',
                line={"color": runs_color, "width": lineWidth},
                opacity=opacity_runs,
            ))
        
    fig2.add_trace(
        go.Scatter(
            x=time_and_frames_list[:, 1],
            y=objective_1_avrg_list,
            name="evo",
            mode='lines',
            line={"color": evo_color, "width": lineWidth},
        ))
    fig2.add_trace(
        go.Scatter(
            x=time_and_frames_list[:, 1],
            y=base_line_objective_1_avrg_list,
            name="greedy",
            mode='lines',
            line={ "color": greedy_color, "width": lineWidth},
        ))
    fig3.add_trace(
            go.Scatter(
                x=time_and_frames_list[:, 1],
                y=objective_2_avrg_list,
                name="evo",
                mode='lines',
                line={ "color": evo_color, "width": lineWidth},
            ))
    fig3.add_trace(
        go.Scatter(
            x=time_and_frames_list[:, 1],
            y=base_line_objective_2_avrg_list,
            name="greedy",
            mode='lines',
            line={ "color": greedy_color, "width": lineWidth},
        ))
    ## normalized objectives plot
    fig4 = go.Figure()

    fig4.update_layout(
        title= config_name,
        xaxis_title="time in ticks",
        yaxis_title="-AVG_DL",
        template= template_one_diagram)
    
    fig5 = go.Figure()
    fig5.update_layout(
        title= config_name,
        xaxis_title="time in ticks",
        yaxis_title="PC",
        template=template_one_diagram)

    for index, _ in enumerate(objective_1_list):
        fig4.add_trace(
            go.Scatter(
                x=time_and_frames_list[:, 1],
                y=objective_1_normalized_list[index],
                name="run_i",
                showlegend=True if index == 0 else False,
                mode='lines',
                line={"color": runs_color, "width": lineWidth},
                opacity=opacity_runs,
            ))
        ###TODO delete###
        subplots4x3.add_trace(
            go.Scatter(
                x=time_and_frames_list[:, 1],
                y=objective_1_normalized_list[index],
                name="run_i",
                showlegend=False,
                mode='lines',
                line={"color": runs_color, "width": lineWidth},
                opacity=opacity_runs,
            ), row=1, col=col_index)
        
        fig5.add_trace(
            go.Scatter(
                x=time_and_frames_list[:, 1],
                y=objective_2_normalized_list[index],
                name="run_i",
                showlegend=True if index == 0 else False,
                mode='lines',
                line={"color": runs_color, "width": lineWidth},
                opacity=opacity_runs,
            ))
        
        ###TODO delete###
        subplots4x3.add_trace( go.Scatter(
                x=time_and_frames_list[:, 1],
                y=objective_2_normalized_list[index],
                name="run_i",
                showlegend=False,
                mode='lines',
                line={"color": runs_color, "width": lineWidth},
                opacity=opacity_runs,
            ), row=2, col=col_index)
        
    fig4.add_trace(
        go.Scatter(
            x=time_and_frames_list[:, 1],
            y=objective_1_normalized_avrg_list,
            name="evo",
            mode='lines',
            line={"color": evo_color, "width": lineWidth},
        ))
    ###TODO delete
    subplots4x3.add_trace(
        go.Scatter(
            x=time_and_frames_list[:, 1],
            y=objective_1_normalized_avrg_list,
            name="evo",
            mode='lines',
            line={"color": evo_color, "width": lineWidth},
        ), row=1, col=col_index
    )
    fig4.add_trace(
        go.Scatter(
            x=time_and_frames_list[:, 1],
            y=base_line_objective_1_normalized_avrg_list,
            name="greedy",
            mode='lines',
            line={"color": greedy_color, "width": lineWidth},
        ))
    ##TODO delete###
    subplots4x3.add_trace(
        go.Scatter(
            x=time_and_frames_list[:, 1],
            y=base_line_objective_1_normalized_avrg_list,
            name="greedy",
            mode='lines',
            line={"color": greedy_color, "width": lineWidth},
        ),
        row=1, col=col_index
    )
    fig5.add_trace(
        go.Scatter(
            x=time_and_frames_list[:, 1],
            y=objective_2_normalized_avrg_list,
            name="evo" ,
            mode='lines',
            line={"color": evo_color, "width": lineWidth}
        ))
    ## TODO delete
    subplots4x3.add_trace(
        go.Scatter(
            x=time_and_frames_list[:, 1],
            y=objective_2_normalized_avrg_list,
            name="evo" ,
            mode='lines',
            line={"color": evo_color, "width": lineWidth}
        ), row=2, col=col_index
    )
    fig5.add_trace(
        go.Scatter(
            x=time_and_frames_list[:, 1],
            y=base_line_objective_2_normalized_avrg_list,
            name="greedy",
            mode='lines',
            line={"color": greedy_color, "width": lineWidth}
        ))
    ##TODO delete
    subplots4x3.add_trace(
        go.Scatter(
            x=time_and_frames_list[:, 1],
            y=base_line_objective_2_normalized_avrg_list,
            name="greedy",
            mode='lines',
            line={"color": greedy_color, "width": lineWidth}
        ), row=2, col=col_index
    )
        
    print("######## plots for experiment {configuration_name} #########".format(configuration_name =config_name))
    if show_diagram:
        fig2.show()
        fig3.show()
        fig4.show()
        fig5.show()
        
    if export_diagrams:
        diagram_path= "diagrams/" + network_name + "/" + moving_portion + "/" + config_name + "/"
        if not os.path.exists(diagram_path):
            os.makedirs(diagram_path)
        
        fig2.write_image(diagram_path +"AVG_DL.pdf")
        fig3.write_image(diagram_path +"PC.pdf")
        fig4.write_image(diagram_path +"AVG_DL_normalized.pdf")
        fig5.write_image(diagram_path +"PC_normalized.pdf")


############# hypervolume plots #########################
        
def create_avrg_hypervolume_ratio_dataset(
    network_name: str,
    config_name: str, 
    moving_portion: str,
    baseline_name: str, 
    show_diagram: bool = False, 
    export_diagrams: bool = False,
    write_to_ratio_file: bool=False, 
    col_index= 0):

    config_results_directory = "./{datastore_directory_name}/{network_name}/{configuration_name}/".format(network_name=network_name, configuration_name= config_name, datastore_directory_name = datastore_directory_name)
    baseline_results_directory = "./{datastore_directory_name}/{network_name}/{configuration_name}/".format(network_name=network_name, configuration_name= baseline_name, datastore_directory_name=datastore_directory_name)
       
    
    with open("{results_directory}{configuration_name}.json".format(results_directory=config_results_directory, configuration_name = config_name)) as openfile:
        config_json_object = json.load(openfile)


    hv_matrix = np.empty(shape=(int(config_json_object["iterations"]), int(config_json_object["running_time_in_s"] + 1)))
    hv_baseline = np.empty(shape=(int(config_json_object["running_time_in_s"] + 1)))

    objecitve_1_min_max_pertick_list = np.full(shape=( int(config_json_object["running_time_in_s"] + 1), 2), fill_value=np.nan)
    objecitve_2_min_max_pertick_list = np.full(shape=( int(config_json_object["running_time_in_s"] + 1), 2),fill_value=np.nan)

    ## find min, max values for each objective over all runs and configs per tick

    ## create objective list
    config_directory_contents = os.listdir(config_results_directory)
    baseline_directory_contents = os.listdir(baseline_results_directory)
    dirs = [config_directory_contents, baseline_directory_contents]

    for dir_index, dir in enumerate(dirs):
        for run in dir:
            if "objectives_result" in run:
                with open(config_results_directory + "/" + run if dir_index == 0 else baseline_results_directory + "/" + run, mode="r",encoding="utf-8") as openfile:

                    config_json_object = json.load(openfile)
                    selection_history = np.array(config_json_object["objective_history"])[:, -2:]
                    paretofront_history =config_json_object["objectivespace_history"]
                    time_and_frames_list =  np.array(config_json_object["objective_history"])[:, 0:2]
                    
                    for tick_index, _ in enumerate(time_and_frames_list):
                        combined_history_per_tick = []
                        combined_history_per_tick.append(list(selection_history[tick_index][::-1]))

                        if len(paretofront_history[tick_index]) > 0 :
                            for _, current_ind in enumerate(np.array(paretofront_history[tick_index])[:,0:2]):
                                combined_history_per_tick.append(list(current_ind))
                        
                        combined_history_per_tick = np.array(combined_history_per_tick)

                    
                        current_min_obj_1 , current_min_obj_2= combined_history_per_tick.min(axis=0)
                        current_max_obj_1 , current_max_obj_2= combined_history_per_tick.max(axis=0)

                        if np.isnan(objecitve_1_min_max_pertick_list[tick_index][0]) or objecitve_1_min_max_pertick_list[tick_index][0] > current_min_obj_1:
                            objecitve_1_min_max_pertick_list[tick_index][0] = current_min_obj_1
                        
                        if np.isnan(objecitve_1_min_max_pertick_list[tick_index][1]) or objecitve_1_min_max_pertick_list[tick_index][1] < current_max_obj_1:
                            objecitve_1_min_max_pertick_list[tick_index][1] = current_max_obj_1
                        
                        if np.isnan(objecitve_2_min_max_pertick_list[tick_index][0]) or objecitve_2_min_max_pertick_list[tick_index][0] > current_min_obj_2:
                            objecitve_2_min_max_pertick_list[tick_index][0] = current_min_obj_2
                        
                        if np.isnan(objecitve_2_min_max_pertick_list[tick_index][1]) or objecitve_2_min_max_pertick_list[tick_index][1] < current_max_obj_2:
                            objecitve_2_min_max_pertick_list[tick_index][1] = current_max_obj_2
        
    ## calculate all hyper volumes per tick for each config and run  using the  min/max-per tick values for each objective
                        
    for dir_index, dir in enumerate(dirs):

        run_index = 0
        for _, run in enumerate(dir):
            if "objectives_result" in run:
                with open(config_results_directory + "/" + run if dir_index == 0 else baseline_results_directory  + "/" + run, mode="r",encoding="utf-8") as openfile:
        
                    config_json_object = json.load(openfile)
                    
                    objective_1_list = np.array(config_json_object["objective_history"])[:, -1]
                    objective_2_list = np.array(config_json_object["objective_history"])[:, -2]


                    for tick_index, _ in enumerate(objective_1_list):
                        ## normalize objectives
                        normalize_factor_1 = (objecitve_1_min_max_pertick_list[tick_index][1] - objecitve_1_min_max_pertick_list[tick_index][0]) if (objecitve_1_min_max_pertick_list[tick_index][1] - objecitve_1_min_max_pertick_list[tick_index][0]) != 0 else objecitve_1_min_max_pertick_list[tick_index][1]
                        normalize_factor_2 = (objecitve_2_min_max_pertick_list[tick_index][1] - objecitve_2_min_max_pertick_list[tick_index][0]) if (objecitve_2_min_max_pertick_list[tick_index][1] - objecitve_2_min_max_pertick_list[tick_index][0]) != 0 else objecitve_2_min_max_pertick_list[tick_index][1]

                        objective_1_normalized_tick_value = (objective_1_list[tick_index] - objecitve_1_min_max_pertick_list[tick_index][0]) / normalize_factor_1
                        objective_2_normalized_tick_value = (objective_2_list[tick_index] - objecitve_2_min_max_pertick_list[tick_index][0]) / normalize_factor_2

                        ## calculate hv value for this tick
                        approx_ideal = np.array([0,0])
                        approx_nadir = np.array([1,1])

                        
                        hv = Hypervolume(ref_point=np.array([1.1, 1.1]),
                                norm_ref_point=True,
                                zero_to_one=False,
                                ideal=approx_ideal,
                                nadir=approx_nadir)
                        hv_value_tick = hv(np.array([objective_1_normalized_tick_value, objective_2_normalized_tick_value]))
                        
                        if dir_index == 0:
                            hv_matrix[run_index][tick_index] = hv_value_tick
                        else:
                            hv_baseline[tick_index] = hv_value_tick
                run_index += 1

    fig1 = go.Figure()
    fig1.update_layout(
        title=config_name,
        xaxis_title="time in ticks",
        yaxis_title="HV",
        template=template_one_diagram
        )


    ## add experiment hypervolumes 

    for index, value in enumerate(hv_matrix):
        fig1.add_trace(
            go.Scatter(
                x=np.array(range(hv_matrix.shape[1])),
                y=value,
                mode='lines',
                name= f"run_i",
                showlegend=True if index == 0 else False,
                line={"color": runs_color, "width": lineWidth},
                opacity=opacity_runs,
            ))
    fig1.add_trace(
        go.Scatter(
            x=np.array(range(hv_matrix.shape[1])),
            y = hv_baseline,
            mode='lines',
            line={"color": greedy_color, "width": lineWidth},
            name="greedy"
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=np.array(range(hv_matrix.shape[1])),
            y = np.average(hv_matrix, axis=0),
            line={"color": evo_color, "width": lineWidth},
            mode='lines',
            name="evo"
        )
    )
    hypervolume_difference_list = np.average(hv_matrix- hv_baseline, axis=0)
    hv_ratio_list = np.average((hv_matrix / hv_baseline), axis = 0)

    print("accumulated hypervolume difference: {sum}".format(sum=np.sum(hypervolume_difference_list)))
    print("accumulated hypervolume ratio {config_name}/{baseline_name}: {ratio} ".format(baseline_name=baseline_name, config_name=config_name, ratio=np.sum(hv_ratio_list)))
    print("average hypervolume ratio {config_name}/{baseline_name}: {ratio}".format(baseline_name=baseline_name, config_name=config_name, ratio=np.average(hv_ratio_list)))
    colors = ["green" if value >= 0 else "red" for value in hypervolume_difference_list]

    fig6 = go.Figure()

    fig6.update_layout(
        title=config_name,
        xaxis_title="time in ticks",
        yaxis_title="HV",
        bargap=0,
        template=template_one_diagram
        )
    
    fig6.add_trace(
        go.Bar(
            x=np.array(range(hv_matrix.shape[1])),
            y=hypervolume_difference_list,
            marker_color= colors,
            showlegend=False,
            opacity=1,
        )
    )
    ##TODO delete##
    subplots4x3.add_trace( 
        go.Bar(
            x=np.array(range(hv_matrix.shape[1])),
            y=hypervolume_difference_list,
            marker_color= colors,
            showlegend=False,
            opacity=1,
        ), row=3, col=col_index)

    fig6.add_trace(
    go.Scatter(
        x=np.array(range(hv_matrix.shape[1])),
        y=hv_baseline,
        mode='lines',
        line={"color": greedy_color, "width": lineWidth},
        opacity=1,
        name="greedy",
    ))
    ##TODO delete
    subplots4x3.add_trace(
        go.Scatter(
        x=np.array(range(hv_matrix.shape[1])),
        y=hv_baseline,
        mode='lines',
        line={"color": greedy_color, "width": lineWidth},
        opacity=1,
        name="greedy",
    ), row=3, col=col_index
    )
    
    fig6.add_trace(
        go.Scatter(
            x=np.array(range(hv_matrix.shape[1])),
            y=np.average(hv_matrix, axis=0),
            mode='lines',
            line={"color": evo_color, "width": lineWidth},
            opacity=1,
            name= "evo",
        ))
    ##TODO delete
    subplots4x3.add_trace(
        go.Scatter(
            x=np.array(range(hv_matrix.shape[1])),
            y=np.average(hv_matrix, axis=0),
            mode='lines',
            line={"color": evo_color, "width": lineWidth},
            opacity=1,
            name= "evo",
        ), row=3, col=col_index
    )
    
    fig7 = go.Figure()
    fig7.update_layout(
        title=config_name,
        xaxis_title="time in ticks",
        yaxis_title="HV ratio",
        bargap=0,
        template=template_one_diagram
        )
    
    fig7.add_trace(
        go.Bar(
            x=np.array(range(hv_matrix.shape[1])),
            y=hv_ratio_list,
            marker_color= colors,
            showlegend=False,
            opacity=1
        )
    )
    ##TODO delete
    subplots4x3.add_trace(
        go.Bar(
            x=np.array(range(hv_matrix.shape[1])),
            y=hv_ratio_list,
            marker_color= colors,
            showlegend=False,
            opacity=1
        ), row=4, col=col_index
    )
    
    fig7.add_hline(y=1, line_width=lineWidth)

    ##TODO delete
    subplots4x3.add_hline(y=1, row=4,col=col_index,line_width=lineWidth)

    if show_diagram:
        fig6.show()
        fig7.show()
        fig1.show()
    
    if export_diagrams:
        diagram_path= "diagrams/" + network_name + "/" + moving_portion + "/" + config_name + "/"
        if not os.path.exists(diagram_path):
            os.makedirs(diagram_path)
        fig6.write_image(diagram_path+"hv_difference.pdf")
        fig7.write_image(diagram_path +"hv_ratio.pdf")
        fig1.write_image(diagram_path +"hv_runs.pdf")

    if write_to_ratio_file:
        hv_ratio_list = np.average((hv_matrix / hv_baseline), axis=1 )
        
        file_name = "./diagrams/avg_hypervolume_ratios.json"

        if os.path.exists(file_name):
            with open(file_name, "r") as json_file:
                json_object: dict = json.load(json_file)
        else:
            json_object = {}
       
        if network_name not in json_object:
            json_object[network_name] = {config_name: list(hv_ratio_list)}
        else:
            json_object[network_name][config_name] = list(hv_ratio_list)

        with open(file_name, "w") as json_file:
            json.dump(json_object, json_file)


################ boxplots ##########

## create objective list by combining datasets of all experiments
def create_box_plots_averaged(
        network_name:str, 
        config_name:str, 
        baseline_name: str, 
        show_diagram: bool,
        export_diagrams: bool,
        moving_portion: str):
    shape = (0, 1)
    config_results_directory = "./{datastore_directory_name}/{network_name}/{configuration_name}/".format(network_name=network_name, configuration_name= config_name, datastore_directory_name = datastore_directory_name)
    with open("{results_directory}{configuration_name}.json".format(results_directory=config_results_directory, configuration_name =config_name)) as openfile:
        config_json_object = json.load(openfile)
        shape = (0, int(config_json_object["running_time_in_s"] + 1))

    # user_dl_datarate variance
    variance_list = np.empty(shape)
    # add user_dl_datarate interquartile range
    iqr_list  = np.empty(shape)
    # add user_dl_datarate lower quartile median
    q25_list = np.empty(shape)
    # add user_dl_datarate total median
    q50_list = np.empty(shape)
    # add user_dl_datarate upper median
    q75_list = np.empty(shape)
    # add user_dl_datarate min value
    min_list = np.empty(shape)
    # add user_dl_datarate max value
    max_list = np.empty(shape)
    objective_1_list = np.empty(shape)
    time_and_frames_list = np.empty(shape[1])
    
    # user_dl_datarate variance
    base_line_variance_list = np.empty(shape)
    # add user_dl_datarate interquartile range
    base_line_iqr_list  = np.empty(shape)
    # add user_dl_datarate lower quartile median
    base_line_q25_list = np.empty(shape)
    # add user_dl_datarate total median
    base_line_q50_list = np.empty(shape)
    # add user_dl_datarate upper median
    base_line_q75_list = np.empty(shape)
    # add user_dl_datarate min value
    base_line_min_list = np.empty(shape)
    # add user_dl_datarate max value
    base_line_max_list = np.empty(shape)
    base_line_objective_1_list = np.empty(shape)


    results_directory = "./{datastore_directory_name}/{network_name}/{configuration_name}/".format(network_name=network_name, configuration_name= config_name, datastore_directory_name=datastore_directory_name)
    base_line_results_directory = "./{datastore_directory_name}/{network_name}/{configuration_name}/".format(network_name=network_name, configuration_name= baseline_name, datastore_directory_name=datastore_directory_name)

    
    with open(results_directory + "/" + "averaged_objectives.json", mode="r",encoding="utf-8") as openfile:

        result_dict_object = np.array(json.load(openfile))

        objective_1_list = np.append(objective_1_list, -np.array([result_dict_object[:, 2]]), axis=0)

        # user_dl_datarate variance
        variance_list = np.append( variance_list, np.array([result_dict_object[:, 4]]), axis=0)
        # add user_dl_datarate interquartile range
        iqr_list  = np.append(iqr_list, np.array([result_dict_object[:, 5]]), axis=0)
        # add user_dl_datarate lower quartile median
        q25_list = np.append(q25_list, np.array([result_dict_object[:, 6]]), axis=0)
        # add user_dl_datarate total median
        q50_list = np.append(q50_list, np.array([result_dict_object[:, 7]]), axis=0)
        # add user_dl_datarate upper median
        q75_list = np.append(q75_list, np.array([result_dict_object[:, 8]]), axis=0)
        # add user_dl_datarate min value
        min_list = np.append(min_list, np.array([result_dict_object[:, 9]]), axis=0)
        # add user_dl_datarate max value
        max_list = np.append(max_list, np.array([result_dict_object[:, 10]]), axis=0)

        time_and_frames_list = result_dict_object[:, 0:2]
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(y =result_dict_object[:,6:], boxpoints="outliers"))
        fig_box.update_traces(q1=result_dict_object[:, 6], median=result_dict_object[:, 7],
                        q3=result_dict_object[:, 8],mean=-np.array([result_dict_object[:, 2]]))

        fig_box.update_layout(
        title=config_name,
        xaxis_title="time in ticks",
        yaxis_title="boxplot",
        template=template_one_diagram,
        bargap=0
        )
        if show_diagram:
            fig_box.show()

        with open(base_line_results_directory + "/" + "averaged_objectives.json", mode="r",encoding="utf-8") as openfile:

            result_dict_object = np.array(json.load(openfile))

            base_line_objective_1_list = np.append(base_line_objective_1_list, -np.array([result_dict_object[:, 2]]), axis=0)

            # user_dl_datarate variance
            base_line_variance_list = np.append( base_line_variance_list, np.array([result_dict_object[:, 4]]), axis=0)
            # add user_dl_datarate interquartile range
            base_line_iqr_list  = np.append(base_line_iqr_list, np.array([result_dict_object[:, 5]]), axis=0)
            # add user_dl_datarate lower quartile median
            base_line_q25_list = np.append(base_line_q25_list, np.array([result_dict_object[:, 6]]), axis=0)
            # add user_dl_datarate total median
            base_line_q50_list = np.append(base_line_q50_list, np.array([result_dict_object[:, 7]]), axis=0)
            # add user_dl_datarate upper median
            base_line_q75_list = np.append(base_line_q75_list, np.array([result_dict_object[:, 8]]), axis=0)
            # add user_dl_datarate min value
            base_line_min_list = np.append(base_line_min_list, np.array([result_dict_object[:, 9]]), axis=0)
            # add user_dl_datarate max value
            base_line_max_list = np.append(base_line_max_list, np.array([result_dict_object[:, 10]]), axis=0)

            time_and_frames_list = result_dict_object[:, 0:2]
            fig_box = go.Figure()
            
            fig_box.add_trace(go.Box(y =result_dict_object[:,6:], name="greedy", boxpoints="outliers"))
            fig_box.update_traces(q1=result_dict_object[:, 6], median=result_dict_object[:, 7],
                           q3=result_dict_object[:, 8],mean=-np.array([result_dict_object[:, 2]]))

            fig_box.update_layout(
            title="greedy",
            xaxis_title="time in ticks",
            yaxis_title="boxplot",
            template=template_one_diagram,
            bargap=0
            )
            if show_diagram:
                fig_box.show()
    
    for index, _ in enumerate(variance_list):
        ## variance and iqr plots
        fig_variance = go.Figure()
        fig_iqr = go.Figure()

        fig_variance.update_layout(
            title= "standard deviation for DL",
            xaxis_title="time in ticks",
            yaxis_title="DL",
            template=template_one_diagram
            )
        fig_iqr.update_layout(
            title= "IQR",
            xaxis_title="time in ticks",
            yaxis_title="IQR",
            template=template_one_diagram
            )
        
        fig_variance.add_trace(
            go.Scatter(
                x=time_and_frames_list[:, 1],
                y=np.sqrt(variance_list[index]),
                name="evo",
                mode='lines',
                line={"color":evo_color, "width": lineWidth},
            ))
        fig_variance.add_trace(
            go.Scatter(
                x=time_and_frames_list[:, 1],
                y=np.sqrt(base_line_variance_list[index]),
                name="greedy",
                line={"color":greedy_color, "width": lineWidth},
                mode='lines',
            ))
        fig_iqr.add_trace(
            go.Scatter(
                x=time_and_frames_list[:, 1],
                y=iqr_list[index],
                name="evo",
                line={"color":evo_color, "width": lineWidth},
                mode='lines',
            ))
        fig_iqr.add_trace(
            go.Scatter(
                x=time_and_frames_list[:, 1],
                y=base_line_iqr_list[index],
                name="greedy",
                mode='lines',
                line={"color":greedy_color, "width": lineWidth}
            ))
        if show_diagram:
            fig_variance.show()
            fig_iqr.show()

        if export_diagrams:
            diagram_path= "diagrams/" + network_name + "/" + moving_portion + "/" + config_name + "/"
            if not os.path.exists(diagram_path):
                os.makedirs(diagram_path)
            fig_variance.write_image(diagram_path+"DL_variance.pdf")
            fig_iqr.write_image(diagram_path +"DL_iqr.pdf")
            fig_box.write_image(diagram_path + "DL_box_plot.pdf")
            

        print("avrg std devaition {configuration_name}: {value:e}".format(configuration_name =config_name, value=np.sqrt(np.average(variance_list[index]))))
        print("avrg std devaition {configuration_name}: {value:e}".format(configuration_name = baseline_name, value=np.sqrt(np.average(base_line_variance_list[index]))))
        print("avrg IQR {configuration_name}: {value:e}".format(configuration_name = config_name, value=np.average(iqr_list[index])))
        print("avrg IQR {configuration_name}: {value:e}".format(configuration_name = baseline_name, value=np.average(base_line_iqr_list[index])))
        print()
        print()


def select_solution(objective_space: np.ndarray, weights= []):

    approx_ideal = objective_space.min(axis=0)
    approx_nadir = objective_space.max(axis=0)

    # TODO -> handle numpy divide by zero with  np.seterr(divide='ignore', invalid='ignore') maybe
    np.seterr(divide='ignore', invalid='ignore')
    nF = (objective_space - approx_ideal) / (approx_nadir - approx_ideal)
    decomp = ASF()
    
    if len(weights) != objective_space.shape[1]:
        weights = [1/objective_space.shape[1] for _ in range(objective_space.shape[1])]
        
    weights_np= np.array(weights)

    i = decomp.do(nF, 1/weights_np).argmin()
    return i

def create_pareto_history_plots(
        export_diagrams: bool,
        show_diagram: bool,
        datastore_directory_name =datastore_directory_name,
        network_name = "het_C100",
        config_name="het_C100_MP100_MS1,2ms_evo",
        moving_portion="MP100",
        greedy_mirror="het_C100_MP100_MS1,2ms_greedy",
        tick =  1200):
    
    results_directory = "./{datastore_directory_name}/{network_name}/{config_name}".format(datastore_directory_name=datastore_directory_name, network_name=network_name, config_name=config_name)
   
    hv = Hypervolume(ref_point=np.array([1.1, 1.1]),
                        norm_ref_point=True,
                        zero_to_one=False,
                        ideal=[0,0],
                        nadir=[1,1])
    with open(results_directory + "/" + "objectives_result_1.json", mode="r",encoding="utf-8") as openfile:

            json_object = json.load(openfile)

            objectivespace_history = json_object["objectivespace_history"]
            objective_history = np.array(json_object["objective_history"])
            time_and_frames_list = objective_history[:, 0:2]
                    
            objective_1_list =np.array(objective_history[:, -2])
            objective_2_list = np.array(objective_history[:, -1])

            pareto_front = np.array(objectivespace_history[tick])


            
    results_directory = "./{datastore_directory_name}/{network_name}/{config_name}".format(datastore_directory_name=datastore_directory_name, network_name=network_name, config_name=greedy_mirror)
    with open(results_directory + "/" + "objectives_result_1.json", mode="r",encoding="utf-8") as openfile:

        json_object = json.load(openfile)

        objective_history_g = np.array(json_object["objective_history"])
                
        objective_1_list_g =np.array(objective_history_g[:, -2])
        objective_2_list_g = np.array(objective_history_g[:, -1])

        ######## TODO create normalized values
        combined_objecitve_1_values = np.concatenate(([objective_1_list_g[tick]], [objective_1_list[tick]],  pareto_front[:,1]))

        combined_objecitve_2_values = np.concatenate(([objective_2_list_g[tick]], [objective_2_list[tick]],  pareto_front[:,0]))

        normalize_factor_11 = (combined_objecitve_1_values.max() - combined_objecitve_1_values.min()) if (combined_objecitve_1_values.max() - combined_objecitve_1_values.min()) != 0 else combined_objecitve_1_values.max()
        normalize_factor_22 = (combined_objecitve_2_values.max() - combined_objecitve_2_values.min()) if (combined_objecitve_2_values.max() - combined_objecitve_2_values.min()) != 0 else combined_objecitve_2_values.max()
        
        ##### normalization with time-frame min and max values
        objective_1_value_g_normalized = (objective_1_list_g[tick] - combined_objecitve_1_values.min()) / normalize_factor_11
        objective_2_value_g_normalized = (objective_2_list_g[tick] - combined_objecitve_2_values.min()) / normalize_factor_22

        objective_1_value_normalized = (objective_1_list[tick] - combined_objecitve_1_values.min()) / normalize_factor_11
        objective_2_value_normalized = (objective_2_list[tick] - combined_objecitve_2_values.min()) / normalize_factor_22

        pareto_front_objecitve_1_normalized =  np.array((pareto_front[:,1] - combined_objecitve_1_values.min()) / normalize_factor_11)
        pareto_front_objecitve_2_normalized = np.array((pareto_front[:,0] - combined_objecitve_2_values.min()) / normalize_factor_22)

        greedy_hv = hv(np.array([objective_1_value_g_normalized, objective_2_value_g_normalized]))
        evo_hv = hv(np.array([objective_1_value_normalized, objective_2_value_normalized]))
        pareto_hv = hv(np.transpose([pareto_front_objecitve_1_normalized, pareto_front_objecitve_2_normalized]))
    
        print("greedy_hv: {greedy_hv}".format(greedy_hv=greedy_hv))
        print("evo_hv: {evo_hv}".format(evo_hv=evo_hv))
        print("pareto_hv: {pareto_hv}".format(pareto_hv=pareto_hv))
    
    fig1 = go.Figure()
    fig1.update_layout(
        title="pareto front for tick: {tick}".format(tick=tick),
        xaxis_title="PC",
        template=template_one_diagram_paretofront,
        # legend=None,
        showlegend=False,
        yaxis_title="-AVG_DL")

    fig1.add_trace(
    go.Scatter(
        x=np.array(pareto_front)[:,0],
        y=np.array(pareto_front)[:,1],
        mode='markers',
        name="{config_name} pareto".format(config_name=config_name),
        line={"color": evo_color}
        )
    )

    i = select_solution(np.array(pareto_front))
    fig1.add_trace(
    go.Scatter(
            x=np.array(pareto_front[i][0]),
            y=np.array(pareto_front[i][1]),
            mode='markers',
            name= "asf selection",
            opacity= 0.5,
            marker=dict(size=10)
            )
    )
    
    ### add actual simulation values
    fig1.add_trace(
        go.Scatter(
                x=[objective_2_list[tick]],
                y=[objective_1_list[tick]],
                mode='markers',
                name= "{config_name} value".format(config_name=config_name),
                opacity= 0.5,
                marker=dict(size=10)
                )
        )
    ### add greedy simulation values
    fig1.add_trace(
        go.Scatter(
                x=[objective_2_list_g[tick]],
                y=[objective_1_list_g[tick]],
                mode='markers',
                name= "{greedy_mirror} value".format(greedy_mirror=greedy_mirror),
                line={"color": greedy_color}
                )
        )
    
    fig0 = go.Figure()
    fig0.update_layout(
        title="normalized pareto front for tick: {tick}".format(tick=tick),
        xaxis_title="PC",
        yaxis_title="-AVG_DL",
        # legend=None,
        showlegend=False,
        template=template_one_diagram_paretofront
    )

    fig0.add_trace(
    go.Scatter(
        x=pareto_front_objecitve_2_normalized,
        y=pareto_front_objecitve_1_normalized,
        mode='markers',
        name="{config_name} pareto".format(config_name=config_name),
        line={"color": evo_color}
        )
    )

    i = select_solution(np.array(pareto_front))
    fig0.add_trace(
    go.Scatter(
            x=np.array(pareto_front_objecitve_2_normalized[i]),
            y=np.array(pareto_front_objecitve_1_normalized[i]),
            mode='markers',
            name= "asf selection",
            opacity= 0.5,
            marker=dict(size=10)
            )
    )
    
    ### add actual simulation values
    fig0.add_trace(
        go.Scatter(
                x=[objective_2_value_normalized],
                y=[objective_1_value_normalized],
                mode='markers',
                name= "{config_name} value".format(config_name=config_name),
                opacity= 0.5,
                marker=dict(size=10)
                )
        )
    ### add greedy simulation values
    fig0.add_trace(
        go.Scatter(
                x=[objective_2_value_g_normalized],
                y=[objective_1_value_g_normalized],
                mode='markers',
                name= "{greedy_mirror} value".format(greedy_mirror=greedy_mirror),
                line={"color": greedy_color}
                )
        )
    # fig0.add_trace(
    #     go.Scatter(
    #             x=[1.1],
    #             y=[1.1],
    #             mode='markers',
    #             name= "reference point",
    #             line={"color": greedy_color},
    #             )
    #     )
    if export_diagrams:
        diagram_path= "./diagrams/"
        if not os.path.exists(diagram_path):
            os.makedirs(diagram_path)
            
        fig1.write_image("diagrams/pareto_front_pertick.pdf")
        fig0.write_image("diagrams/pareto_front_pertick_normalized.pdf")
        remove_blank_page(diagram_path)
    
    if show_diagram:
        fig1.show()
        fig0.show()

##### create final boxplots
def create_result_boxplots(prefix: str, show_diagram: bool, export_diagrams: bool):
    fig = make_subplots(
        rows=3,
        cols=3,
        shared_yaxes=True,
        #subplot_titles=list(experiments[network][mp_name])
    )
    fig.update_layout(
            template=template_subplot_diagrams,
            showlegend=False,
        )
    
    fig.update_annotations(font= {
                'family': 'Courier New, monospace',
                'size': 8,
                'color': 'black'
            }
            )
    x_axis_list = ["30% moving UE's", "70% moving UE's", "100% moving UE's"]
    y_axis_list = ["150%", "100%", "50%"]
    title_list = []
    
    with open("./diagrams/avg_hypervolume_ratios.json", "r") as jsonFile:
        json_object = json.load(jsonFile)

        for _, network in enumerate(json_object):
            col_index = 1
            row_index = 1
            velocity = ""
            for _, config_name in enumerate(json_object[network]):
                if "var" in config_name or prefix not in config_name:
                    continue
                
                if f"{prefix}_C50" in config_name:
                    row_index = 3
                elif f"{prefix}_C100" in config_name:
                    row_index = 2
                elif f"{prefix}_C150" in config_name:
                    row_index = 1

                if "1,2ms" in config_name:
                    velocity= "1.2 m/s"
                elif "7ms" in config_name:
                    velocity= "7 m/s"
                elif "14ms" in config_name:
                    velocity= "14 m/s"
                
                if "MP30" in config_name:
                    col_index = 1
                elif "MP70" in config_name:
                    col_index = 2       
                elif "MP100" in config_name:
                    col_index = 3


                fig.add_trace(
                    go.Box(
                        y=json_object[network][config_name],
                        boxpoints='outliers',
                        marker_color=evo_color,
                        line_color=evo_color,
                        name=velocity,
                        line={"width": lineWidth}
                    ),
                    row=row_index,
                    col=col_index)
                
        for index, _ in enumerate(x_axis_list):
            fig.update_xaxes(row=3, col=index+1, title_text=x_axis_list[index])
            fig.update_yaxes(row=index+1, col=1, title_text=y_axis_list[index])
            fig.add_hline(row="all", col="all", line_width=lineWidth, y=1)
        
        if export_diagrams:
            fig.write_image(f"./diagrams/{prefix}_results.pdf")
        if show_diagram:
            fig.show()
        remove_blank_page(f"./diagrams")

    

def remove_blank_page(directory_path:str):

    pdf_files = glob.glob(os.path.join(directory_path, '*.pdf'))
    for pdf_file in pdf_files:
        infile = PdfReader(pdf_file)

        outfile = PdfWriter()
        pdfPage = infile.pages[0]
        outfile.add_page(pdfPage)

        with open(pdf_file, "wb") as f:
            outfile.write(f)

######### main script #######
experiments = {
    "het_C50": {
        "MP30": {
            "het_C50_MP30_MS1,2ms_evo": "het_C50_MP30_MS1,2ms_greedy",
            #"het_C50_MP30_MS1,2ms_evo_var": "het_C50_MP30_MS1,2ms_greedy",
            "het_C50_MP30_MS7ms_evo": "het_C50_MP30_MS7ms_greedy",
            "het_C50_MP30_MS14ms_evo": "het_C50_MP30_MS14ms_greedy",
            #"het_C50_MP30_MS14ms_evo_var": "het_C50_MP30_MS14ms_greedy"
        },
        "MP70": { 
            "het_C50_MP70_MS1,2ms_evo": "het_C50_MP70_MS1,2ms_greedy",
            #"het_C50_MP70_MS1,2ms_evo_var": "het_C50_MP70_MS1,2ms_greedy",
            "het_C50_MP70_MS7ms_evo": "het_C50_MP70_MS7ms_greedy",
            "het_C50_MP70_MS14ms_evo": "het_C50_MP70_MS14ms_greedy",
            #"het_C50_MP70_MS14ms_evo_var": "het_C50_MP70_MS14ms_greedy",
        },
    
    "MP100":{
        "het_C50_MP100_MS1,2ms_evo": "het_C50_MP100_MS1,2ms_greedy",
            #"het_C50_MP100_MS1,2ms_evo_var": "het_C50_MP100_MS1,2ms_greedy",
            "het_C50_MP100_MS7ms_evo": "het_C50_MP100_MS7ms_greedy",
            "het_C50_MP100_MS14ms_evo": "het_C50_MP100_MS14ms_greedy",
            #"het_C50_MP100_MS14ms_evo_var": "het_C50_MP100_MS14ms_greedy",
    },
    },
    "het_C100": {
        "MP30":{
            "het_C100_MP30_MS1,2ms_evo": "het_C100_MP30_MS1,2ms_greedy",
            #    "het_C100_MP30_MS1,2ms_evo_var": "het_C100_MP30_MS1,2ms_greedy",
            "het_C100_MP30_MS7ms_evo": "het_C100_MP30_MS7ms_greedy",
            "het_C100_MP30_MS14ms_evo": "het_C100_MP30_MS14ms_greedy",
            #    "het_C100_MP30_MS14ms_evo_var": "het_C100_MP30_MS14ms_greedy",
        },
        "MP70":{
            "het_C100_MP70_MS1,2ms_evo": "het_C100_MP70_MS1,2ms_greedy",
            #    "het_C100_MP70_MS1,2ms_evo_var": "het_C100_MP70_MS1,2ms_greedy",
            "het_C100_MP70_MS7ms_evo": "het_C100_MP70_MS7ms_greedy",
            "het_C100_MP70_MS14ms_evo": "het_C100_MP70_MS14ms_greedy",
            #    "het_C100_MP70_MS14ms_evo_var": "het_C100_MP70_MS14ms_greedy",
        },
        "MP100":{
            "het_C100_MP100_MS1,2ms_evo": "het_C100_MP100_MS1,2ms_greedy",
            #    "het_C100_MP100_MS1,2ms_evo_var": "het_C100_MP100_MS1,2ms_greedy",
            "het_C100_MP100_MS7ms_evo": "het_C100_MP100_MS7ms_greedy",
            "het_C100_MP100_MS14ms_evo": "het_C100_MP100_MS14ms_greedy",
            #    "het_C100_MP100_MS14ms_evo_var": "het_C100_MP100_MS14ms_greedy",
        },
    },
    "het_C150": {
        "MP30":{
            "het_C150_MP30_MS1,2ms_evo": "het_C150_MP30_MS1,2ms_greedy",
            #    "het_C150_MP30_MS1,2ms_evo_var": "het_C150_MP30_MS1,2ms_greedy",
            "het_C150_MP30_MS7ms_evo": "het_C150_MP30_MS7ms_greedy",
            "het_C150_MP30_MS14ms_evo": "het_C150_MP30_MS14ms_greedy",
            #    "het_C150_MP30_MS14ms_evo_var": "het_C150_MP30_MS14ms_greedy",
        },
    "MP70":{ 
            "het_C150_MP70_MS1,2ms_evo": "het_C150_MP70_MS1,2ms_greedy",
            #    "het_C150_MP70_MS1,2ms_evo_var": "het_C150_MP70_MS1,2ms_greedy",
            "het_C150_MP70_MS7ms_evo": "het_C150_MP70_MS7ms_greedy",
            "het_C150_MP70_MS14ms_evo": "het_C150_MP70_MS14ms_greedy",
            #    "het_C150_MP70_MS14ms_evo_var": "het_C150_MP70_MS14ms_greedy",
    },
    "MP100":{
            "het_C150_MP100_MS1,2ms_evo": "het_C150_MP100_MS1,2ms_greedy",
            #    "het_C150_MP100_MS1,2ms_evo_var": "het_C150_MP100_MS1,2ms_greedy",
            "het_C150_MP100_MS7ms_evo": "het_C150_MP100_MS7ms_greedy",
            "het_C150_MP100_MS14ms_evo": "het_C150_MP100_MS14ms_greedy",
            #    "het_C150_MP100_MS14ms_evo_var": "het_C150_MP100_MS14ms_greedy",
    },
    },
    "hom_C50": {
        "MP30":{
            "hom_C50_MP30_MS1,2ms_evo": "hom_C50_MP30_MS1,2ms_greedy",
            #    "hom_C50_MP30_MS1,2ms_evo_var": "hom_C50_MP30_MS1,2ms_greedy",
            "hom_C50_MP30_MS7ms_evo": "hom_C50_MP30_MS7ms_greedy",
            "hom_C50_MP30_MS14ms_evo": "hom_C50_MP30_MS14ms_greedy",
            #    "hom_C50_MP30_MS14ms_evo_var": "hom_C50_MP30_MS14ms_greedy",
        },
    "MP70":{
            "hom_C50_MP70_MS1,2ms_evo": "hom_C50_MP70_MS1,2ms_greedy",
            #    "hom_C50_MP70_MS1,2ms_evo_var": "hom_C50_MP70_MS1,2ms_greedy",
            "hom_C50_MP70_MS7ms_evo": "hom_C50_MP70_MS7ms_greedy",
            "hom_C50_MP70_MS14ms_evo": "hom_C50_MP70_MS14ms_greedy",
            #    "hom_C50_MP70_MS14ms_evo_var": "hom_C50_MP70_MS14ms_greedy",
    },
    "MP100":{
            "hom_C50_MP100_MS1,2ms_evo": "hom_C50_MP100_MS1,2ms_greedy",
            #    "hom_C50_MP100_MS1,2ms_evo_var": "hom_C50_MP100_MS1,2ms_greedy",
            "hom_C50_MP100_MS7ms_evo": "hom_C50_MP100_MS7ms_greedy",
            "hom_C50_MP100_MS14ms_evo": "hom_C50_MP100_MS14ms_greedy",
            #    "hom_C50_MP100_MS14ms_evo_var": "hom_C50_MP100_MS14ms_greedy",
    },
    },
    "hom_C100": {
        "MP30":{
            "hom_C100_MP30_MS1,2ms_evo": "hom_C100_MP30_MS1,2ms_greedy",
            #    "hom_C100_MP30_MS1,2ms_evo_var": "hom_C100_MP30_MS1,2ms_greedy",
            "hom_C100_MP30_MS7ms_evo": "hom_C100_MP30_MS7ms_greedy",
            "hom_C100_MP30_MS14ms_evo": "hom_C100_MP30_MS14ms_greedy",
            #    "hom_C100_MP30_MS14ms_evo_var": "hom_C100_MP30_MS14ms_greedy",
        },
    "MP70":{
            "hom_C100_MP70_MS1,2ms_evo": "hom_C100_MP70_MS1,2ms_greedy",
            #    "hom_C100_MP70_MS1,2ms_evo_var": "hom_C150_MP70_MS1,2ms_greedy",
            "hom_C100_MP70_MS7ms_evo": "hom_C100_MP70_MS7ms_greedy",
            "hom_C100_MP70_MS14ms_evo": "hom_C100_MP70_MS14ms_greedy",
            #    "hom_C100_MP70_MS14ms_evo_var": "hom_C100_MP70_MS14ms_greedy",
    },
    "MP100":{
            "hom_C100_MP100_MS1,2ms_evo": "hom_C100_MP100_MS1,2ms_greedy",
            #    "hom_C100_MP100_MS1,2ms_evo_var": "hom_C100_MP100_MS1,2ms_greedy",
            "hom_C100_MP100_MS7ms_evo": "hom_C100_MP100_MS7ms_greedy",
            "hom_C100_MP100_MS14ms_evo": "hom_C100_MP100_MS14ms_greedy",
            #    "hom_C100_MP100_MS14ms_evo_var": "hom_C100_MP100_MS14ms_greedy",
    }
    },
    "hom_C150": {
        "MP30":{
            "hom_C150_MP30_MS1,2ms_evo": "hom_C150_MP30_MS1,2ms_greedy",
            #    "hom_C150_MP30_MS1,2ms_evo_var": "hom_C150_MP30_MS1,2ms_greedy",
            "hom_C150_MP30_MS7ms_evo": "hom_C150_MP30_MS7ms_greedy",
            "hom_C150_MP30_MS14ms_evo": "hom_C150_MP30_MS14ms_greedy",
            #    "hom_C150_MP30_MS14ms_evo_var": "hom_C150_MP30_MS14ms_greedy",
        },
    "MP70":{
            "hom_C150_MP70_MS1,2ms_evo": "hom_C150_MP70_MS1,2ms_greedy",
            #    "hom_C150_MP70_MS1,2ms_evo_var": "hom_C150_MP70_MS1,2ms_greedy",
            "hom_C150_MP70_MS7ms_evo": "hom_C150_MP70_MS7ms_greedy",
            "hom_C150_MP70_MS14ms_evo": "hom_C150_MP70_MS14ms_greedy",
            #    "hom_C150_MP70_MS14ms_evo_var": "hom_C150_MP70_MS14ms_greedy",
    },
    "MP100":{      
            "hom_C150_MP100_MS1,2ms_evo": "hom_C150_MP100_MS1,2ms_greedy",
            #    "hom_C150_MP100_MS1,2ms_evo_var": "hom_C150_MP100_MS1,2ms_greedy",
            "hom_C150_MP100_MS7ms_evo": "hom_C150_MP100_MS7ms_greedy",
            "hom_C150_MP100_MS14ms_evo": "hom_C150_MP100_MS14ms_greedy",
            #    "hom_C150_MP100_MS14ms_evo_var": "hom_C150_MP100_MS14ms_greedy",
    }
    
    },
}

experimentsTest = {
    "het_C50": {
        "MP30": {
            "het_C50_MP30_MS1,2ms_evo": "het_C50_MP30_MS1,2ms_greedy",
            #"het_C50_MP30_MS1,2ms_evo_var": "het_C50_MP30_MS1,2ms_greedy",
            "het_C50_MP30_MS7ms_evo": "het_C50_MP30_MS7ms_greedy",
            "het_C50_MP30_MS14ms_evo": "het_C50_MP30_MS14ms_greedy",
            #"het_C50_MP30_MS14ms_evo_var": "het_C50_MP30_MS14ms_greedy"
            }
    }
}

## for creating the diagrams and hv-ratio file
for _, network in enumerate(experiments):

    for _, mp_name in enumerate(experiments[network]):
        subplots4x3 = make_subplots(
            rows=4,
            cols=3,
            shared_yaxes=True,
            subplot_titles=list(experiments[network][mp_name])
            )

        subplots4x3.update_layout(
            template=template_subplot_diagrams,
            showlegend=False,
        )
        subplots4x3.update_annotations(font= {
                'family': 'Courier New, monospace',
                'size': 8,
                'color': 'black'
            }
            )

        subplots4x3.update_yaxes(
        title_text="-AVG_DL", row=1, col=1
        )
        subplots4x3.update_yaxes(
        title_text="PC", row=2, col=1
        )
        subplots4x3.update_yaxes(
        title_text="HV", row=3, col=1
        )
        subplots4x3.update_yaxes(
        title_text="HV_ratio", row=4, col=1
        )
        subplots4x3.update_xaxes(
            title_text="ticks", row=4, col=1
        )
        subplots4x3.update_xaxes(
            title_text="ticks", row=4, col=2
        )
        subplots4x3.update_xaxes(
            title_text="ticks", row=4, col=3
        )
        for config_index, config_name in enumerate(experiments[network][mp_name]):
            create_objective_diagrams(
                network_name=network,
                baseline_name=experiments[network][mp_name][config_name],
                col_index=config_index+1,
                config_name=config_name,
                moving_portion=mp_name,
                show_diagram=False,
                export_diagrams=True

            )
            create_avrg_hypervolume_ratio_dataset(
                network_name=network, 
                config_name=config_name, 
                col_index=config_index+1,
                export_diagrams=True,
                moving_portion=mp_name,
                show_diagram=False,
                baseline_name=experiments[network][mp_name][config_name], 
                write_to_ratio_file=True)
            create_box_plots_averaged(
                network_name=network,
                config_name=config_name,
                baseline_name=experiments[network][mp_name][config_name],
                export_diagrams=True,
                show_diagram=False,
                moving_portion=mp_name,
            )
            remove_blank_page(f"./diagrams/{network}/{mp_name}/{config_name}")
        
        subplots4x3.write_image(f"./diagrams/{network}/{mp_name}/MS_compare_plot.pdf")
        remove_blank_page(f"./diagrams/{network}/{mp_name}")


create_result_boxplots("het", show_diagram=False, export_diagrams=True)
        
create_pareto_history_plots(export_diagrams=True, show_diagram=False)