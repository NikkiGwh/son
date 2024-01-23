#!/bin/bash
#
#SBATCH --job-name=nns
#SBATCH --output=log/Array_test.%A_%a.log
#SBATCH --partition=all
#SBATCH --nodes=1-5 --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-35
#SBATCH --cpus-per-task=4
# 1440

experimentName=sonNiklas
declare -a configNames

## thesis
# configNames[0]="./experiment_senarios_configs/het_C50/het_C50_MP30_MS1,2ms_evo.json"
# configNames[1]="./experiment_senarios_configs/het_C50/het_C50_MP30_MS1,2ms_greedy.json"
# configNames[2]="./experiment_senarios_configs/het_C50/het_C50_MP30_MS14ms_evo.json"
# configNames[3]="./experiment_senarios_configs/het_C50/het_C50_MP30_MS14ms_greedy.json"
configNames[0]="./experiment_senarios_configs/het_C50/het_C50_MP30_MS7ms_evo.json"
configNames[1]="./experiment_senarios_configs/het_C50/het_C50_MP30_MS7ms_greedy.json"

# configNames[4]="./experiment_senarios_configs/het_C50/het_C50_MP70_MS1,2ms_evo.json"
# configNames[5]="./experiment_senarios_configs/het_C50/het_C50_MP70_MS1,2ms_greedy.json"
# configNames[6]="./experiment_senarios_configs/het_C50/het_C50_MP70_MS14ms_evo.json"
# configNames[7]="./experiment_senarios_configs/het_C50/het_C50_MP70_MS14ms_greedy.json"
configNames[2]="./experiment_senarios_configs/het_C50/het_C50_MP70_MS7ms_evo.json"
configNames[3]="./experiment_senarios_configs/het_C50/het_C50_MP70_MS7ms_greedy.json"

# configNames[8]="./experiment_senarios_configs/het_C50/het_C50_MP100_MS1,2ms_evo.json"
# configNames[9]="./experiment_senarios_configs/het_C50/het_C50_MP100_MS1,2ms_greedy.json"
# configNames[10]="./experiment_senarios_configs/het_C50/het_C50_MP100_MS14ms_evo.json"
# configNames[11]="./experiment_senarios_configs/het_C50/het_C50_MP100_MS14ms_greedy.json"
configNames[4]="./experiment_senarios_configs/het_C50/het_C50_MP100_MS7ms_evo.json"
configNames[5]="./experiment_senarios_configs/het_C50/het_C50_MP100_MS7ms_greedy.json"

# configNames[12]="./experiment_senarios_configs/het_C100/het_C100_MP30_MS1,2ms_evo.json"
# configNames[13]="./experiment_senarios_configs/het_C100/het_C100_MP30_MS1,2ms_greedy.json"
# configNames[14]="./experiment_senarios_configs/het_C100/het_C100_MP30_MS14ms_evo.json"
# configNames[15]="./experiment_senarios_configs/het_C100/het_C100_MP30_MS14ms_greedy.json"
configNames[6]="./experiment_senarios_configs/het_C100/het_C100_MP30_MS7ms_evo.json"
configNames[7]="./experiment_senarios_configs/het_C100/het_C100_MP30_MS7ms_greedy.json"

# configNames[16]="./experiment_senarios_configs/het_C100/het_C100_MP70_MS1,2ms_evo.json"
# configNames[17]="./experiment_senarios_configs/het_C100/het_C100_MP70_MS1,2ms_greedy.json"
# configNames[18]="./experiment_senarios_configs/het_C100/het_C100_MP70_MS14ms_evo.json"
# configNames[19]="./experiment_senarios_configs/het_C100/het_C100_MP70_MS14ms_greedy.json"
configNames[8]="./experiment_senarios_configs/het_C100/het_C100_MP70_MS7ms_evo.json"
configNames[9]="./experiment_senarios_configs/het_C100/het_C100_MP70_MS7ms_greedy.json"

# configNames[20]="./experiment_senarios_configs/het_C100/het_C100_MP100_MS1,2ms_evo.json"
# configNames[21]="./experiment_senarios_configs/het_C100/het_C100_MP100_MS1,2ms_greedy.json"
# configNames[22]="./experiment_senarios_configs/het_C100/het_C100_MP100_MS14ms_evo.json"
# configNames[23]="./experiment_senarios_configs/het_C100/het_C100_MP100_MS14ms_greedy.json"
configNames[10]="./experiment_senarios_configs/het_C100/het_C100_MP100_MS7ms_evo.json"
configNames[11]="./experiment_senarios_configs/het_C100/het_C100_MP100_MS7ms_greedy.json"

# configNames[24]="./experiment_senarios_configs/het_C150/het_C150_MP30_MS1,2ms_evo.json"
# configNames[25]="./experiment_senarios_configs/het_C150/het_C150_MP30_MS1,2ms_greedy.json"
# configNames[26]="./experiment_senarios_configs/het_C150/het_C150_MP30_MS14ms_evo.json"
# configNames[27]="./experiment_senarios_configs/het_C150/het_C150_MP30_MS14ms_greedy.json"
configNames[12]="./experiment_senarios_configs/het_C150/het_C150_MP30_MS7ms_evo.json"
configNames[13]="./experiment_senarios_configs/het_C150/het_C150_MP30_MS7ms_greedy.json"

# configNames[28]="./experiment_senarios_configs/het_C150/het_C150_MP70_MS1,2ms_evo.json"
# configNames[29]="./experiment_senarios_configs/het_C150/het_C150_MP70_MS1,2ms_greedy.json"
# configNames[30]="./experiment_senarios_configs/het_C150/het_C150_MP70_MS14ms_evo.json"
# configNames[31]="./experiment_senarios_configs/het_C150/het_C150_MP70_MS14ms_greedy.json"
configNames[14]="./experiment_senarios_configs/het_C150/het_C150_MP70_MS7ms_evo.json"
configNames[15]="./experiment_senarios_configs/het_C150/het_C150_MP70_MS7ms_greedy.json"

# configNames[32]="./experiment_senarios_configs/het_C150/het_C150_MP100_MS1,2ms_evo.json"
# configNames[33]="./experiment_senarios_configs/het_C150/het_C150_MP100_MS1,2ms_greedy.json"
# configNames[34]="./experiment_senarios_configs/het_C150/het_C150_MP100_MS14ms_evo.json"
# configNames[35]="./experiment_senarios_configs/het_C150/het_C150_MP100_MS14ms_greedy.json"
configNames[16]="./experiment_senarios_configs/het_C150/het_C150_MP100_MS7ms_evo.json"
configNames[17]="./experiment_senarios_configs/het_C150/het_C150_MP100_MS7ms_greedy.json"

# configNames[36]="./experiment_senarios_configs/hom_C50/hom_C50_MP30_MS1,2ms_evo.json"
# configNames[37]="./experiment_senarios_configs/hom_C50/hom_C50_MP30_MS1,2ms_greedy.json"
# configNames[38]="./experiment_senarios_configs/hom_C50/hom_C50_MP30_MS14ms_evo.json"
# configNames[39]="./experiment_senarios_configs/hom_C50/hom_C50_MP30_MS14ms_greedy.json"
configNames[18]="./experiment_senarios_configs/hom_C50/hom_C50_MP30_MS7ms_evo.json"
configNames[19]="./experiment_senarios_configs/hom_C50/hom_C50_MP30_MS7ms_greedy.json"

# configNames[40]="./experiment_senarios_configs/hom_C50/hom_C50_MP70_MS1,2ms_evo.json"
# configNames[41]="./experiment_senarios_configs/hom_C50/hom_C50_MP70_MS1,2ms_greedy.json"
# configNames[42]="./experiment_senarios_configs/hom_C50/hom_C50_MP70_MS14ms_evo.json"
# configNames[43]="./experiment_senarios_configs/hom_C50/hom_C50_MP70_MS14ms_greedy.json"
configNames[20]="./experiment_senarios_configs/hom_C50/hom_C50_MP70_MS7ms_evo.json"
configNames[21]="./experiment_senarios_configs/hom_C50/hom_C50_MP70_MS7ms_greedy.json"

# configNames[44]="./experiment_senarios_configs/hom_C50/hom_C50_MP100_MS1,2ms_evo.json"
# configNames[45]="./experiment_senarios_configs/hom_C50/hom_C50_MP100_MS1,2ms_greedy.json"
# configNames[46]="./experiment_senarios_configs/hom_C50/hom_C50_MP100_MS14ms_evo.json"
# configNames[47]="./experiment_senarios_configs/hom_C50/hom_C50_MP100_MS14ms_greedy.json"
configNames[22]="./experiment_senarios_configs/hom_C50/hom_C50_MP100_MS7ms_evo.json"
configNames[23]="./experiment_senarios_configs/hom_C50/hom_C50_MP100_MS7ms_greedy.json"


# configNames[48]="./experiment_senarios_configs/hom_C100/hom_C100_MP30_MS1,2ms_evo.json"
# configNames[49]="./experiment_senarios_configs/hom_C100/hom_C100_MP30_MS1,2ms_greedy.json"
# configNames[50]="./experiment_senarios_configs/hom_C100/hom_C100_MP30_MS14ms_evo.json"
# configNames[51]="./experiment_senarios_configs/hom_C100/hom_C100_MP30_MS14ms_greedy.json"
configNames[24]="./experiment_senarios_configs/hom_C100/hom_C100_MP30_MS7ms_evo.json"
configNames[25]="./experiment_senarios_configs/hom_C100/hom_C100_MP30_MS7ms_greedy.json"

# configNames[52]="./experiment_senarios_configs/hom_C100/hom_C100_MP70_MS1,2ms_evo.json"
# configNames[53]="./experiment_senarios_configs/hom_C100/hom_C100_MP70_MS1,2ms_greedy.json"
# configNames[54]="./experiment_senarios_configs/hom_C100/hom_C100_MP70_MS14ms_evo.json"
# configNames[55]="./experiment_senarios_configs/hom_C100/hom_C100_MP70_MS14ms_greedy.json"
configNames[26]="./experiment_senarios_configs/hom_C100/hom_C100_MP70_MS7ms_evo.json"
configNames[27]="./experiment_senarios_configs/hom_C100/hom_C100_MP70_MS7ms_greedy.json"

# configNames[56]="./experiment_senarios_configs/hom_C100/hom_C100_MP100_MS1,2ms_evo.json"
# configNames[57]="./experiment_senarios_configs/hom_C100/hom_C100_MP100_MS1,2ms_greedy.json"
# configNames[58]="./experiment_senarios_configs/hom_C100/hom_C100_MP100_MS14ms_evo.json"
# configNames[59]="./experiment_senarios_configs/hom_C100/hom_C100_MP100_MS14ms_greedy.json"
configNames[28]="./experiment_senarios_configs/hom_C100/hom_C100_MP100_MS7ms_evo.json"
configNames[29]="./experiment_senarios_configs/hom_C100/hom_C100_MP100_MS7ms_greedy.json"

# configNames[60]="./experiment_senarios_configs/hom_C150/hom_C150_MP30_MS1,2ms_evo.json"
# configNames[61]="./experiment_senarios_configs/hom_C150/hom_C150_MP30_MS1,2ms_greedy.json"
# configNames[62]="./experiment_senarios_configs/hom_C150/hom_C150_MP30_MS14ms_evo.json"
# configNames[63]="./experiment_senarios_configs/hom_C150/hom_C150_MP30_MS14ms_greedy.json"
configNames[30]="./experiment_senarios_configs/hom_C150/hom_C150_MP30_MS7ms_evo.json"
configNames[31]="./experiment_senarios_configs/hom_C150/hom_C150_MP30_MS7ms_greedy.json"

# configNames[64]="./experiment_senarios_configs/hom_C150/hom_C150_MP70_MS1,2ms_evo.json"
# configNames[65]="./experiment_senarios_configs/hom_C150/hom_C150_MP70_MS1,2ms_greedy.json"
# configNames[66]="./experiment_senarios_configs/hom_C150/hom_C150_MP70_MS14ms_evo.json"
# configNames[67]="./experiment_senarios_configs/hom_C150/hom_C150_MP70_MS14ms_greedy.json"
configNames[32]="./experiment_senarios_configs/hom_C150/hom_C150_MP70_MS7ms_evo.json"
configNames[33]="./experiment_senarios_configs/hom_C150/hom_C150_MP70_MS7ms_greedy.json"

# configNames[68]="./experiment_senarios_configs/hom_C150/hom_C150_MP100_MS1,2ms_evo.json"
# configNames[69]="./experiment_senarios_configs/hom_C150/hom_C150_MP100_MS1,2ms_greedy.json"
# configNames[70]="./experiment_senarios_configs/hom_C150/hom_C150_MP100_MS14ms_evo.json"
# configNames[71]="./experiment_senarios_configs/hom_C150/hom_C150_MP100_MS14ms_greedy.json"
configNames[34]="./experiment_senarios_configs/hom_C150/hom_C150_MP100_MS7ms_evo.json"
configNames[35]="./experiment_senarios_configs/hom_C150/hom_C150_MP100_MS7ms_greedy.json"

### paper
# configNames[0]="./experiment_paper_configs/het_C150_MP100_MS14ms_evo_var.json"



srun hostname
srun nproc


echo ${configNames[${SLURM_ARRAY_TASK_ID}]} started
echo ${SLURM_ARRAY_TASK_ID}

## paper


# srun apptainer exec nns.sif python network_simulation_script.py het_C50 ${configNames[${SLURM_ARRAY_TASK_ID}]}
# echo apptainer exec nns.sif python network_simulation_script.py het_C50 ${configNames[${SLURM_ARRAY_TASK_ID}]}



## all
if [ $SLURM_ARRAY_TASK_ID -le 5 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py het_C50 ${configNames[${SLURM_ARRAY_TASK_ID}]}
    echo apptainer exec nns.sif python network_simulation_script.py het_C50 ${configNames[${SLURM_ARRAY_TASK_ID}]}
elif [ $SLURM_ARRAY_TASK_ID -le 11 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py het_C100 ${configNames[${SLURM_ARRAY_TASK_ID}]}
    echo apptainer exec nns.sif python network_simulation_script.py het_C100 ${configNames[${SLURM_ARRAY_TASK_ID}]}
elif [ $SLURM_ARRAY_TASK_ID -le 17 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py het_C150 ${configNames[${SLURM_ARRAY_TASK_ID}]}
    echo apptainer exec nns.sif python network_simulation_script.py het_C150 ${configNames[${SLURM_ARRAY_TASK_ID}]}
elif [ $SLURM_ARRAY_TASK_ID -le 23 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py hom_C50 ${configNames[${SLURM_ARRAY_TASK_ID}]}
    echo apptainer exec nns.sif python network_simulation_script.py hom_C50 ${configNames[${SLURM_ARRAY_TASK_ID}]}
elif [ $SLURM_ARRAY_TASK_ID -le 29 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py hom_C100 ${configNames[${SLURM_ARRAY_TASK_ID}]}
    echo apptainer exec nns.sif python network_simulation_script.py hom_C100 ${configNames[${SLURM_ARRAY_TASK_ID}]}
elif [ $SLURM_ARRAY_TASK_ID -le 35 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py hom_C150 ${configNames[${SLURM_ARRAY_TASK_ID}]}
    echo apptainer exec nns.sif python network_simulation_script.py hom_C150 ${configNames[${SLURM_ARRAY_TASK_ID}]}
else
echo ${SLURM_ARRAY_TASK_ID}
fi

echo ${configNames[${SLURM_ARRAY_TASK_ID}]} finished