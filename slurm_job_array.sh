#!/bin/bash
#
#SBATCH --job-name=nns
#SBATCH --output=log/Array_test.%A_%a.log
#SBATCH --partition=all
#SBATCH --nodes=1-5 --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-47
#SBATCH --cpus-per-task=4
# 1440

# . /data/spack/share/spack/setup-env.sh
# spack load openjdk@11.0.2%gcc@8.3.1 arch=linux-centos8-x86_64

experimentName=sonNiklas
# declare -a combinations
declare -a configNames

configNames[0]="./experiment_senarios_configs/het_C50/het_C50_MP30_MS1,2ms_evo.json"
configNames[1]="./experiment_senarios_configs/het_C50/het_C50_MP30_MS1,2ms_greedy.json"
configNames[2]="./experiment_senarios_configs/het_C50/het_C50_MP30_MS14ms_evo.json"
configNames[3]="./experiment_senarios_configs/het_C50/het_C50_MP30_MS14ms_greedy.json"
configNames[4]="./experiment_senarios_configs/het_C50/het_C50_MP70_MS1,2ms_evo.json"
configNames[5]="./experiment_senarios_configs/het_C50/het_C50_MP70_MS1,2ms_greedy.json"
configNames[6]="./experiment_senarios_configs/het_C50/het_C50_MP70_MS14ms_evo.json"
configNames[7]="./experiment_senarios_configs/het_C50/het_C50_MP70_MS14ms_greedy.json"


configNames[8]="./experiment_senarios_configs/het_C100/het_C100_MP30_MS1,2ms_evo.json"
configNames[9]="./experiment_senarios_configs/het_C100/het_C100_MP30_MS1,2ms_greedy.json"
configNames[10]="./experiment_senarios_configs/het_C100/het_C100_MP30_MS14ms_evo.json"
configNames[11]="./experiment_senarios_configs/het_C100/het_C100_MP30_MS14ms_greedy.json"
configNames[12]="./experiment_senarios_configs/het_C100/het_C100_MP70_MS1,2ms_evo.json"
configNames[13]="./experiment_configs/het_C100/het_C100_MP70_MS1,2ms_greedy.json"
configNames[14]="./experiment_senarios_configs/het_C100/het_C100_MP70_MS14ms_evo.json"
configNames[15]="./experiment_senarios_configs/het_C100/het_C100_MP70_MS14ms_greedy.json"


configNames[16]="./experiment_senarios_configs/het_C200/het_C200_MP30_MS1,2ms_evo.json"
configNames[17]="./experiment_senarios_configs/het_C200/het_C200_MP30_MS1,2ms_greedy.json"
configNames[18]="./experiment_senarios_configs/het_C200/het_C200_MP30_MS14ms_evo.json"
configNames[19]="./experiment_senarios_configs/het_C200/het_C200_MP30_MS14ms_greedy.json"
configNames[20]="./experiment_senarios_configs/het_C200/het_C200_MP70_MS1,2ms_evo.json"
configNames[21]="./experiment_senarios_configs/het_C200/het_C200_MP70_MS1,2ms_greedy.json"
configNames[22]="./experiment_senarios_configs/het_C200/het_C200_MP70_MS14ms_evo.json"
configNames[23]="./experiment_senarios_configs/het_C200/het_C200_MP70_MS14ms_greedy.json"


configNames[24]="./experiment_senarios_configs/hom_C50/hom_C50_MP30_MS1,2ms_evo.json"
configNames[25]="./experiment_senarios_configs/hom_C50/hom_C50_MP30_MS1,2ms_greedy.json"
configNames[26]="./experiment_senarios_configs/hom_C50/hom_C50_MP30_MS14ms_evo.json"
configNames[27]="./experiment_senarios_configs/hom_C50/hom_C50_MP30_MS14ms_greedy.json"
configNames[28]="./experiment_senarios_configs/hom_C50/hom_C50_MP70_MS1,2ms_evo.json"
configNames[29]="./experiment_senarios_configs/hom_C50/hom_C50_MP70_MS1,2ms_greedy.json"
configNames[30]="./experiment_senarios_configs/hom_C50/hom_C50_MP70_MS14ms_evo.json"
configNames[31]="./experiment_senarios_configs/hom_C50/hom_C50_MP70_MS14ms_greedy.json"


configNames[32]="./experiment_senarios_configs/hom_C100/hom_C100_MP30_MS1,2ms_evo.json"
configNames[33]="./experiment_senarios_configs/hom_C100/hom_C100_MP30_MS1,2ms_greedy.json"
configNames[34]="./experiment_senarios_configs/hom_C100/hom_C100_MP30_MS14ms_evo.json"
configNames[35]="./experiment_senarios_configs/hom_C100/hom_C100_MP30_MS14ms_greedy.json"
configNames[36]="./experiment_senarios_configs/hom_C100/hom_C100_MP70_MS1,2ms_evo.json"
configNames[37]="./experiment_senarios_configs/hom_C100/hom_C100_MP70_MS1,2ms_greedy.json"
configNames[38]="./experiment_senarios_configs/hom_C100/hom_C100_MP70_MS14ms_evo.json"
configNames[39]="./experiment_senarios_configs/hom_C100/hom_C100_MP70_MS14ms_greedy.json"


configNames[40]="./experiment_senarios_configs/hom_C200/hom_C200_MP30_MS1,2ms_evo.json"
configNames[41]="./experiment_senarios_configs/hom_C200/hom_C200_MP30_MS1,2ms_greedy.json"
configNames[42]="./experiment_senarios_configs/hom_C200/hom_C200_MP30_MS14ms_evo.json"
configNames[43]="./experiment_senarios_configs/hom_C200/hom_C200_MP30_MS14ms_greedy.json"
configNames[44]="./experiment_senarios_configs/hom_C200/hom_C200_MP70_MS1,2ms_evo.json"
configNames[45]="./experiment_senarios_configs/hom_C200/hom_C200_MP70_MS1,2ms_greedy.json"
configNames[46]="./experiment_senarios_configs/hom_C200/hom_C200_MP70_MS14ms_evo.json"
configNames[47]="./experiment_senarios_configs/hom_C200/hom_C200_MP70_MS14ms_greedy.json"

# index=0
# b=0

# runs

# for k in 2 3; do
#   for ms in 10 15 20 24 30; do
#     for obs in NO CH LA; do
#       for ele in 1 2 3 M; do

#         for alg in DE; do
#           for isg in YENSK RANDOM; do
#             for back in "-b"; do
#               combinations[$index]="$k $ms $obs $ele $alg $dm $back"
#               index=$((index + 1))
#             done
#           done
#         done
#       done
#     done
#   done

# done
# parameters=(${combinations[${SLURM_ARRAY_TASK_ID}]})

# k=${parameters[0]}
# ms=${parameters[1]}
# obs=${parameters[2]}
# ele=${parameters[3]}
# alg=${parameters[4]}
# dm=${parameters[5]}
# back=${parameters[6]}


srun hostname
srun nproc
# echo ${experimentName}FRD${alg}${dm}${k}${obs}${ms}${ele}${back}EXSTARTED

echo ${configNames[${SLURM_ARRAY_TASK_ID}]} started
# java -jar morp-benchmark-suite-1.0.0-SNAPSHOT.one-jar.jar -ex $experimentName -kn $k -ms $ms -ele $ele -o $obs ${back} -p 100 -g 200 -n NSGAIIDMS${alg}${dm} -pnm ${alg} -dm ${dm} -df EUCLID -inter -ir 31

if [ $SLURM_ARRAY_TASK_ID -le 7 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py het_C50 ${configNames[${SLURM_ARRAY_TASK_ID}]}
elif [ $SLURM_ARRAY_TASK_ID -le 15 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py het_C100 ${configNames[${SLURM_ARRAY_TASK_ID}]}
elif [ $SLURM_ARRAY_TASK_ID -le 23 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py het_C200 ${configNames[${SLURM_ARRAY_TASK_ID}]}
elif [ $SLURM_ARRAY_TASK_ID -le 31 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py hom_C50 ${configNames[${SLURM_ARRAY_TASK_ID}]}
elif [ $SLURM_ARRAY_TASK_ID -le 39 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py hom_C100 ${configNames[${SLURM_ARRAY_TASK_ID}]}
elif [ $SLURM_ARRAY_TASK_ID -le 47 ]
then
    srun apptainer exec nns.sif python network_simulation_script.py hom_C200 ${configNames[${SLURM_ARRAY_TASK_ID}]}
else
echo ${SLURM_ARRAY_TASK_ID}
fi



echo ${configNames[${SLURM_ARRAY_TASK_ID}]} finished