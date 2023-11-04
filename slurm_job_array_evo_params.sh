#!/bin/bash
#
#SBATCH --job-name=nns
#SBATCH --output=log/Array_test.%A_%a.log
#SBATCH --partition=all
#SBATCH --nodes=1-5 --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
# 1440

# . /data/spack/share/spack/setup-env.sh
# spack load openjdk@11.0.2%gcc@8.3.1 arch=linux-centos8-x86_64

experimentName=sonNiklas
# declare -a combinations
declare -a configNames

configNames[0]="./experiment_evo_params_configs/het_C100_MP70_MS1,2ms_09mut_03mutvar_sonrand.json"
configNames[1]="./experiment_evo_params_configs/het_C100_MP70_MS1,2ms_09mut_06mutvar_sonrand.json"
configNames[2]="./experiment_evo_params_configs/het_C100_MP70_MS1,2ms_09mut_09mutvar_sonrand.json"
configNames[3]="./experiment_evo_params_configs/het_C100_MP70_MS1,2ms_09mut_defaultmutvar_sonrand.json"


configNames[4]="./experiment_evo_params_configs/het_C100_MP70_MS1,2ms_09mut_03mutvar.json"
configNames[5]="./experiment_evo_params_configs/het_C100_MP70_MS1,2ms_09mut_06mutvar.json"
configNames[6]="./experiment_evo_params_configs/het_C100_MP70_MS1,2ms_09mut_09mutvar.json"
configNames[7]="./experiment_evo_params_configs/het_C100_MP70_MS1,2ms_09mut_defaultmutvar.json"

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



srun apptainer exec nns.sif python network_simulation_script.py het_C100 ${configNames[${SLURM_ARRAY_TASK_ID}]}

echo ${configNames[${SLURM_ARRAY_TASK_ID}]} finished