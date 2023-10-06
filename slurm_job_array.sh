#!/bin/bash
#
#SBATCH --job-name=naturalsi
#SBATCH --output=logs/Array_test.%A_%a.log
#SBATCH --partition=ci
#SBATCH --nodes=1-5 --ntasks=1
#SBATCH --mem-per-cpu=2500
#SBATCH --array=0-71
#SBATCH --nodelist=ant6,ant7
#SBATCH --exclude=ant1,ant10

# 1440

# . /data/spack/share/spack/setup-env.sh
# spack load openjdk@11.0.2%gcc@8.3.1 arch=linux-centos8-x86_64

experimentName=exISG
declare -a combinations
index=0

b=0

# runs

for k in 2 3; do
  for ms in 10 15 20 24 30; do
    for obs in NO CH LA; do
      for ele in 1 2 3 M; do

        for alg in DE; do
          for isg in YENSK RANDOM; do
            for back in "-b"; do
              combinations[$index]="$k $ms $obs $ele $alg $dm $back"
              index=$((index + 1))
            done
          done
        done
      done
    done
  done

done
parameters=(${combinations[${SLURM_ARRAY_TASK_ID}]})

k=${parameters[0]}
ms=${parameters[1]}
obs=${parameters[2]}
ele=${parameters[3]}
alg=${parameters[4]}
dm=${parameters[5]}
back=${parameters[6]}

/bin/hostname
echo ${experimentName}FRD${alg}${dm}${k}${obs}${ms}${ele}${back}EXSTARTED
java -jar morp-benchmark-suite-1.0.0-SNAPSHOT.one-jar.jar -ex $experimentName -kn $k -ms $ms -ele $ele -o $obs ${back} -p 100 -g 200 -n NSGAIIDMS${alg}${dm} -pnm ${alg} -dm ${dm} -df EUCLID -inter -ir 31
echo ${experimentName}FRD${alg}${dm}${k}${obs}${ms}${ele}${back}EXENDED
/bin/hostname
