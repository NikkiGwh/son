import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from son_main_script import Son
from matplotlib import pyplot as plt
from enum import Enum
from pymoo.decomposition.asf import ASF


class ObjectiveEnum(Enum):
    AVG_LOAD = "AVG_LOAD"
    AVG_SINR = "AVG_SINR"
    OVERLOAD = "OVERLOAD"
    AVG_POWER_CONSUMPTION = "AVG_POWER_CONSUMPTION"
    AVG_RSSI = "AVG_RSSI"
    AVG_DL_RATE = "AVG_DL_RATE"
    ENERGY_EFFICIENCY = "ENERGY_EFFICIENCY"


class AlgorithmEnum(Enum):
    NSGA2 = "NSGA2"
    NSGA3 = "NSGA3"


class CrossoverEnum(Enum):
    ONE_POINT_CROSSOVER = "ONE_POINT_CROSSOVER"
    TWO_POINT_CROSSOVERR = "TWO_POINT_CROSSOVERR"
    SBX_CROSSOVER = "SBX_CROSSOVER"


class MutationEnum(Enum):
    RANDOM_FLIP = "RANDOM_FLIP"
    SMALL_BS_FLIP = "SMALL_BS_FLIP"
    BIG_BS_FLIP = "BIG_BS_FLIP"
    PM_MUTATION = "PM_MUTATION"


class SamplingEnum(Enum):
    RANDOM_SAMPLING = "RANDOM_SAMPLING"
    SMALL_BS_FIRST_SAMPLING = "SMALL_BS_FIRST_SAMPLING"
    BIG_BS_FIRST_SAMPLING = "BIG_BS_FIRST_SAMPLING"
    HIGH_RSSI_FIRST_SAMPLING = "HIGH_RSSI_FIRST_SAMPLING"


class SonProblemElementWise(ElementwiseProblem):
    def __init__(self, obj_dict: list[str], son: Son):
        # prepare network
        self.son = son
        self.obj_dict = obj_dict
        n_var = len(list(filter(self.son.filter_user_nodes, self.son.graph.nodes.data())))

        # prepare problem parameter
        binary_activation_profile_encoding = self.son.get_edge_activation_encoding_from_graph()
        xu = np.array([])
        for _, cell_edges_encoded in enumerate(binary_activation_profile_encoding):
            xu = np.append(xu, len(cell_edges_encoded))

        # call super class constructor
        super().__init__(n_var=n_var, n_obj=len(obj_dict), xl=np.full_like(xu, 1), xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):

       # convert list[int] back to list[str] encoding
        x_binary_str_list = []
        for active_edge_cell_pos_index, active_edge_cell_pos in enumerate(x):
            encoding = ""
            for i in range(int(self.xu[active_edge_cell_pos_index])):
                encoding += "1" if i+1 == active_edge_cell_pos else "0"
            x_binary_str_list.append(encoding)

         # apply x (activation profile) to update network results

        self.son.apply_edge_activation_encoding_to_graph(x_binary_str_list)

        # prepare objectives
        objectives = np.array([])

        if ObjectiveEnum.AVG_LOAD.value in self.obj_dict:
            objectives = np.append(objectives, self.son.get_average_network_load())
        if ObjectiveEnum.OVERLOAD.value in self.obj_dict:
            objectives = np.append(objectives, self.son.get_avg_overlad())
        if ObjectiveEnum.AVG_POWER_CONSUMPTION.value in self.obj_dict:
            objectives = np.append(objectives, self.son.get_total_energy_consumption())
        if ObjectiveEnum.AVG_SINR.value in self.obj_dict:
            objectives = np.append(objectives, -self.son.get_average_sinr())
        if ObjectiveEnum.AVG_DL_RATE.value in self.obj_dict:
            objectives = np.append(objectives, -self.son.get_average_dl_datarate())
        if ObjectiveEnum.AVG_RSSI.value in self.obj_dict:
            objectives = np.append(objectives, -self.son.get_average_rssi())

        out["F"] = objectives


class SonSampling(Sampling):
    def _do(self, problem: SonProblemElementWise, n_samples, **kwargs):
        X = np.empty((n_samples, problem.n_var), int)

        for i in range(n_samples):
            for j in range(problem.n_var):
                X[i][j] = np.random.randint(problem.xl[j], problem.xu[j]+1)
        return X


class SonCrossover(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        super().__init__(n_parents=2, n_offsprings=1, prob=0.4)

    def _do(self, problem: SonProblemElementWise, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        # for each mating of n_parents=2 -> n_offsprings=1 offspring is produced
        n_parents, n_matings, n_var = X.shape

        # The output Y with the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full((self.n_offsprings, n_matings, n_var), None)

        # for each mating take required parents and do crossover
        for k in range(n_matings):
            # get the first and the second parent
            parent_a, parent_b = X[0, k], X[1, k]
            # take first half from parent_a and rest from parent_b
            off_a = np.append(parent_a[0:int(n_var/2)], parent_b[int(n_var/2):])
            Y[0, k] = off_a
        return Y


class SonMutation(Mutation):

    def __init__(self):
        super().__init__()

    def _do(self, problem: SonProblemElementWise, X, **kwargs):
        # for each individual
        for k, individual in enumerate(X):
            r = np.random.random()
            # with a probabilty of 30% switch on another random edge
            if r < 0.3:
                random_index = np.random.randint(0, problem.n_var)
                X[k][random_index] = np.random.randint(
                    problem.xl[random_index],
                    problem.xu[random_index] + 1)

        return X


class SonDublicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return np.array_equal(a, b)


################################ main ###################
# TODO add weigthing parameters for objectives with augumented scalarization function
# TODO add parameters for different termination criteria

def start_optimization(
        pop_size: int,
        n_offsprings: int,
        n_generations: int,
        termination: str,
        sampling: str,
        crossover: str,
        mutation: str,
        eliminate_duplicates: bool,
        objectives: list[str],
        algorithm: str,
        son_obj: Son,
        file_name: str):

    pymooAlgorithm = None
    samplingConfig = None
    mutationConfig = None
    crossoverConfig = None

    # sampling
    if (sampling == SamplingEnum.RANDOM_SAMPLING.value):
        samplingConfig = IntegerRandomSampling()
    else:
        samplingConfig = SonSampling()

    # crossover
    if (crossover == CrossoverEnum.SBX_CROSSOVER.value):
        crossoverConfig = SBX(prob=1.0, eta=3, vtype=float, repair=RoundingRepair())
    else:
        crossoverConfig = SonCrossover()

    # mutation
    if (mutation == MutationEnum.PM_MUTATION.value):
        mutationConfig = PM(prob=1.0, eta=3, vtype=float, repair=RoundingRepair())
    else:
        mutationConfig = SonMutation()

    # algorithm config
    if (algorithm == AlgorithmEnum.NSGA3.value):
        # create the reference directions to be used for the optimization
        ref_dirs = get_reference_directions("uniform", len(objectives), n_partitions=12)
        pymooAlgorithm = NSGA3(pop_size=pop_size,
                               sampling=samplingConfig,
                               crossover=crossoverConfig,
                               mutation=mutationConfig,
                               n_offsprings=n_offsprings,
                               eliminate_duplicates=eliminate_duplicates,
                               n_generations=n_generations,
                               ref_dirs=ref_dirs)
    else:
        pymooAlgorithm = NSGA2(pop_size=pop_size,
                               n_offsprings=n_offsprings,
                               sampling=samplingConfig,
                               crossover=crossoverConfig,
                               mutation=mutationConfig,
                               eliminate_duplicates=eliminate_duplicates,
                               n_generations=n_generations

                               )

    sonProblem = SonProblemElementWise(obj_dict=objectives, son=son_obj)
    # start computatoin with  termination criteria

    result = minimize(sonProblem, pymooAlgorithm,
                      termination=("n_gen", n_generations), seed=1, verbose=True)

    decisionSpace = result.X
    objectiveSpace = result.F
    print(objectiveSpace)
    print(decisionSpace)

    # convert encoding (decisionspace results) back to  binary encoding
    converted_encoding: list[list[str]] = []
    for _, x in enumerate(decisionSpace):
        x_binary_str_list: list[str] = []
        for active_edge_cell_pos_index, active_edge_cell_pos in enumerate(x):
            encoding = ""
            for i in range(int(sonProblem.xu[active_edge_cell_pos_index])):
                encoding += "1" if i+1 == active_edge_cell_pos else "0"
            x_binary_str_list.append(encoding)
        converted_encoding.append(x_binary_str_list)

    # save encoding to excel
    sonProblem.son.save_edge_activation_profile_to_file(
        converted_encoding, result_file_name=file_name + "_encoding.xlsx",
        result_sheet_name="encoding")
    # save all result individuums as json
    for i, individuum in enumerate(converted_encoding):
        sonProblem.son.apply_edge_activation_encoding_to_graph(individuum)
        sonProblem.son.save_json_adjacency_graph_to_file(
            filename=file_name + "_result_" + str(i + 1) + ".json")

# Augmented Scalarization Function (ASF)

# decomp = ASF()

# order or weights, len must match number of objectives => [load, overload, power_consumption, sinr]

# weights_1_0 = np.array([0.5, 0.5])
# weights_0_1 = np.array([0, 1])

# picks_list_1_0: list[list[int]] = []
# picks_list_0_1: list[list[int]] = []


# sonProblem = SonProblemElementWise(obj_dict=[ObjectiveEnum.AVG_LOAD, ObjectiveEnum.AVG_SINR])
# results = minimize(sonProblem, sonAlgorithm_NSGA2_1,
#                    termination=("n_gen", 300), seed=1, verbose=True)

# select one result for current hour and push in result array

# decisionSpace = results.X
# objectiveSpace = results.F
# print(objectiveSpace)
# print(decisionSpace)

# normalize objective space

# ideal = objectiveSpace.min(axis=0)
# nadir = objectiveSpace.max(axis=0)

# nF = (objectiveSpace - ideal) / (nadir - ideal)

# index_objectiveSpace_1_0 = decomp.do(nF, 1/weights_1_0).argmin()
# index_objectiveSpace_0_1 = decomp.do(nF, 1/weights_0_1).argmin()

# picks_list_1_0.append(decisionSpace[index_objectiveSpace_1_0])
# picks_list_0_1.append(decisionSpace[index_objectiveSpace_0_1])

# plt.figure(figsize=(7, 5))
# plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors='none', edgecolors='blue')
# plt.scatter(
#     nF[index_objectiveSpace_1_0, 0],
#     nF[index_objectiveSpace_1_0, 1],
#     marker="x", color="red", s=200)
# plt.title("Objective Space normalized")

# plt.figure(figsize=(7, 5))
# plt.scatter(objectiveSpace[:, 0], objectiveSpace[:, 1],
#             s=30, facecolors='none', edgecolors='blue')
# plt.scatter(
#     objectiveSpace[index_objectiveSpace_1_0, 0],
#     objectiveSpace[index_objectiveSpace_1_0, 1],
#     marker="x", color="red", s=200)
# plt.title("Objective Space")

# plt.show()


if __name__ == "__main__":
    son = Son(adjacencies_file_name="datastore/smallnetwork/smallnetwork_adjacencies.json",
              parameter_config_file_name="datastore/smallnetwork/smallnetwork_network_config.json")
    start_optimization(
        100, 20, 10, "", SamplingEnum.SMALL_BS_FIRST_SAMPLING.value, CrossoverEnum.ONE_POINT_CROSSOVER.value,
        MutationEnum.RANDOM_FLIP.value, True, [ObjectiveEnum.AVG_SINR.value, ObjectiveEnum.AVG_LOAD.value],
        AlgorithmEnum.NSGA2.value, son, "datastore/smallnetwork/algorithm_config_1/ind")
