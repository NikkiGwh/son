import json
import multiprocessing
from pdb import run
from typing_extensions import runtime
import numpy as np
from pandas import MultiIndex
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from son_main_script import Son
from enum import Enum
from pymoo.decomposition.asf import ASF
from pymoo.core.callback import Callback
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.pntx import SinglePointCrossover, TwoPointCrossover

from pymoo.termination import get_termination
import networkx as nx


class ObjectiveEnum(Enum):
    AVG_LOAD = "AVG_LOAD"
    AVG_SINR = "AVG_SINR"
    OVERLOAD = "OVERLOAD"
    POWER_CONSUMPTION = "POWER_CONSUMPTION"
    AVG_RSSI = "AVG_RSSI"
    AVG_DL_RATE = "AVG_DL_RATE"
    ENERGY_EFFICIENCY = "ENERGY_EFFICIENCY"


class RunningMode(Enum):
    STATIC = "STATIC"
    LIVE = "LIVE"


class AlgorithmEnum(Enum):
    NSGA2 = "NSGA2"
    NSGA3 = "NSGA3"


class CrossoverEnum(Enum):
    ONE_POINT_CROSSOVER = "ONE_POINT_CROSSOVER"
    UNIFORM_CROSSOVER = "UNIFORM_CROSSOVER"
    SBX_CROSSOVER = "SBX_CROSSOVER"
    SON_CROSSOVER = "SON_CROSSOVER"


class MutationEnum(Enum):
    RANDOM_FLIP = "RANDOM_FLIP"
    BIT_FLIP = "BIT_FLIP"
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
        # prepara flags
        # self.users_changed_index_list: list[int] = []
        # prepare network
        self.son_original = son
        self.obj_dict = obj_dict
        n_var = len(
            list(filter(self.son_original.filter_user_nodes, self.son_original.graph.nodes.data())))

        # prepare problem parameter
        max_activation_values = self.son_original.get_edge_activation_encoding_from_graph()
        xu = np.array([])
        for _, max_value in enumerate(max_activation_values):
            xu = np.append(xu, max_value)

        # call super class constructor
        super().__init__(n_var=n_var, n_obj=len(obj_dict), xl=np.full_like(xu, 1), xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        self.son_original.apply_edge_activation_encoding_to_graph(x)
        # prepare objectives
        objectives = np.array([])

        if ObjectiveEnum.AVG_LOAD.value in self.obj_dict:
            objectives = np.append(objectives, self.son_original.get_average_network_load())
        if ObjectiveEnum.OVERLOAD.value in self.obj_dict:
            objectives = np.append(objectives, self.son_original.get_avg_overlad())
        if ObjectiveEnum.POWER_CONSUMPTION.value in self.obj_dict:
            objectives = np.append(objectives, self.son_original.get_total_energy_consumption())
        if ObjectiveEnum.ENERGY_EFFICIENCY.value in self.obj_dict:
            objectives = np.append(objectives, -self.son_original.get_energy_efficiency())
        if ObjectiveEnum.AVG_SINR.value in self.obj_dict:
            objectives = np.append(objectives, -self.son_original.get_average_sinr())
        if ObjectiveEnum.AVG_DL_RATE.value in self.obj_dict:
            objectives = np.append(objectives, -self.son_original.get_average_dl_datarate())
        if ObjectiveEnum.AVG_RSSI.value in self.obj_dict:
            objectives = np.append(objectives, -self.son_original.get_average_rssi())

        out["F"] = np.array(objectives)


class SonSampling(Sampling):
    def _do(self, problem: SonProblemElementWise, n_samples, **kwargs):
        X = np.empty((n_samples, problem.n_var), int)

        for i in range(n_samples):
            for j in range(problem.n_var):
                X[i][j] = np.random.randint(problem.xl[j], problem.xu[j]+1)
        return X


class SeedSampling(Sampling):
    def __init__(self, seed_pop) -> None:
        self.seed_pop = seed_pop
        super().__init__()

    def _do(self, problem: SonProblemElementWise, n_samples, **kwargs):
        X = np.empty((n_samples, problem.n_var), int)
        for i in range(n_samples):

            for j in range(problem.n_var):
                if self.seed_pop[i][j] > problem.xu[j] or self.seed_pop[i][j] < problem.xl[j]:
                    X[i][j] = np.random.randint(problem.xl[j], problem.xu[j]+1)
                else:
                    X[i][j] = self.seed_pop[i][j]
        return X


class SeedSampling2(Sampling):
    def __init__(self, seed_pop) -> None:
        self.seed_pop = seed_pop
        super().__init__()

    def _do(self, problem: SonProblemElementWise, n_samples, **kwargs):
        return self.seed_pop


class SonCrossover(Crossover):
    def __init__(self, **kwargs):

        # define the crossover: number of parents and number of offsprings
        super().__init__(n_parents=2, n_offsprings=1, **kwargs)

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


class MyCallback(Callback):

    def __init__(
            self, pymoo_message_queue: multiprocessing.Queue,
            editor_message_queue: multiprocessing.Queue, graph: nx.Graph, running_mode: str) -> None:
        super().__init__()

        self.data["external_termination"] = False
        self.data["external_reset"] = False
        self.data["graph"] = graph
        self.pymoo_message_queue = pymoo_message_queue
        self.editor_message_queue = editor_message_queue
        self.running_mode = running_mode
        self.n_gen_since_last_fetch = 0

    def notify(self, algorithm: GeneticAlgorithm):
        queue_filled = False
        if self.running_mode == RunningMode.LIVE.value:
            self.n_gen_since_last_fetch += 1
            # read editor message queue
            while self.editor_message_queue.empty() is False:
                callback_obj = self.editor_message_queue.get()

                if callback_obj["terminate"] == True:
                    queue_filled = True
                    self.data["external_termination"] = True
                    self.pymoo_message_queue.put(
                        {"decision_space": algorithm.pop.get("X"),
                         "objective_space": algorithm.pop.get("F"),
                         "finished": False,
                         "n_gen_since_last_fetch": self.n_gen_since_last_fetch,
                         "n_gen": 0
                         })
                    algorithm.termination.terminate()
                elif callback_obj["reset"] == True and callback_obj["graph"] is not False:
                    queue_filled = True
                    self.data["external_reset"] = True
                    self.data["graph"] = callback_obj["graph"]

                    self.pymoo_message_queue.put(
                        {"decision_space": algorithm.pop.get("X"),
                         "objective_space": algorithm.pop.get("F"),
                         "finished": False,
                         "n_gen_since_last_fetch": self.n_gen_since_last_fetch,
                         "n_gen": 0
                         })
                    algorithm.termination.terminate()
                elif callback_obj["send_results"] == True:
                    queue_filled = True
                    self.pymoo_message_queue.put(
                        {"decision_space": algorithm.pop.get("X"),
                         "objective_space": algorithm.pop.get("F"),
                         "finished": False,
                         "n_gen_since_last_fetch": self.n_gen_since_last_fetch,
                         "n_gen": algorithm.n_gen
                         })
                    self.n_gen_since_last_fetch = 0

            # write to pymoo message queue
            if queue_filled is False:
                self.pymoo_message_queue.put(
                    {"decision_space": False,
                     "objective_space": False,
                     "finished": False,
                     "n_gen_since_last_fetch": self.n_gen_since_last_fetch,
                     "n_gen": algorithm.n_gen
                     })

################################ main ###################
# TODO take SOn as arguemtn away and replace with dicts, graph or even array


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
        folder_path: str,
        pymoo_message_queue: multiprocessing.Queue,
        editor_message_queue: multiprocessing.Queue,
        running_mode: str,
        prob_mutation: float = 0.3,
        prob_crossover: float = 0.9):

    pymooAlgorithm = None
    samplingConfig = None
    mutationConfig = None
    crossoverConfig = None

    verbose = True
    history = True
    if running_mode == RunningMode.LIVE.value:
        verbose = True
        history = False
    else:
        verbose = True
        history = True

    # sampling
    if (sampling == SamplingEnum.RANDOM_SAMPLING.value):
        samplingConfig = SonSampling()
    else:
        samplingConfig = SonSampling()

    # crossover
    if (crossover == CrossoverEnum.SBX_CROSSOVER.value):
        crossoverConfig = SBX(prob=prob_crossover, eta=3, vtype=float, repair=RoundingRepair())
    elif (crossover == CrossoverEnum.UNIFORM_CROSSOVER.value):
        crossoverConfig = UniformCrossover(prob=prob_crossover)
    elif (crossover == CrossoverEnum.ONE_POINT_CROSSOVER.value):
        crossoverConfig = SinglePointCrossover(prob=prob_crossover)
    elif (crossover == CrossoverEnum.SON_CROSSOVER.value):
        crossoverConfig = SonCrossover(prob=prob_crossover)
    else:
        crossoverConfig = SonCrossover(prob=prob_crossover)

    # mutation
    if (mutation == MutationEnum.PM_MUTATION.value):
        mutationConfig = PolynomialMutation(
            prob=prob_mutation, eta=3, vtype=float, repair=RoundingRepair())
    elif (mutation == MutationEnum.BIT_FLIP):
        mutationConfig = BitflipMutation(prob=prob_mutation, prob_var=0.3)
    else:
        mutationConfig = SonMutation()

    # algorithm config
    if (algorithm == AlgorithmEnum.NSGA3.value):
        # create the reference directions to be used for the optimization in NSGA3
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

    # build termination criterias

    termination_obj = get_termination("n_gen", n_generations)
    # start computatoin with  termination criteria

    result = minimize(
        sonProblem, pymooAlgorithm, termination=termination_obj, seed=1, verbose=verbose,
        save_history=history,
        callback=MyCallback(
            pymoo_message_queue=pymoo_message_queue, editor_message_queue=editor_message_queue,
            graph=son_obj.graph, running_mode=running_mode))

    if running_mode == RunningMode.LIVE.value:
        while result.algorithm.callback.data["external_reset"] and result.algorithm.callback.data["external_termination"] == False:

            # reinitialize algorithm config

            new_graph: nx.Graph = nx.from_edgelist(
                result.algorithm.callback.data["graph"]["edge_list_with_attributes"])
            nx.set_node_attributes(
                new_graph, result.algorithm.callback.data["graph"]["node_dic_with_attributes"])
            son_obj.graph = new_graph

            sonProblem = SonProblemElementWise(
                obj_dict=objectives, son=son_obj)

            samplingConfig = SeedSampling(seed_pop=result.pop.get("X"))

            if (algorithm == AlgorithmEnum.NSGA3.value):
                # create the reference directions to be used for the optimization in NSGA3
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
                                       n_generations=n_generations)
            result = minimize(sonProblem, pymooAlgorithm, termination=termination_obj, seed=1,
                              verbose=verbose, save_history=history,
                              callback=MyCallback(
                                  pymoo_message_queue=pymoo_message_queue,
                                  editor_message_queue=editor_message_queue,
                                  graph=son_obj.graph,
                                  running_mode=running_mode))

        pymoo_message_queue.put(
            {"decision_space": result.X,
             "objective_space": result.F,
             "finished": False,
             "n_gen_since_last_fetch": False,
             "n_gen": False
             })

    decisionSpace = result.X
    objectiveSpace = result.F
    exec_time = result.exec_time
    print("------- execution time in ms ------")
    print(exec_time)

    if running_mode == RunningMode.STATIC.value:
        n_evals_list = []        # corresponding number of function evaluations\
        hist_F = []              # the objective space values in each generation
        hist_cv = []             # constraint violation in each generation
        hist_cv_avg = []         # average constraint violation in the whole population

        for algo in result.history:
            # store the number of function evaluations
            n_evals_list.append(algo.evaluator.n_eval)

            # retrieve the optimum from the algorithm
            opt = algo.opt

            # store the least contraint violation and the average in each population
            hist_cv.append(opt.get("CV").min())
            hist_cv_avg.append(algo.pop.get("CV").mean())

            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas].tolist())

        # save all result individuums as json and create objective result dict
        objective_result_dic = {
            "optimization_objectives": objectives,
            "results": [],
            "decisionSpace": decisionSpace.tolist(),
            "objectiveSpace": objectiveSpace.tolist(),
            "history": {
                "n_evals": n_evals_list,
                "objective_space_opt": hist_F,
                "hist_cv": hist_cv,
                "hist_cv_avg": hist_cv_avg
            }}

        for i, individuum in enumerate(decisionSpace):
            sonProblem.son_original.apply_edge_activation_encoding_to_graph(individuum)
            objective_result_dic["results"].append(
                ("ind_result_" +
                 str(i + 1),
                 {ObjectiveEnum.AVG_SINR.name: sonProblem.son_original.get_average_sinr(),
                  ObjectiveEnum.AVG_RSSI.name: sonProblem.son_original.get_average_rssi(),
                  ObjectiveEnum.AVG_LOAD.name: sonProblem.son_original.get_average_network_load(),
                  ObjectiveEnum.POWER_CONSUMPTION.name: sonProblem.son_original.get_total_energy_consumption(),
                  ObjectiveEnum.ENERGY_EFFICIENCY.name: sonProblem.son_original.get_energy_efficiency(),
                  ObjectiveEnum.AVG_DL_RATE.name: sonProblem.son_original.get_average_dl_datarate()}))
            sonProblem.son_original.save_json_adjacency_graph_to_file(
                filename=folder_path + "ind_result_" + str(i + 1) + ".json")

        # save objecitve results to overview file
        # Convert the list to JSON
        json_data = json.dumps(objective_result_dic)

        # Save JSON data to a file
        file_path = folder_path + "objectives_result.json"
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(json_data)

    pymoo_message_queue.put({"decision_space": False,
                             "objective_space": False,
                             "finished": True,
                             "n_gen_since_last_fetch": False,
                             "n_gen": False
                             })


if __name__ == "__main__":
    print("nothing")
