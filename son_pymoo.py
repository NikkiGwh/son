from copy import deepcopy
import json
import multiprocessing
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.base.genetic import GeneticAlgorithm, Algorithm
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
from son_main_script import Son
from pymoo.core.variable import Real, get
from enum import Enum
from pymoo.decomposition.asf import ASF
from pymoo.core.callback import Callback
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.termination import get_termination
from pymoo.core.termination import Termination
import networkx as nx


class ObjectiveEnum(Enum):
    AVG_LOAD = "AVG_LOAD"
    AVG_SINR = "AVG_SINR"
    OVERLOAD = "OVERLOAD"
    POWER_CONSUMPTION = "POWER_CONSUMPTION"
    AVG_RSSI = "AVG_RSSI"
    AVG_DL_RATE = "AVG_DL_RATE"
    TOTAL_ENERGY_EFFICIENCY = "TOTAL_ENERGY_EFFICIENCY"
    AVG_ENERGY_EFFICENCY = "AVG_ENERGY_EFFICIENCY"

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
    BIT_FLIP = "BIT_FLIP"
    PM_MUTATION = "PM_MUTATION"
    SON_MUTATION= "SON_MUTATION"
class SamplingEnum(Enum):
    SON_RANDOM_SAMPLING = "SON_RANDOM_SAMPLING"
    # SMALL_BS_FIRST_SAMPLING = "SMALL_BS_FIRST_SAMPLING"
    # BIG_BS_FIRST_SAMPLING = "BIG_BS_FIRST_SAMPLING"
    HIGH_RSSI_FIRST_SAMPLING = "HIGH_RSSI_FIRST_SAMPLING"

def convert_design_space_ind_to_decision_space_ind(
        son: Son, decision_space_ind: dict[str, str]):
    
    possible_activation_dict = son.get_possible_activations_dict()

    # create decision space encoding for individuum -> orienting on possible_activation_matrix order
    ind_decision_space = []
    for _, user_id in enumerate(possible_activation_dict):
        ind_decision_space.append(possible_activation_dict[user_id].index(decision_space_ind[user_id]))

    return ind_decision_space

def convert_design_space_pop_to_decision_space_pop(
        son: Son, design_space_pop: list[dict[str, str]],
        repair=True):
    pop = []
    possible_activation_dict = son.get_possible_activations_dict()

    for _, individuum_dict in enumerate(design_space_pop):
        for _, user_id in enumerate(individuum_dict):
            # repair individuum if neccessary
            if repair and individuum_dict[user_id] not in possible_activation_dict[user_id]:

                individuum_dict[user_id] = son.greedy_assign_user_to_bs(
                    user_id)

        # create decision space encoding for individuum -> orienting on possible_activation_matrix order
        ind_decision_space = []
        for _, user_id in enumerate(possible_activation_dict):
            ind_decision_space.append(possible_activation_dict[user_id].index(
                individuum_dict[user_id]))
        pop.append(ind_decision_space)
    return pop


def convert_decision_space_ind_to_design_space_ind(
        son: Son, decision_space_ind: list[int]):

    individuum = {}
    possible_activation_dict = son.get_possible_activations_dict()
    possible_activatino_list = list(possible_activation_dict)

    for index, value in enumerate(decision_space_ind):
        individuum[possible_activatino_list[index]
                   ] = possible_activation_dict[possible_activatino_list[index]][value]

    return individuum

def convert_decision_Space_pop_to_design_space_pop(
        son: Son, decision_space_pop: list[list[int]],
        repair=False):
    pop = []

    for _, individuum in enumerate(decision_space_pop):
        pop.append(convert_decision_space_ind_to_design_space_ind(son, individuum))

    return pop


def select_solution(son: Son, decision_space, objective_space: np.ndarray,
                    weights= []):
    '''Picks one solution with ASF and given weighting

    Keyword arguments:

    decision_space -- of type list[list[int]]

    objective_space -- of type numpy array (list[list[float]])

    weights -- of type numpy array (list[list[float]]), summing up to 1,
    vector of length equal to len(objective_space)
    '''
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
    design_space_ind = convert_decision_space_ind_to_design_space_ind(son, decision_space[i])
    return design_space_ind


class SonProblemElementWise(ElementwiseProblem):
    def __init__(self, obj_dict: list[str], son: Son):

        # prepare network
        self.son_original = son
        edge_list_with_attributes = self.son_original.graph.edges
        node_dic_with_attributes = {}
        for _, node in enumerate(self.son_original.graph.nodes.data()):
            node_dic_with_attributes[node[0]] = node[1]

        new_graph: nx.Graph = nx.from_edgelist(edge_list_with_attributes)
        nx.set_node_attributes(new_graph, node_dic_with_attributes)
        self.son_original.graph = new_graph
        self.possible_activation_dict: dict[str, list[str]
                                            ] = self.son_original.get_possible_activations_dict()

        # prepare objectives
        self.obj_dict = obj_dict
        n_var = len(
            list(filter(self.son_original.filter_user_nodes, self.son_original.graph.nodes.data())))

        # prepare problem parameter
        xu = np.array([])
        for _, user_id in enumerate(self.possible_activation_dict):
            xu = np.append(xu, len(self.possible_activation_dict[user_id])-1)

        # call super class constructor
        super().__init__(n_var=n_var, n_obj=len(obj_dict), xl=np.full_like(xu, 0), xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x_design_space = convert_decision_space_ind_to_design_space_ind(self.son_original, x)
        self.son_original.apply_activation_dict(x_design_space)
        # prepare objectives
        objectives = np.array([])

        if ObjectiveEnum.AVG_LOAD.value in self.obj_dict:
            objectives = np.append(objectives, self.son_original.get_average_network_load())
        if ObjectiveEnum.OVERLOAD.value in self.obj_dict:
            objectives = np.append(objectives, self.son_original.get_avg_overlad())
        if ObjectiveEnum.POWER_CONSUMPTION.value in self.obj_dict:
            objectives = np.append(objectives, self.son_original.get_total_energy_consumption())
        if ObjectiveEnum.TOTAL_ENERGY_EFFICIENCY.value in self.obj_dict:
            objectives = np.append(objectives, -self.son_original.get_total_energy_efficiency())
        if ObjectiveEnum.AVG_ENERGY_EFFICENCY.value in self.obj_dict:
            objectives = np.append(objectives, -self.son_original.get_avg_energy_efficiency())
        if ObjectiveEnum.AVG_SINR.value in self.obj_dict:
            objectives = np.append(objectives, -self.son_original.get_average_sinr())
        if ObjectiveEnum.AVG_DL_RATE.value in self.obj_dict:
            objectives = np.append(objectives, -self.son_original.get_average_dl_datarate())
        if ObjectiveEnum.AVG_RSSI.value in self.obj_dict:
            objectives = np.append(objectives, -self.son_original.get_average_rssi())
        out["F"] = np.array(objectives)


def get_upper_lower_boundaries_from_graph(son: Son):

    xu = np.array(son.get_possible_activations_dict())
    xl = np.full_like(xu, 1)

    return (xl, xu)


def repair_population(pop: np.ndarray, xl: np.ndarray, xu: np.ndarray):
    for individuum_index, individuum in enumerate(pop):
        for gene_index, gene_value in enumerate(individuum):
            if gene_value > xu[gene_index] or gene_value < xl[gene_index]:
                pop[individuum_index][gene_index] = np.random.randint(
                    xl[gene_index], xu[gene_index]+1)
    return pop

class SonRepairSampling(Sampling):

    def __init__(self, seed_pop_design_space, target_pop_size) -> None:
        self.seed_pop_design_space = seed_pop_design_space
        self.target_pop_size = target_pop_size
        super().__init__()

    def _do(self, problem: SonProblemElementWise, n_samples, **kwargs):

        # convert seed_pop to decision space and repair it to match current topology
        seed_pop_decision_space = convert_design_space_pop_to_decision_space_pop(
            problem.son_original, self.seed_pop_design_space, repair=True)
        
        return seed_pop_decision_space


class SonRandomSampling(Sampling):
    def _do(self, problem: SonProblemElementWise, n_samples, **kwargs):
        X = np.empty((n_samples, problem.n_var), int)

        for i in range(n_samples):
            for j in range(problem.n_var):
                X[i][j] = np.random.randint(problem.xl[j], problem.xu[j]+1)
        return X
class HighRssiFirstSampling(Sampling):
    def _do(self, problem: SonProblemElementWise, n_samples, **kwargs):
        X = np.empty((n_samples, problem.n_var), int)

        # create the first n_samples-1 individuums randomly
        for i in range(n_samples-1):
            for j in range(problem.n_var):
                X[i][j] = np.random.randint(problem.xl[j], problem.xu[j]+1)
        
        # last individuum is created with greedy assignment (highest rssi connection for each user)
        greedy_activation_design_space = problem.son_original.find_activation_profile_greedy_user()
        greedy_activation_decision_space = convert_design_space_ind_to_decision_space_ind(problem.son_original, greedy_activation_design_space)
        X[-1] = greedy_activation_decision_space

        return X


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

    def __init__(self, prob:float, prob_var):
        super().__init__(prob=prob, prob_var=prob_var)
    
   
    def _do(self, problem: SonProblemElementWise, X, **kwargs):
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        flip = np.random.random(X.shape) < prob_var
        possible_activations_dict = problem.son_original.get_possible_activations_dict()
        possible_activations_list = list(possible_activations_dict)

        for ind_index, ind_decision_space in enumerate(Xp):
            for variable_index, active_bs in enumerate(ind_decision_space):
                # print(possible_activations_dict[possible_activations_list[variable_index]])
                # print(active_bs)
                # print(flip[ind_index][variable_index])
                # print("------")
                new_active_bs = active_bs
                if flip[ind_index][variable_index]:
                    while active_bs == new_active_bs and len(possible_activations_dict[possible_activations_list[variable_index]]) > 1:
                        new_active_bs = np.random.randint(0, len(possible_activations_dict[possible_activations_list[variable_index]]))
                    Xp[ind_index][variable_index] = new_active_bs
        return Xp

    def do(self, problem: SonProblemElementWise, pop, inplace=True, **kwargs):

        # if not inplace copy the population first
        if not inplace:
            pop = deepcopy(pop)

        n_mut = len(pop)

        # get the variables to be mutated -> only the new offsprings are considered
        X = pop.get("X")
        # retrieve the mutation variables
        Xp = self._do(problem, X, **kwargs)

        # the likelihood for a mutation on the individuals
        prob = get(self.prob, size=n_mut)

        mut = np.random.random(size=n_mut) <= prob

        # store the mutated individual back to the population
        pop[mut].set("X", Xp[mut])

        return pop


class SonDublicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return np.array_equal(a, b)

class MyNoTermination(Termination):
    def __init__(self) -> None:
        super().__init__()

        # the algorithm can be forced to terminate by setting this attribute to true
        self.force_termination_niklas = False

    def has_terminated(self):
        return self.force_termination_niklas
    
    def terminate(self):
        self.force_termination_niklas = True
    def _update(self, algorithm):
        return 0.0
class MyCallback(Callback):

    def __init__(self, pymoo_message_queue: multiprocessing.Queue,
                 editor_message_queue: multiprocessing.Queue, son: Son, running_mode: str,
                 total_gen=0) -> None:
        super().__init__()

        self.data["external_termination"] = False
        self.data["external_reset"] = False
        self.data["graph"] = son.graph
        self.son = son
        self.total_gen = total_gen
        self.pymoo_message_queue = pymoo_message_queue
        self.editor_message_queue = editor_message_queue
        self.running_mode = running_mode
        self.n_gen_since_last_fetch = 0
        self.n_gen_since_last_reset = 0
        self.trigger_terminate = False

    def notify(self, algorithm: Algorithm):

        # TODO adjust decision space boundaies here and don't restart optimization
        # TODO repair current population here

        if self.running_mode == RunningMode.LIVE.value or self.running_mode == RunningMode.STATIC.value:
            # update counter
            self.n_gen_since_last_fetch += 1
            self.n_gen_since_last_reset += 1
            
            # read editor message queue
            while self.editor_message_queue.empty() is False:
                queue_obj = self.editor_message_queue.get()

                if queue_obj["terminate"] == True:
                    ##### react to terminate #####
                    self.data["external_termination"] = True
                    self.trigger_terminate = True
                elif queue_obj["reset"] == True and queue_obj["graph"] is not False:
                    ##### react to reset #####
                    self.data["external_reset"] = True
                    self.data["graph"] = queue_obj["graph"]
                    self.n_gen_since_last_reset = 0
                    self.trigger_terminate = True


            ####### normal result propagation after ######

            activation_dict = select_solution(
                self.son,
                decision_space=algorithm.pop.get("X"),
                objective_space=algorithm.pop.get("F"))
            
            self.n_gen_since_last_fetch = 0
            
            if self.pymoo_message_queue.empty() and not self.trigger_terminate:
                self.pymoo_message_queue.put(
                    {"activation_dict": activation_dict,
                        "objective_space": algorithm.pop.get("F"),
                        "finished": False,
                        "just_resetted": True if self.n_gen_since_last_reset == 0 else False,
                        "n_gen_since_last_fetch": self.n_gen_since_last_fetch,
                        "n_gen_since_last_reset": self.n_gen_since_last_reset,
                        "n_gen": algorithm.n_gen + self.total_gen
                        })
            
            if self.trigger_terminate:
                algorithm.termination.terminate()
                self.trigger_terminate = False
                self.pymoo_message_queue.put(
                    {"activation_dict": activation_dict,
                        "objective_space": algorithm.pop.get("F"),
                        "finished": False,
                        "just_resetted": True if self.n_gen_since_last_reset == 0 else False,
                        "n_gen_since_last_fetch": self.n_gen_since_last_fetch,
                        "n_gen_since_last_reset": self.n_gen_since_last_reset,
                        "n_gen": algorithm.n_gen + self.total_gen
                        })

################################ main ###################

def start_optimization(
        pop_size: int,
        n_offsprings: int, ## pop_size is pymoo default
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
        mutation_prob: float = 0.9,  # 0.9 is pymoo default
        crossover_prob: float = 0.9,  # 0.9 is pymoo default
        mutation_prob_var = None,
        seed=None):

    pymooAlgorithm = None
    samplingConfig = None
    mutationConfig = None
    crossoverConfig = None

    verbose = True
    history = True

    if running_mode == RunningMode.LIVE.value:
        history = False
    else:
        history = True

    # sampling
    if (sampling == SamplingEnum.SON_RANDOM_SAMPLING.value):
        samplingConfig = SonRandomSampling()
    elif sampling == SamplingEnum.HIGH_RSSI_FIRST_SAMPLING.value:
        samplingConfig = HighRssiFirstSampling()
    else:
        samplingConfig = SonRandomSampling()

    # crossover
    if (crossover == CrossoverEnum.SBX_CROSSOVER.value):
        crossoverConfig = SBX(prob=crossover_prob, eta=3, vtype=float, repair=RoundingRepair())
    elif (crossover == CrossoverEnum.UNIFORM_CROSSOVER.value):
        crossoverConfig = UniformCrossover(prob=crossover_prob)
    elif (crossover == CrossoverEnum.ONE_POINT_CROSSOVER.value):
        crossoverConfig = SinglePointCrossover(prob=crossover_prob)
    elif (crossover == CrossoverEnum.SON_CROSSOVER.value):
        crossoverConfig = SonCrossover(prob=crossover_prob)
    else:
        crossoverConfig = SonCrossover(prob=crossover_prob)

    # mutation
    if (mutation == MutationEnum.PM_MUTATION.value):
        mutationConfig = PolynomialMutation(
            prob=mutation_prob, prob_var=mutation_prob_var, eta=3, vtype=int)
    elif (mutation == MutationEnum.BIT_FLIP):
        mutationConfig = BitflipMutation(prob=mutation_prob, prob_var=mutation_prob_var)
    else:
        mutationConfig = SonMutation(prob=mutation_prob, prob_var=mutation_prob_var)

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
    if running_mode == RunningMode.STATIC.value:
        termination_obj = get_termination("n_gen", n_generations)
    else:
        termination_obj = MyNoTermination()


    result = minimize(sonProblem, pymooAlgorithm, termination=termination_obj, seed=seed,
                      verbose=verbose, save_history=history,
                      callback=MyCallback(
                          pymoo_message_queue=pymoo_message_queue,
                          editor_message_queue=editor_message_queue, son=son_obj,
                          running_mode=running_mode))
    total_gen = 0
    if running_mode == RunningMode.LIVE.value:
        # total_gen = result.algorithm.n_gen
        total_gen = 0
        while result.algorithm.callback.data["external_reset"] and result.algorithm.callback.data["external_termination"] == False:

            # create design_space_pop with old son object -> otherwise conversion is wrong,
            # no repair necessary ?

            design_space_pop = convert_decision_Space_pop_to_design_space_pop(
                son_obj, result.pop.get("X"), repair=False)

            # reinitialize algorithm config
            new_graph: nx.Graph = nx.from_edgelist(
                result.algorithm.callback.data["graph"]["edge_list_with_attributes"])
            nx.set_node_attributes(
                new_graph, result.algorithm.callback.data["graph"]["node_dic_with_attributes"])
            son_obj.graph = new_graph

            sonProblem = SonProblemElementWise(obj_dict=objectives, son=son_obj)

            samplingConfig = SonRepairSampling(
                seed_pop_design_space=design_space_pop,
                target_pop_size=pop_size)

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

            total_gen += result.algorithm.n_gen-1
            result = minimize(
                sonProblem, pymooAlgorithm, termination=termination_obj, seed=seed,
                verbose=verbose, save_history=history,
                callback=MyCallback(
                    pymoo_message_queue=pymoo_message_queue,
                    editor_message_queue=editor_message_queue,
                    son=son_obj,
                    running_mode=running_mode, total_gen=total_gen))


    if running_mode == RunningMode.STATIC.value:
        total_gen = 0
        designSpace = convert_decision_Space_pop_to_design_space_pop(son_obj, result.X, repair=False)
        objectiveSpace = result.F
        exec_time = result.exec_time
        print("------- staic execution time in ms ------")
        print(exec_time)
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
            "designSpace": designSpace,
            "objectiveSpace": objectiveSpace.tolist(),
            "history": {
                "n_evals": n_evals_list,
                "objective_space_opt": hist_F,
                "hist_cv": hist_cv,
                "hist_cv_avg": hist_cv_avg
            }}
        # convert designspae result to adjacency json
        for i, individuum in enumerate(designSpace):
            sonProblem.son_original.apply_activation_dict(individuum)
            objective_result_dic["results"].append(
                ("ind_result_" +
                 str(i + 1),
                 {ObjectiveEnum.AVG_SINR.name: sonProblem.son_original.get_average_sinr(),
                  ObjectiveEnum.AVG_RSSI.name: sonProblem.son_original.get_average_rssi(),
                  ObjectiveEnum.AVG_LOAD.name: sonProblem.son_original.get_average_network_load(),
                  ObjectiveEnum.POWER_CONSUMPTION.name: sonProblem.son_original.get_total_energy_consumption(),
                  ObjectiveEnum.TOTAL_ENERGY_EFFICIENCY.name: sonProblem.son_original.get_total_energy_efficiency(),
                  ObjectiveEnum.AVG_ENERGY_EFFICENCY.name: sonProblem.son_original.get_avg_energy_efficiency(),
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

    activation_dict = select_solution(
        son_obj,
        decision_space=result.X,
        objective_space=result.F)
    pymoo_message_queue.put(
        {"activation_dict": activation_dict,
            "objective_space": result.F,
            "finished": True,
            "n_gen_since_last_fetch": 0,
            "n_gen_since_last_reset": 0,
            "just_resetted": False,
            "n_gen": total_gen + result.algorithm.n_gen-1
            })



if __name__ == "__main__":
    print("nothing")
