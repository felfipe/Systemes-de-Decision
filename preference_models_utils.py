import json
import gzip
from typing import Iterable, TextIO, List, Generator
import dataclasses
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
import plotly.express as px
import pandas as pd

from gurobipy import Model, GRB, Var, MVar, LinExpr, MLinExpr, quicksum

import networkx as nx
from typing import Tuple
import matplotlib.pyplot as plt


@dataclass
class Solution:
    Nm: int  # nombre de membres
    Nc: int  # nombre de compétences
    Np: int  # nombre de projets
    Nj: int  # délai maximal des projets

    nom_mapping: int # crée un mapping entre nom et index
    travail_mapping: int # crée un mapping entre travail et index
    qualifications_mapping: int # crée un mapping entre qualifications et index
    jour_mapping: int # crée un mapping entre jour et index

    qualifications: List[str]  # liste de compétences distinctes
    staff: List[str]  # liste des noms des membres
    vacations: List[List[int]]  # jours des congé par membre (dans le même ordre que staff)
    projects: List[str]  # liste des noms des projets

    T: np.ndarray  # variable de décision principale, shape (Nm, Np, Nc, Nj)
    R: np.ndarray  # indique si un projet est réalisé, shape (Np,)
    De: np.ndarray  # jour de début de chaque projet, shape (Np,)
    F: np.ndarray  # jour de fin de chaque projet, shape (Np,)
    Re: np.ndarray  # rétard par projet, shape (Np,)
    Dm: int  # durée maximale d'un projet
    Af: np.ndarray  # indique si une personne a travaillé sur un projet, shape (Nm, Np)
    Mp: int  # indique le nombre maximum de projets par personne

    f1: int
    f2: int
    f3: int

    gap: float

    @staticmethod
    def from_python_dict(d: dict) -> "Solution":
        kwargs = {}
        for field in dataclasses.fields(Solution):
            if field.type == np.ndarray:
                kwargs[field.name] = np.asarray(d[field.name])
            else:
                kwargs[field.name] = d[field.name]

        return Solution(**kwargs)


class SolutionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Solution):
            return obj.__dict__
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Instance:
    model: Model

    Nm: int  # nombre de membres
    Nc: int  # nombre de compétences
    Np: int  # nombre de projets
    Nj: int  # délai maximal des projets

    qualifications: List[str]  # liste de compétences distinctes
    staff: List[str]  # liste des noms des membres
    vacations: List[List[int]]  # jours des congé par membre (dans le même ordre que staff)
    projects: List[str]  # liste des noms des projets

    H: np.ndarray  # attribution des compétences, shape (Nm, Nc)
    C: np.ndarray  # congés, shape (Nm, Nj)
    Q: np.ndarray  # compétences réquises par projet, shape (Np, Nc)
    Rev: np.ndarray  # revenu des projets, shape (Np,)
    P: np.ndarray  # pénalité des projets, shape (Np,)
    Dl: np.ndarray  # deadline des projets, shape (Np,)

    T: MVar  # variable de décision principale, shape (Nm, Np, Nc, Nj)
    R: MVar  # indique si un projet est réalisé, shape (Np,)
    De: MVar  # jour de début de chaque projet, shape (Np,)
    F: MVar  # jour de fin de chaque projet, shape (Np,)
    Re: MVar  # rétard par projet, shape (Np,)
    Dm: Var  # durée maximale d'un projet
    Af: MVar  # indique si une personne a travaillé sur un projet, shape (Nm, Np)
    Mp: Var  # indique le nombre maximum de projets par personne

    f1: MLinExpr
    f2: Var
    f3: Var

    def __init__(self, instance_path: str) -> None:
        with open(instance_path) as f:
            instance = json.load(f)

        self.qualifications = list(instance["qualifications"])
        self.staff = [person["name"] for person in instance["staff"]]
        self.vacations = [person["vacations"] for person in instance["staff"]]
        self.projects = [job["name"] for job in instance["jobs"]]

        # Tailles des données
        self.Nm = len(instance["staff"])
        self.Nc = len(self.qualifications)
        self.Np = len(instance["jobs"])
        self.Nj = instance["horizon"]

        # Création des matrices d'entrée
        self.H = np.zeros((self.Nm, self.Nc), dtype=np.int32)
        self.C = np.zeros((self.Nm, self.Nj), dtype=np.int32)
        self.Q = np.zeros((self.Np, self.Nc), dtype=np.int32)
        self.Rev = np.zeros(self.Np, dtype=np.int32)
        self.P = np.zeros(self.Np, dtype=np.int32)
        self.Dl = np.zeros(self.Np, dtype=np.int32)

        for i, person in enumerate(instance["staff"]):
            for qualif in person["qualifications"]:
                k = self.qualifications.index(qualif)
                self.H[i, k] = 1
            for vacation in person["vacations"]:
                self.C[i, vacation - 1] = 1

        for j, job in enumerate(instance["jobs"]):
            for qualif, days in job["working_days_per_qualification"].items():
                k = self.qualifications.index(qualif)
                self.Q[j, k] = days
            self.Rev[j] = job["gain"]
            self.P[j] = job["daily_penalty"]
            self.Dl[j] = job["due_date"]

        # Création du modèle Gurobi
        self.model = Model("CompuOpti")

        # Variables de decision
        self.T = self.model.addMVar((self.Nm, self.Np, self.Nc, self.Nj), vtype=GRB.BINARY, name="T")
        self.R = self.model.addMVar(self.Np, vtype=GRB.BINARY, name="R")
        self.De = self.model.addMVar(self.Np, lb=1, ub=self.Nj, vtype=GRB.INTEGER, name="De")
        self.F = self.model.addMVar(self.Np, lb=1, ub=self.Nj, vtype=GRB.INTEGER, name="F")
        self.Re = self.model.addMVar(self.Np, lb=0, ub=self.Nj-1, vtype=GRB.INTEGER, name="Re")
        self.Dm = self.model.addVar(lb=0, ub=self.Nj, vtype=GRB.INTEGER, name="Dm")
        self.Af = self.model.addMVar((self.Nm, self.Np), vtype=GRB.BINARY, name="Af")
        self.Mp = self.model.addVar(lb=0, ub=self.Np, vtype=GRB.INTEGER, name="Mp")

        # Contrainte de qualification
        for j in range(self.Np):
            for ell in range(self.Nj):
                self.model.addConstr(self.T[:, j, :, ell] <= self.H)

        # Contrainte d’unicité de l’affectation
        for i in range(self.Nm):
            for ell in range(self.Nj):
                self.model.addConstr(self.T[i, :, :, ell].sum() <= 1)

        # Contrainte de congé
        for j in range(self.Np):
            for k in range(self.Nc):
                self.model.addConstr(self.T[:, j, k, :] <= 1 - self.C)

        # Contraintes d’unicité de la réalisation d’un projet et de couverture des qualifications
        for j in range(self.Np):
            for k in range(self.Nc):
                self.model.addConstr(self.T[:, j, k, :].sum() >= self.R[j] * self.Q[j, k])

        # Contraintes sur les variables d'affectation (Af et Mp)
        for k in range(self.Nc):
            for ell in range(self.Nj):
                self.model.addConstr(self.Af >= self.T[:, :, k, ell])
        for i in range(self.Nm):
            self.model.addConstr(self.Mp >= self.Af[i, :].sum())

        # Contraintes sur la durée d’un projet
        for i in range(self.Nm):
            for k in range(self.Nc):
                for ell in range(self.Nj):
                    self.model.addConstr(self.De <= (ell + 1) * self.T[i, :, k, ell] + self.Nj * (1 - self.T[i, :, k, ell]))
                    self.model.addConstr(self.F >= (ell + 1) * self.T[i, :, k, ell])
        self.model.addConstr(self.Re >= self.F - self.Dl)
        self.model.addConstr(self.Dm >= self.F - self.De + self.R)

        # Fonctions objectifs
        self.f1 = -(self.Rev.T @ self.R - self.P.T @ self.Re)
        self.f2 = self.Mp
        self.f3 = self.Dm

        self.model.setObjective(self.f1, GRB.MINIMIZE)

    def get_current_solution(self) -> Solution:
        kwargs = {}
        for field in dataclasses.fields(Solution):
            kwargs[field.name] = self._get_value(field.name)
        return Solution(**kwargs)

    def iter_solutions(self) -> Generator[Solution, None, None]:
        if not self.is_multiscene:
            yield self.get_current_solution()
            return

        initial_value = self.model.params.ScenarioNumber
        for i in range(self.model.NumScenarios):
            self.model.params.ScenarioNumber = i
            try:
                if np.isinf(self.objective_value):
                    print(f'Scenario {i} is infeasible or no feasible solution found')
                else:
                    yield self.get_current_solution()
                    print(f'Got scenario {i}')
            except GurobiError as e:
                print(f'Error getting solution for scenario {i}: {e}')
        self.model.params.ScenarioNumber = initial_value

    def get_solutions(self) -> List[Solution]:
        return list(self.iter_solutions())

    @property
    def is_multiscene(self) -> bool:
        return self.model.NumScenarios > 0

    @property
    def objective_value(self) -> float:
        if self.is_multiscene:
            return self.model.ScenNObjVal
        else:
            return self.model.ObjVal

    @property
    def gap(self) -> float:
        if self.is_multiscene:
            bound = self.model.ScenNObjBound
        else:
            bound = self.model.ObjBound

        obj = self.objective_value
        if obj == 0:
            return 0 if abs(bound - obj) < 1e-5 else np.inf

        return abs((obj - bound) / obj)

    def _get_value(self, field_name: str):
        if field_name == "f1":
            raw = self.objective_value
        else:
            raw = getattr(self, field_name)
            if isinstance(raw, (Var, MVar)):
                raw = raw.ScenNX if self.is_multiscene else raw.X

        if isinstance(raw, float) and field_name != "gap":
            val = int(round(raw))
            assert abs(raw - raw) < 1e-5
        elif isinstance(raw, np.ndarray):
            val = raw.round().astype(np.int32)
            assert np.abs(raw - val).max() < 1e-5
        else:
            val = raw

        return val


def _open_output(path: str, mode: str) -> TextIO:
    if path.lower().endswith(".json.gz"):
        return gzip.open(path, mode=mode + "t", encoding="utf-8")  # type: ignore
    elif path.lower().endswith(".json"):
        return open(path, mode=mode)  # type: ignore

    ext = path.split('.')[-1]
    raise ValueError(f"Unsupported file format: {ext}. Expected .json or .json.gz")


def save_output(out: Iterable[Solution], path: str):
    it = iter(out)
    with _open_output(path, "w") as f:
        f.write("[")
        json.dump(next(it), f, cls=SolutionEncoder)
        for s in it:
            f.write(", ")
            json.dump(s, f, cls=SolutionEncoder)
        f.write("]")


def load_output(*paths: str) -> List[Solution]:
    assert len(paths) > 0
    out: List[Solution] = []

    for path in paths:
        with _open_output(path, "r") as f:
            out += json.load(f, object_hook=Solution.from_python_dict)

    return out


def filter_solutions(solutions : List[Solution]) -> List[Solution]:
    filtered_solutions = []

    def dominate(s1 : Solution, s2 : Solution):
        obj1 = np.array([s1.f1, s1.f2, s1.f3])
        obj2 = np.array([s2.f1, s2.f2, s2.f3])
        temp = obj1 - obj2
        
        if max(temp) > 0 or min(temp) == 0:
            return False
        
        return True

    for i, si in enumerate(solutions):
        non_dominated = True
        for j, sj in enumerate(solutions):
            if i == j:
                continue
            if dominate(sj, si):
                non_dominated = False    
                break
        if non_dominated:
            filtered_solutions.append(si)

    return filtered_solutions

def plot_obj_values(solutions):
    obj_f1 = [s.f1 for s in solutions]
    obj_f2 = [s.f2 for s in solutions]
    obj_f3 = [s.f3 for s in solutions]


    
    fig = px.scatter_3d(x=obj_f1, y=obj_f2, z=obj_f3)
    fig.show()


def plot_solution(solution : Solution):
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    ax.set_yticks([i+0.5 for i in range(len(solution.staff))], labels=solution.staff)
    # Horizontal bar plot with gaps
    ymargin = 0.1
    height = solution.Nm - ymargin
    width = solution.Nj    
    xmargin = ymargin / height * width

    ax.set_ylim(0, solution.Nm+ymargin)
    ax.set_xlim(0.5-xmargin, solution.Nj+0.5+xmargin)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))

    norm = matplotlib.colors.Normalize(vmin=0, vmax=solution.Np+1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='tab20')

    patches = [mpatches.Patch(color=mapper.to_rgba(i), label=proj) for i, proj in enumerate(solution.projects)]
    ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1))

    for i in range(solution.Nm):
        p_staff_list = []
        projects = []

        # plot planning
        for p in range(solution.Np):
            for c in range(solution.Nc):   
                for jour, worked in enumerate(solution.T[i][p][c]):
                    jour+=0.5 # center
                    if(worked == 1):
                        p_staff_list.append((jour, 1))
                        projects.append(p)
                        ax.annotate(solution.qualifications[c], (jour +0.5, i+0.5+ymargin/2), ha='center', va='center')

        ax.broken_barh(p_staff_list, (i + ymargin, 1 - ymargin), facecolors=mapper.to_rgba(projects))
        ax.set_xlabel('Day')              
        

def set_color(val: str, data: ModelData):
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
    if '_' not in val: return 
    val = val.split('_')[1]
    switch = {
        data.travail_mapping[proj]: f'background-color: {colors[proj]}'
        for proj in range(data.Np)
    }
    return switch.get(val)

def get_planning_from_solution(data: ModelData):
    planning = np.empty((data.Nm, data.Np, data.Nc, data.Nj), dtype=np.int64)
    for i in range(data.Nm):
        for j in range(data.Np):
            for k in range(data.Nc):
                for l in range(data.Nj):
                    planning[i][j][k][l] = data.T[i][j][k][l].X
    return planning

def color_mapping(data: ModelData):
    df_colors = {
        data.travail_mapping[proj]: ['x_'+data.travail_mapping[proj]]
        for proj in range(data.Np)
    }
    df_colors = pd.DataFrame(df_colors)
    df_colors = df_colors.style.applymap(set_color)
    return df_colors
                
def create_planning(data: ModelData):
    planning = data.T.ScenNX
    df = {
        day: [
            data.qualifications_mapping[planning[member, :, :, day].sum(axis=0).argmax()]+"_"+data.travail_mapping[planning[member, :, :, day].sum(axis=1).argmax()]
            if data.T.ScenNX[member][planning[member, :, :, day].sum(axis=1).argmax()][planning[member, :, :, day].sum(axis=0).argmax()][day] == 1
            else 'X'
            for member in range(data.Nm)
        ] 
        for day in range(data.Nj)
    }
    df['membre'] = data.nom_mapping.values()
    df = pd.DataFrame(df)
    df.set_index('membre', drop=True, inplace=True)
    df = df.style.applymap(lambda x: set_color(x, data))
    return df

def create_tables(data: ModelData):
    df_competences = pd.DataFrame(data.H)
    df_competences.rename(columns=lambda x: data.qualifications_mapping[int(x)], inplace=True)
    df_competences['membre'] = data.nom_mapping.values() 
    df_competences.set_index('membre', drop=True, inplace=True)

    df_projets = pd.DataFrame(data.Q)
    df_projets.rename(columns=lambda x: data.qualifications_mapping[int(x)], inplace=True)
    df_projets['projet'] = data.travail_mapping.values() 
    df_projets.set_index('projet', drop=True, inplace=True)

    return df_competences, df_projets


class ModelDataSommePonderee:
    dim:     int                    # Nombre de dimensions = nombre de criteres
    epsilon: int                    # Set to relax constraints
    nombreChoix: int                # Nombre de choix (solutions non dominée)
    
    for jour in range(solution.Nj+1):
        ax.axvline(jour+0.5, c='gray', lw=0.5)

def get_preorder(choices: list, preference_list: list[tuple]):

    model_data = ModelDataSommePonderee()
    model_preorder = Model("Preorder")

    model_data.dim = 3
    model_data.epsilon = 0.01
    model_data.nombreChoix = len(choices)

    model_data.Poids = {d : model_preorder.addVar(vtype = GRB.CONTINUOUS, lb=0, ub=0.5, name = f'w{d}') for d in range(1, model_data.dim + 1)}

    model_data.SommePonderee = {l+1 : quicksum([model_data.Poids[i+1]*choices[l][i] for i in range(model_data.dim)]) for l in range(model_data.nombreChoix)}

    model_preorder.addConstr(quicksum(model_data.Poids.values()) == 1., name="Somme des poids = 1")
    for idx, (a, b) in enumerate(preference_list):
        model_preorder.addConstr(model_data.SommePonderee[a] >= model_data.SommePonderee[b] + model_data.epsilon, name="C_"+str(idx))


    model_preorder.params.outputflag = 0
    model_preorder.update()

    model_data.PreOrdre = {sol+1 : list() for sol in range(model_data.nombreChoix)}

    for i in range(1, model_data.nombreChoix):
        for j in range(i+1, model_data.nombreChoix+1):
            obj1 = model_data.SommePonderee[i] - model_data.SommePonderee[j]
            model_preorder.setObjective(obj1, GRB.MAXIMIZE)
            model_preorder.optimize()
            if model_preorder.objVal < 0:
                model_data.PreOrdre[j].append(i)
            else:
                obj2 = - obj1
                model_preorder.setObjective(obj2, GRB.MAXIMIZE)
                model_preorder.optimize()
                if model_preorder.objVal < 0:
                    model_data.PreOrdre[i].append(j)

    return model_preorder, model_data

def create_graph(data: ModelDataSommePonderee, draw:bool = False):
    G = nx.DiGraph()

    for node in range(data.nombreChoix):
        G.add_node(node+1)

    for node_i in range(data.nombreChoix):
        for node_j in data.PreOrdre[node_i+1]:
            G.add_edge(node_i+1, node_j)
    
    if draw:
        pos = nx.circular_layout(G)
        nx.draw(G, pos,with_labels = True)
        
    return G

def dominate(s1, s2):
    temp = s1 - s2
    if max(temp) > 0 or min(temp) == 0:
        return False
    
    return True

def generate_pairs(ranking: list):
    pairs = []
    for i in range(len(ranking)):
        for j in range(i+1, len(ranking)):
            pairs.append((ranking[i], ranking[j]))
    return pairs

class ModelDataUTA:
    nbRanking: int
    dim: int
    L: int
    epsilon: float

    dim_vals: list
    scores: list

    s_i_x_k: dict[int, dict[int, MVar]]
    s: dict[int, MVar]
    eps_pos: dict[int, MVar]
    eps_neg: dict[int, MVar]

    f1: MLinExpr

def run_model_uta(choices: list, ranking: list):
    
    data = ModelDataUTA()
    model_uta = Model("UTA")

    data.nbRanking = len(ranking)
    data.dim = choices.shape[1]
    data.L = 3

    pairs = generate_pairs(ranking)
    data.dim_vals = []
    for i in range(data.dim):
        data.dim_vals.append(np.linspace(choices[:, i].min(), choices[:, i].max(), data.L))

    data.s_i_x_k = {}
    for i in range(data.dim): 
        data.s_i_x_k[i] = {}
        for k in range(data.L):
            data.s_i_x_k[i][k] = model_uta.addVar(vtype=GRB.CONTINUOUS, name=f's_{i}(x_{k})')

    data.s = {}
    for idx in range(data.nbRanking):
        point = ranking[idx]
        sum_dim = 0
        for d in range(data.dim):
            multiplier = (choices[point][d] - data.dim_vals[d][(data.dim_vals[d] - choices[point][d]).argmin()]) / (data.dim_vals[d][(data.dim_vals[d] - choices[point][d]).argmin()+ 1 ] - data.dim_vals[d][(data.dim_vals[d] - choices[point][d]).argmin()])
            sum_dim += data.s_i_x_k[d][(data.dim_vals[d] - choices[point][d]).argmin()] + multiplier*(data.s_i_x_k[d][(data.dim_vals[d] - choices[point][d]).argmin()+1] - data.s_i_x_k[d][(data.dim_vals[d] - choices[point][d]).argmin()])
        data.s[point] = sum_dim

    data.eps_pos = {ranking[j]: model_uta.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'sig_pos_{j}') for j in range(data.nbRanking)}
    data.eps_neg = {ranking[j]: model_uta.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'sig_neg_{j}') for j in range(data.nbRanking)}
    
    data.epsilon = 0.01

    for i,j in pairs:
            model_uta.addConstr(data.s[i] - data.eps_pos[i] + data.eps_neg[i] >= data.s[j] - data.eps_pos[j] + data.eps_neg[j] + data.epsilon)

    for i in range(data.dim):
        for k in range(data.L-1):
            model_uta.addConstr(data.s_i_x_k[i][k+1] - data.s_i_x_k[i][k] >= data.epsilon) 

    for i in range(data.dim):
        model_uta.addConstr(data.s_i_x_k[i][0] == 0)

    model_uta.addConstr(quicksum([data.s_i_x_k[i][data.L-1] for i in range(data.dim)]) == 1)

    data.f1 = quicksum(data.eps_pos.values()) + quicksum(data.eps_neg.values())

    model_uta.setObjective(data.f1, GRB.MINIMIZE)
    model_uta.update()

    model_uta.params.outputflag = 0
    model_uta.reset()
    model_uta.optimize()

    data.scores = []
    for point in range(len(choices)):
        sum_dim = 0
        for d in range(data.dim):
            multiplier = (choices[point][d] - data.dim_vals[d][(data.dim_vals[d] - choices[point][d]).argmin()]) / (data.dim_vals[d][(data.dim_vals[d] - choices[point][d]).argmin()+ 1 ] - data.dim_vals[d][(data.dim_vals[d] - choices[point][d]).argmin()])
            sum_dim += data.s_i_x_k[d][(data.dim_vals[d] - choices[point][d]).argmin()] + multiplier*(data.s_i_x_k[d][(data.dim_vals[d] - choices[point][d]).argmin()+1] - data.s_i_x_k[d][(data.dim_vals[d] - choices[point][d]).argmin()])

        data.scores.append(sum_dim.getValue())
    
    return  model_uta, data


class PrometheeI(object):

    def __init__(self, dim: int, choices: np.array, W: np.array, S: np.array):
        self.dim = dim
        self.choices = choices
        self.W = W
        self.S = S

        self.P = []
        self.entering_flow = []
        self.leaving_flow = []
    
    def gaussian_criterion(self, d: np.array, sig=0.5) -> float:
        return 1 - np.e**(-(d**2/(2*sig**2)))

    def get_preferences(self) -> np.array:

        pi = np.zeros((len(self.choices), len(self.choices), self.dim))
        distances = np.zeros((len(self.choices), len(self.choices), self.dim))
        for i in range(len(self.choices)):
            for j in range(len(self.choices)):
                if i == j: continue
                for d in range(self.dim):
                    dist = self.choices[i][d] - self.choices[j][d]
                    distances[i][j][d] = dist
                    if dist < 0: dist = 0
                    pi[i][j][d] = self.W[d]*self.gaussian_criterion(dist, sig=self.S[d])
        self.P = pi.sum(axis=2)
        return self.P

    def get_flows(self) -> Tuple[np.array, np.array]:
        if len(self.P) == 0: 
            self.get_preferences()
        self.leaving_flow  = self.P.sum(axis = 1)
        self.entering_flow = self.P.sum(axis = 0)
        return self.leaving_flow, self.entering_flow

    def is_strongly_preferred(self, a: int, b: int) -> bool:
        return self.entering_flow[a] > self.entering_flow[b]

    def is_strongly_indifferent(self, a: int, b: int) -> bool:
        return self.entering_flow[a] == self.entering_flow[b]

    def is_weakly_preferred(self, a: int, b: int) -> bool:
        return self.leaving_flow[a] < self.leaving_flow[b]

    def is_weakly_indifferent(self, a: int, b: int) -> bool:
        return self.entering_flow[a] == self.entering_flow[b]

    def outranks(self, a: int, b:int) -> bool:
        return (self.is_strongly_preferred(a, b) and self.is_weakly_preferred(a, b)) or (self.is_strongly_preferred(a,b) and self.is_weakly_indifferent(a,b))

    def is_indifferent(self, a: int, b: int) -> bool:
        return self.is_strongly_indifferent(a, b) and self.is_weakly_indifferent(a, b)

    def get_preorder(self) -> Tuple[np.array,np.array]:
        if len(self.entering_flow) == 0:
            self.get_flows()

        partial_preorder_str = np.empty((len(self.choices), len(self.choices)), dtype='U')
        partial_preorder = {}
        for i in range(len(self.choices)):
            partial_preorder[i] = []
            for j in range(len(self.choices)):
                if self.outranks(i, j):
                    partial_preorder_str[i, j] = 'P' 
                    partial_preorder[i].append(j)
                elif self.is_indifferent(i, j):
                    partial_preorder_str[i, j] = 'I'    
                else:
                    partial_preorder_str[i, j] = 'R'    
        self.partial_preorder_str = partial_preorder_str
        self.partial_preorder = partial_preorder
        return self.partial_preorder, self.partial_preorder_str 
    
    
    def plot_preferences(self) -> None:
        fig, ax = plt.subplots()

        ax.matshow(self.P.T, cmap=plt.cm.Blues)

        for i in range(len(self.choices)):
            for j in range(len(self.choices)):
                ax.text(i, j, self.partial_preorder_str[j, i], va='center', ha='center')
        ax.plot()

    def plot_graph(self) -> nx.DiGraph:
        fig, ax = plt.subplots()
        G = nx.DiGraph()

        for node in range(len(self.choices)):
            G.add_node(node+1)

        for node_i in range(len(self.choices)):
            for node_j in self.partial_preorder[node_i]:
                G.add_edge(node_i+1, node_j+1)
        
        pos = nx.circular_layout(G)
        nx.draw(G, pos,with_labels = True)
            
        return G
