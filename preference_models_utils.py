import json
from typing import List, Tuple

import numpy as np
import pandas as pd

from gurobipy import Model, GRB, Var, MVar, LinExpr, MLinExpr, quicksum

import networkx as nx
from typing import Tuple
import matplotlib.pyplot as plt


class ModelData:
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

    f1: MLinExpr
    f2: MLinExpr
    f3: LinExpr


def create_model(instance_path: str) -> Tuple[Model, ModelData]:
    with open(instance_path) as f:
        instance = json.load(f)
    

    d = ModelData()
    d.qualifications = list(instance["qualifications"])
    d.staff = [person["name"] for person in instance["staff"]]
    d.nom_mapping = {x: instance['staff'][x]['name'] for x in range(len(instance['staff']))}
    d.travail_mapping = {x:instance['jobs'][x]['name'] for x in range(len(instance['jobs']))}
    d.qualifications_mapping = {x:instance['qualifications'][x] for x in range(len(instance['qualifications']))}
    d.jour_mapping = np.arange(instance['horizon'])+1

    # Tailles des données
    d.Nm = len(instance["staff"])
    d.Nc = len(d.qualifications)
    d.Np = len(instance["jobs"])
    d.Nj = instance["horizon"]

    # Création des matrices d'entrée
    d.H = np.zeros((d.Nm, d.Nc), dtype=np.int32)
    d.C = np.zeros((d.Nm, d.Nj), dtype=np.int32)
    d.Q = np.zeros((d.Np, d.Nc), dtype=np.int32)
    d.Rev = np.zeros(d.Np, dtype=np.int32)
    d.P = np.zeros(d.Np, dtype=np.int32)
    d.Dl = np.zeros(d.Np, dtype=np.int32)

    for i, person in enumerate(instance["staff"]):
        for qualif in person["qualifications"]:
            k = d.qualifications.index(qualif)
            d.H[i, k] = 1
        for vacation in person["vacations"]:
            d.C[i, vacation - 1] = 1

    for j, job in enumerate(instance["jobs"]):
        for qualif, days in job["working_days_per_qualification"].items():
            k = d.qualifications.index(qualif)
            d.Q[j, k] = days
        d.Rev[j] = job["gain"]
        d.P[j] = job["daily_penalty"]
        d.Dl[j] = job["due_date"]

    # Création du modèle Gurobi
    m = Model("CompuOpti")

    # Variables de decision
    d.T = m.addMVar((d.Nm, d.Np, d.Nc, d.Nj), vtype=GRB.BINARY, name="T")
    d.R = m.addMVar(d.Np, vtype=GRB.BINARY, name="Realise")
    d.De = m.addMVar(d.Np, lb=1, ub=d.Nj, vtype=GRB.INTEGER, name="De")
    d.F = m.addMVar(d.Np, lb=1, ub=d.Nj, vtype=GRB.INTEGER, name="F")
    d.Re = m.addMVar(d.Np, lb=0, ub=d.Nj-1, vtype=GRB.INTEGER, name="Re")
    d.Dm = m.addVar(lb=0, ub=d.Nj, vtype=GRB.INTEGER, name="Dm")
    d.Af = m.addMVar((d.Nm, d.Np), vtype=GRB.BINARY, name="Af")

    # Contrainte de qualification
    for j in range(d.Np):
        for ell in range(d.Nj):
            m.addConstr(d.T[:, j, :, ell] <= d.H)

    # Contrainte d’unicité de l’affectation
    for i in range(d.Nm):
        for ell in range(d.Nj):
            m.addConstr(d.T[i, :, :, ell].sum() <= 1)

    # Contrainte de congé
    for j in range(d.Np):
        for k in range(d.Nc):
            m.addConstr(d.T[:, j, k, :] <= 1 - d.C)

    # Contraintes d’unicité de la réalisation d’un projet et de couverture des qualifications
    for j in range(d.Np):
        for k in range(d.Nc):
            m.addConstr(d.T[:, j, k, :].sum() == d.R[j] * d.Q[j, k])

    # Contraintes sur la variable Af
    for k in range(d.Nc):
        for ell in range(d.Nj):
            m.addConstr(d.Af >= d.T[:, :, k, ell])

    # Contraintes sur la durée d’un projet
    for i in range(d.Nm):
        for k in range(d.Nc):
            for ell in range(d.Nj):
                m.addConstr(d.De <= (ell + 1) * d.T[i, :, k, ell] + d.Nj * (1 - d.T[i, :, k, ell]))
                m.addConstr(d.F >= (ell + 1) * d.T[i, :, k, ell])
    m.addConstr(d.Re >= d.F - d.Dl)
    m.addConstr(d.Dm >= d.F - d.De + d.R)

    # Fonctions objectifs
    d.f1 = -(d.Rev.T @ d.R - d.P.T @ d.Re)
    d.f2 = d.Af.sum()
    d.f3 = d.Dm * 1

    m.setObjective(d.f1, GRB.MINIMIZE)

    return m, d

def get_non_dominated(m: Model, data: ModelData):
    solutions = []
    m.params.outputflag = 0
    m.NumScenarios = (data.Nj+1)*(data.Nm*data.Np + 1)

    c1 = m.addConstr(data.f2 == 0)
    c2 = m.addConstr(data.f3 == 0)

    for i in range(data.Nj+1):
        for j in range(data.Nm*data.Np+1):
            m.params.ScenarioNumber = i*(data.Nm*data.Np+1) + j
            m.ScenNName = 'i = {}, j = {}'.format(i, j)
            c1.ScenNRhs = j
            c2.ScenNRhs = i

    print("Starting optimzation...")

    m.setObjective(data.f1)
    m.reset()
    m.optimize()

    print("done.")

    for s in range(m.NumScenarios):
        m.params.ScenarioNumber = s
        solutions.append([m.ScenNObjVal, data.Af.ScenNX.sum(), data.Dm.ScenNX])

    m.remove(c1)
    m.remove(c2)

    solutions = np.array(solutions)
    solutions = solutions.round().astype(int)

    filtered_solutions = []
    scenario = []

    for i in range(solutions.shape[0]):
        non_dominated = True
        for j in range(solutions.shape[0]):
            if i == j:
                continue
            if dominate(solutions[j], solutions[i]):
                non_dominated = False        
                break
        if non_dominated:
            scenario.append(i)
            filtered_solutions.append(solutions[i])

    return filtered_solutions, scenario 
        

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
    
    Poids: dict[int, MVar]                     # Poids à attribuer à chaque critere
    SommePonderee: dict[int, MLinExpr]             # Somme ponderee pour cree un score
    PreOrdre: dict[int, int]

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