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
from utils import Solution

def set_color(val: str, data: Solution):
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
    if '_' not in val: return 
    val = val.split('_')[1]
    switch = {
        data.travail_mapping[proj]: f'background-color: {colors[proj]}'
        for proj in range(data.Np)
    }
    return switch.get(val)

def get_planning_from_solution(data: Solution):
    planning = np.empty((data.Nm, data.Np, data.Nc, data.Nj), dtype=np.int64)
    for i in range(data.Nm):
        for j in range(data.Np):
            for k in range(data.Nc):
                for l in range(data.Nj):
                    planning[i][j][k][l] = data.T[i][j][k][l].X
    return planning

def color_mapping(data: Solution):
    df_colors = {
        data.travail_mapping[proj]: ['x_'+data.travail_mapping[proj]]
        for proj in range(data.Np)
    }
    df_colors = pd.DataFrame(df_colors)
    df_colors = df_colors.style.applymap(set_color)
    return df_colors
                
def create_planning(data: Solution):
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

def create_tables(data: Solution):
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
