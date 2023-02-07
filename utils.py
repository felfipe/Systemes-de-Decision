import json
from typing import List, Tuple

import numpy as np
import pandas as pd

from gurobipy import Model, GRB, Var, MVar, LinExpr, MLinExpr, quicksum
import networkx as nx


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


class ModelPreordreData:
    dim:     int                    # Nombre de dimensions = nombre de criteres
    epsilon: int                    # Set to relax constraints
    nombreChoix: int                # Nombre de choix (solutions non dominée)
    
    Poids: dict[int, MVar]                     # Poids à attribuer à chaque critere
    SommePonderee: dict[int, MLinExpr]             # Somme ponderee pour cree un score
    PreOrdre: dict[int, int]

def get_preorder(choices: list, preference_list: list[tuple]):

    model_data = ModelPreordreData()
    model_preorder = Model("Preorder")

    model_data.dim = 3
    model_data.epsilon = 0
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

def create_graph(data: ModelPreordreData, draw:bool = False):
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