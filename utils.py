import json
from typing import List, Tuple

import numpy as np

from gurobipy import Model, GRB, Var, MVar, LinExpr, MLinExpr


class ModelData:
    Nm: int  # nombre de membres
    Nc: int  # nombre de compétences
    Np: int  # nombre de projets
    Nj: int  # délai maximal des projets

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
