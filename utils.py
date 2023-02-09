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

from gurobipy import Model, GRB, Var, MVar, MLinExpr, GurobiError


@dataclass
class Solution:
    Nm: int  # nombre de membres
    Nc: int  # nombre de compétences
    Np: int  # nombre de projets
    Nj: int  # délai maximal des projets

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
        
    # plot vacations
    for p in range(solution.Nm):
        for j in solution.vacations[p]:
            ax.annotate("Vacation", (j, p+0.5+ymargin/2), ha='center', va='center', rotation=90)
    
    for jour in range(solution.Nj+1):
        ax.axvline(jour+0.5, c='gray', lw=0.5)

            
    plt.show()
