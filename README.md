# Bienvenue chez CompuOpti 🤓
## Membres du groupe

- Fernando KURIKE MATSUMOTO
- José Lucas DE MELO COSTA
- Victor Felipe DOMINGUES DO AMARALRAL

## Aperçu du projet 🤔

Vous êtes ingénieur-développeur chez CompuOpti, une entreprise qui aide ses clients à optimiser leurs décisions grâce à ses solutions. Chez CompuOpti, nous aimons que notre personnel soit affecté aux projets de manière efficace 💻🤖

Et c'est là que vous entrez en jeu! Nous avons besoin de votre aide pour planifier les affectations de personnel pour les projets de nos clients.

Nous avons besoin d'affecter nos ingénieurs-développeurs sur des projets spécifiques qui nécessitent des compétences différentes (optimisation, gestion de projet, développement web, etc.). Par exemple, un projet peut nécessiter 6 jours/personne de compétences A, 2 jours/personne de compétences B, et 5 jours/personne de compétences C.

Chaque membre de notre personnel a ses propres compétences, issues d'un ensemble donné (par exemple, {A, B, C, D, E}), ainsi que des jours de congé prédéfinis pendant l'horizon de temps considéré.

De plus, chaque projet a une date de livraison négociée avec le client, et il est important de ne pas la dépasser, sinon des pénalités financières par jour de retard seront inscrites dans le contrat.

Notre PDG, Margaux Dourtille, cherche à maximiser le bénéfice total des projets réalisés. Et il y a plusieurs critères à prendre en compte:

- Maximiser le résultat financier de l'entreprise en maximisant le bénéfice, incluant les pénalités éventuelles
- Minimiser le nombre de projets sur lesquels un collaborateur est affecté, afin que les employés n'aient pas à changer de projet trop souvent.

## Modélisation 

Ce projet concerne les systèmes de décision. Les entrées comprennent :
- Le nombre de personnes (membres) à faire travailler, $N_m$.
- Le nombre de compétences possibles, $N_c$.
- Le nombre de projets, $N_p$.
- La durée maximale du projet, $N_j$.

L'affectation des compétences est modélisée par $H \in \mathbb{B}^{N_m\times N_c}$, où $H_{i,j}=1$ si la personne $i$ possède la compétence $j$ et $0$ sinon.

Les vacances sont également modélisées sous la forme d'une matrice $C \in \mathbb{B}^{N_m \times N_j}$, où $C_{i,\ell}=1$ si la personne $i$ est en vacances le jour $\ell$ et $0$ sinon.

Les exigences du projet sont modélisées par $\text{Nc}$. \in \mathbb{N}^{N_p \times N_c}$, avec $\text{Nc}_{j,k} = c_i$, où $c_i \in \{0, \dots, N_c\}$ est le nombre de compétences de type $k$ dont le projet $j$ a besoin.

Le revenu d'un projet $j$, $\text{Rev}_j \in \mathbb{R}$ est également une entrée.

L'objectif de ce projet est de déterminer l'affectation optimale des personnes aux projets afin de maximiser les revenus. Les contraintes sont les suivantes :
- Chaque projet doit avoir le nombre requis de compétences
- Chaque personne ne peut travailler que sur un seul projet à la fois.
- Chaque personne ne peut travailler sur des projets que pendant ses jours de disponibilité (pas pendant les vacances).

## Solution
La solution consiste à résoudre un problème d'optimisation en utilisant des techniques de programmation mathématique.

## Exigences
- Connaissance de la programmation mathématique et des techniques d'optimisation
- Familiarité avec la programmation en python

## Exécution du code
1. Cloner le référentiel
2. Exécuter le script principal 

## Résultats
Le résultat de ce code sera l'affectation optimale des personnes aux projets, ainsi que le revenu maximum atteint.

## Contact
Pour toute question ou problème, veuillez contacter l'un des membres du groupe.
