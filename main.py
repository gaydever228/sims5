import numpy as np
from copy import deepcopy
import itertools

from agents_and_ideas import Agent, Idea
from manager import GraphManager
from agent_generator import AgentGenerator
from game import Game
from visual import draw_bip, animate
N = 10
M = 4
coefs ={'mil1':1, 'mil10':0.2, 'mil00':0.05, 'mil01':1/N}
mod = 'mil00'
game = Game(40, 20, model = mod, c = coefs, dens = 0.1, method='erdos')
#draw_bip(game.agents)

snapshots, edge_changes, utilities, eq = game.evolve_anim_by_one()
#snapshots, edge_changes, utilities, eq = game.evolve_sim()
animate(snapshots, edge_changes, utilities, eq, filename="agent_evolution_" + mod + ".mp4", interval=1000)

#game.print_agents_analysis()
#game.evolve()