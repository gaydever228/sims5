import numpy as np
from copy import deepcopy
import itertools

from agents_and_ideas import Agent, Idea
from manager import GraphManager
from agent_generator import AgentGenerator
from game import Game
from visual import draw_bip, animate

game = Game(20, 30, model = 'mil11', c1 = 0.15)
#draw_bip(game.agents)

snapshots, edge_changes = game.evolve_anim()
animate(snapshots, edge_changes, filename="agent_evolution.mp4", interval=800)

#game.print_agents_analysis()
#game.evolve()