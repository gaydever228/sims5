import numpy as np
from copy import deepcopy
import itertools

from agents_and_ideas import Agent, Idea
from manager import GraphManager
from agent_generator import AgentGenerator
from game import Game

game = Game(100, 200, model = 'mil11', c1 = 0.1)

game.print_agents_analysis()
game.evolve()