import numpy as np
from copy import deepcopy
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall, shortest_path


from agents_and_ideas import Agent, Idea



class GraphManager:
    """Общий класс-контейнер для управления всеми агентами и идеями"""

    def __init__(self):
        self.agents: set[Agent] = set()
        self.ideas: dict[int, Idea] = {}
        self.N = None
        self.M = None

    def add_agent(self, agent: 'Agent'):
        """Добавляет агента в систему"""
        self.agents.add(agent)
        agent._system = self  # Даем агенту ссылку на систему
        self._update_ideas()
        self.update_utilities()

    def add_agents(self, agents: set['Agent']):
        """Добавляет множество агентов"""
        for agent in agents:
            agent._system = self
        self.agents.update(agents)
        self.N = len(self.agents)
        self._update_ideas()
        self.shortest()
        self.update_utilities()

    def _update_ideas(self):
        """Обновляет все идеи после изменения агентов"""
        if not self.agents:
            return

        # Определяем максимальную длину вектора hedges
        max_length = max(len(agent.hedges) for agent in self.agents)

        # Создаем или обновляем идеи для каждой позиции
        for i in range(max_length):
            if i not in self.ideas:
                self.ideas[i] = Idea(i, self.agents)
            else:
                self.ideas[i].update_agents(self.agents)
        self.M = len(self.ideas)
    def update_utilities(self):
        """
        Обновляет поле U у всех агентов значениями рассчитанной полезности
        """
        self.shortest()
        for agent in self.agents:
            agent.U = agent.utility()
    def adj_matrix(self):
        """Возвращает матрицу смежности агентов с весом ребра = степень идеи"""
        matrix = np.zeros((self.N, self.N))
        matrix += self.N*2
        done_agents = set()
        for agent in self.agents:
            done_agents.add(agent)
            one_indices = [i for i, val in enumerate(agent.hedges) if val == 1]
            for i in one_indices:
                weight = len(self.ideas[i].agents)
                for inner_agent in self.ideas[i].agents.difference(done_agents):
                    if weight < matrix[agent.identifier, inner_agent.identifier] or weight < matrix[inner_agent.identifier, agent.identifier]:
                        matrix[agent.identifier, inner_agent.identifier] = weight
                        matrix[inner_agent.identifier, agent.identifier] = weight
        for i in range(self.N):
            for j in range(self.N):
                if matrix[i, j] == self.N * 2:
                    matrix[i, j] = np.inf

        #print(matrix)
        #self.matrix = matrix
        return matrix
    def individual_adj(self, agent: Agent, matrix = None):
        """Возвращает матрицу смежности агентов с весом ребра = степень идеи"""
        if matrix is None:
            matrix = self.adj_matrix()
        vector = np.zeros(self.N)
        vector += self.N * 2
        one_indices = [i for i, val in enumerate(agent.hedges) if val == 1]
        for i in one_indices:
            weight = len(self.ideas[i].agents)
            #print(f"IDEAS {self.ideas[i].agents}")
            for inner_agent in self.ideas[i].agents.difference({agent}):
                if weight < vector[inner_agent.identifier]:
                    vector[inner_agent.identifier] = weight
        for i in range(self.N):
            if vector[i] == self.N * 2:
                vector[i] = np.inf
        matrix[agent.identifier][:] = vector
        matrix[:][agent.identifier] = vector
        #print(vector)
        # self.matrix = matrix
        return matrix
    def shortest(self):
        """находит кратчайшие пути для каждой пары агентов, записывает матрицу расстояний в self.dist_matrix"""
        adj = self.adj_matrix()
        dist_sp = shortest_path(adj, method='auto', directed=False)
        self.dist_matrix = dist_sp
        #return dist_sp
    def individual_shortest(self, agent_id, adj = None):
        """находит кратчайшие пути для конкретного агента, возвращает вектор?"""
        if adj is None:
            adj = self.adj_matrix()
        dist_sp = shortest_path(adj, method='auto', directed=False, indices=agent_id)
        #print(dist_sp)
        return dist_sp

    def get_idea(self, identifier: int) -> 'Idea':
        """Возвращает идею по идентификатору"""
        if identifier not in self.ideas:
            self.ideas[identifier] = Idea(identifier, self.agents)
        return self.ideas[identifier]

    def get_all_agents(self) -> set['Agent']:
        """Возвращает всех агентов"""
        return self.agents.copy()

    def get_all_ideas(self) -> dict[int, 'Idea']:
        """Возвращает все идеи"""
        return self.ideas.copy()