import numpy as np
from copy import deepcopy
import itertools

from agents_and_ideas import Agent, Idea



class GraphManager:
    """Общий класс-контейнер для управления всеми агентами и идеями"""

    def __init__(self):
        self.agents: set[Agent] = set()
        self.ideas: dict[int, Idea] = {}

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
        self._update_ideas()
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

    def update_utilities(self):
        """
        Обновляет поле U у всех агентов значениями рассчитанной полезности

        Args:
            method: метод расчета полезности
        """
        for agent in self.agents:
            agent.U = agent.utility()

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