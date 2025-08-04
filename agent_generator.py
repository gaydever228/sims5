import numpy as np
from copy import deepcopy
import itertools

from agents_and_ideas import Agent, Idea
from manager import GraphManager


class AgentGenerator:
    """Класс для генерации агентов для тестирования"""

    def __init__(self, N, M, seed=None):
        """
        Инициализация генератора
        Args:
            seed: зерно для воспроизводимости результатов
        """
        if seed is not None:
            np.random.seed(seed)
        self.N = N
        self.M = M

    def generate_random_agents(self, model='mil1', alpha=2, c={'mil1':1, 'mil10':0.2, 'mil00':0.05, 'mil01':1}):
        """
        Генерирует случайных агентов
        Args:
            model: модель расчета полезности
            alpha: параметр альфа для модели mil10
            c1: Множитель стоимости
        Returns:
            Множество агентов
        """
        agents = set()
        for i in range(self.N):
            hedges = np.random.randint(0, 2, self.M).tolist()
            agent = Agent(hedges, i, model, alpha, c[model])
            agents.add(agent)
        return agents

    def generate_uniform_density_agents(self, density=0.5, model='mil1', alpha=2, c={'mil1':1, 'mil10':0.2, 'mil00':0.05, 'mil01':1}):
        """
        Генерирует агентов с заданной плотностью единиц
        Args:
            count: количество агентов
            vector_length: длина бинарного вектора
            density: плотность единиц (от 0 до 1)
            model: модель расчета полезности
            alpha: параметр альфа для модели mil10
            c1: множитель стоимости
        Returns:
            Множество агентов
        """
        agents = set()
        ones_count = int(self.M * density)

        for i in range(self.N):
            hedges = [0] * self.M
            # Случайно выбираем позиции для единиц
            positions = np.random.choice(self.M, ones_count, replace=False)
            for pos in positions:
                hedges[pos] = 1

            agent = Agent(hedges, i, model, alpha, c[model])
            agents.add(agent)
        return agents

    def generate_structured_agents(self, pattern_type='clusters', model='mil1', alpha=2, c={'mil1':1, 'mil10':0.2, 'mil00':0.05, 'mil01':1}):
        """
        Генерирует агентов со структурированными паттернами
        Args:
            count: количество агентов
            vector_length: длина бинарного вектора
            pattern_type: тип паттерна ('clusters', 'alternating', 'blocks')
            model: модель расчета полезности
            alpha: параметр альфа для модели mil10
        Returns:
            Множество агентов
        """
        agents = set()

        for i in range(self.N):
            if pattern_type == 'clusters':
                hedges = self._generate_cluster_pattern(self.M)
            elif pattern_type == 'alternating':
                hedges = self._generate_alternating_pattern(self.M, i)
            elif pattern_type == 'blocks':
                hedges = self._generate_block_pattern(self.M, i)
            else:
                hedges = np.random.randint(0, 2, self.M).tolist()

            agent = Agent(hedges, i, model, alpha, c[model])
            agents.add(agent)

        return agents

    def generate_similar_agents(self, base_agent, max_hamming_distance=1, model = 'mil1', c={'mil1':1, 'mil10':0.2, 'mil00':0.05, 'mil01':1}):
        """
        Генерирует агентов, похожих на базового агента
        Args:
            base_agent: базовый агент
            count: количество новых агентов
            max_hamming_distance: максимальное расстояние Хэмминга от базового агента
        Returns:
            Множество агентов
        """
        agents = set()
        vector_length = len(base_agent.hedges)

        # Генерируем всех агентов в пределах заданного расстояния Хэмминга
        candidates = []
        for distance in range(1, max_hamming_distance + 1):
            for positions in itertools.combinations(range(vector_length), distance):
                new_hedges = base_agent.hedges.copy()
                for pos in positions:
                    new_hedges[pos] = 1 - new_hedges[pos]
                candidates.append(new_hedges)

        # Выбираем случайных кандидатов
        if len(candidates) > self.N - 1:
            selected_indices = np.random.choice(len(candidates), self.N - 1, replace=False)
            selected = [candidates[i] for i in selected_indices]
        else:
            selected = candidates

        for i, hedges in enumerate(selected):
            agent = Agent(hedges, base_agent.identifier + i + 100,
                          base_agent.model, base_agent.alpha, c = c[model])
            agents.add(agent)

        return agents

    def generate_normal_distribution_agents(self, mean_density=0.5, std=0.2, model='mil1', alpha=2, c={'mil1':1, 'mil10':0.2, 'mil00':0.05, 'mil01':1}):
        """
        Генерирует агентов с плотностью, распределенной по нормальному закону
        Args:
            count: количество агентов
            vector_length: длина бинарного вектора
            mean_density: средняя плотность
            std: стандартное отклонение
            model: модель расчета полезности
            alpha: параметр альфа для модели mil10
        Returns:
            Множество агентов
        """
        agents = set()
        densities = np.random.normal(mean_density, std, self.N)
        densities = np.clip(densities, 0, 1)  # Ограничиваем от 0 до 1

        for i, density in enumerate(densities):
            ones_count = int(self.M * density)
            hedges = [0] * self.M

            if ones_count > 0:
                positions = np.random.choice(self.M, ones_count, replace=False)
                for pos in positions:
                    hedges[pos] = 1

            agent = Agent(hedges, i, model, alpha, c1)
            agents.add(agent)

        return agents

    def generate_beta_distribution_agents(self, alpha_param=2, beta_param=2, model='mil1', alpha=2, c={'mil1':1, 'mil10':0.2, 'mil00':0.05, 'mil01':1}):
        """
        Генерирует агентов с плотностью, распределенной по бета-распределению
        Args:
            count: количество агентов
            vector_length: длина бинарного вектора
            alpha_param: параметр альфа бета-распределения
            beta_param: параметр бета бета-распределения
            model: модель расчета полезности
            alpha: параметр альфа для модели mil10
        Returns:
            Множество агентов
        """
        agents = set()
        densities = np.random.beta(alpha_param, beta_param, self.N)

        for i, density in enumerate(densities):
            ones_count = int(self.M * density)
            hedges = [0] * self.M

            if ones_count > 0:
                positions = np.random.choice(self.M, ones_count, replace=False)
                for pos in positions:
                    hedges[pos] = 1

            agent = Agent(hedges, i, model, alpha, c[model])
            agents.add(agent)

        return agents

    def _generate_cluster_pattern(self):
        """Генерирует паттерн с кластерами единиц"""
        hedges = [0] * self.M
        cluster_size = np.random.randint(2, max(3, self.M // 3))
        start_pos = np.random.randint(0, self.M - cluster_size + 1)

        for i in range(start_pos, start_pos + cluster_size):
            hedges[i] = 1

        return hedges

    def _generate_alternating_pattern(self, agent_idx):
        """Генерирует чередующийся паттерн"""
        start_bit = agent_idx % 2
        return [(i + start_bit) % 2 for i in range(self.M)]

    def _generate_block_pattern(self, agent_idx):
        """Генерирует блочный паттерн"""
        block_size = max(2, self.M // 4)
        hedges = [0] * self.M

        # Определяем начало блока на основе индекса агента
        start_block = (agent_idx * block_size) % self.M
        end_block = min(start_block + block_size, self.M)

        for i in range(start_block, end_block):
            hedges[i] = 1

        return hedges