import numpy as np
from copy import deepcopy
import itertools

from agents_and_ideas import Agent, Idea
from manager import GraphManager
from agent_generator import AgentGenerator

class Game:
    def __init__(self, N: int, M: int, model='mil1', alpha=2, c1=0.2):
        """
        Инициализация игры

        Args:
            N: количество агентов
            M: количество идей
        """
        self.N = N
        self.M = M
        gen = AgentGenerator(N, M)
        self.agents = gen.generate_random_agents(model=model, alpha=alpha, c1=c1)

    # Вспомогательные функции для анализа
    def analyze_agents(self):
        """
        Анализирует набор агентов
        Args:
            agents: множество агентов
        Returns:
            Словарь с аналитикой
        """
        if not self.agents:
            return {}

        vector_length = self.M
        total_agents = self.N

        # Подсчет единиц по позициям
        position_counts = [0] * vector_length
        for agent in self.agents:
            for i, bit in enumerate(agent.hedges):
                position_counts[i] += bit

        # Средняя плотность
        total_ones = sum(sum(agent.hedges) for agent in self.agents)
        avg_density = total_ones / (total_agents * vector_length)

        # Расстояния Хэмминга
        agents_list = list(self.agents)
        hamming_distances = []
        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                distance = agents_list[i].hamming_distance(agents_list[j])
                hamming_distances.append(distance)

        return {
            'total_agents': total_agents,
            'vector_length': vector_length,
            'avg_density': avg_density,
            'position_densities': [count / total_agents for count in position_counts],
            'avg_hamming_distance': sum(hamming_distances) / len(hamming_distances) if hamming_distances else 0,
            'min_hamming_distance': min(hamming_distances) if hamming_distances else 0,
            'max_hamming_distance': max(hamming_distances) if hamming_distances else 0
        }

    def print_agents_analysis(self):
        """Выводит анализ агентов"""
        analysis = self.analyze_agents()

        print(f"Анализ {analysis['total_agents']} агентов:")
        print(f"  Длина вектора: {analysis['vector_length']}")
        print(f"  Средняя плотность: {analysis['avg_density']:.3f}")
        print(f"  Среднее расстояние Хэмминга: {analysis['avg_hamming_distance']:.2f}")
        print(f"  Мин/макс расстояние Хэмминга: {analysis['min_hamming_distance']}/{analysis['max_hamming_distance']}")
        print(f"  Плотность по позициям: {[f'{d:.2f}' for d in analysis['position_densities']]}")

    def evolve(self):
        system = GraphManager()

        # Добавляем агентов в систему
        system.add_agents(self.agents)

        print("\n" + "=" * 50)
        print("Состояние агентов:")
        for agent in list(self.agents):
            print(f"{agent}")

        print("\n" + "=" * 50)
        print("Пошаговое изменение")
        flag = False
        round = 1
        while flag == False:
            temp_flag = True
            print(f'раунд {round}')
            for agent in list(self.agents):
                print("\n" + "=" * 50)
                print(f"Агент {agent.identifier} думает:")

                print(f"ДО: {agent}")
                strategy_applied = agent.make_best_move()
                if strategy_applied:
                    print(f"ПОСЛЕ: {agent}")
                    temp_flag = False
                    print("\n" + "=" * 50)
                    print("Состояние всех агентов после изменения:")
                    for inner_agent in list(self.agents):
                        print(f"{inner_agent.identifier}, полезность: {inner_agent.U}, вектор: {inner_agent.hedges}")
                    print("\n" + "=" * 50)
                    print("Состояние всех идей после изменения:")
                    all_ideas = system.get_all_ideas()
                    for ididea, idea in all_ideas.items():
                        print(
                            f"{idea.identifier}, степень: {idea.get_deg()}, средняя полезность: {idea.get_average_utility()}")
                else:
                    print("нет улучшающих ходов")
            flag = temp_flag
            round += 1

        print("\n" + "=" * 50)
        print("Типа равновесие:")
        all_agents = system.get_all_agents()
        for inner_agent in list(all_agents):
            print(
                f" Агент {inner_agent.identifier} с полезностью {inner_agent.U} выбирает стратегию {inner_agent.hedges}")
        all_ideas = system.get_all_ideas()
        for ididea, idea in all_ideas.items():
            print(f'Идея {ididea} со степенью {idea.get_deg()}')