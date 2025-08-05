import numpy as np
from copy import deepcopy
import itertools



class Agent:
    """Класс для объектов X"""

    def __init__(self, hedges: list[int], identifier: int = 0, model='mil1', alpha=2, c={'mil10':0.2, 'mil00':0.05, 'mil01':1}):
        """
        Инициализация объекта первого типа

        Args:
            hedges: бинарный вектор принятия идей
            identifier: id агента
            model: mil1, mil10, mil01, mil00, tbd...
            alpha: степень функции
            c: словарь коэффициентов, нужных для каждой модели
        """
        # Проверяем, что вектор действительно бинарный
        if not all(bit in [0, 1] for bit in hedges):
            raise ValueError("Вектор должен содержать только 0 и 1")

        self.hedges = hedges
        self.U = 0
        self.identifier = identifier
        self.M = len(hedges)
        self._system = None
        self.model = model
        self.alpha = alpha
        self.c = c[model]

    def __str__(self):
        return f"Agent(id={self.identifier}, vector={self.hedges}, u={self.U}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return False
        return self.identifier == other.identifier

    @property
    def ideas_dict(self) -> dict[int, 'Idea']:
        """Возвращает словарь идей из системы"""
        if self._system is None:
            raise ValueError("Агент должен быть добавлен в SystemManager")
        return self._system.ideas

    def utility(self):
        """Вычисляет функцию полезности"""
        if self._system is None:
            raise ValueError("Агент должен быть добавлен в GraphManager")
        N = self._system.N
        if self.model == 'mil1':
            total = 0
            for i in range(self.M):
                if self.hedges[i] == 1:
                    if i in self.ideas_dict:
                        deg = self.ideas_dict[i].get_deg()
                        total += deg
            self.U = total
            return total
        elif self.model == 'mil10':
            total = 0
            for i in range(self.M):
                if self.hedges[i] == 1:
                    if i in self.ideas_dict:
                        deg = self.ideas_dict[i].get_deg()
                        total += deg - self.c * deg ** self.alpha - 0.5
            self.U = total
            return total
        elif self.model == 'mil00':
            total = 0
            neighbours = set()
            for i in range(self.M):
                if self.hedges[i] == 1:
                    if i in self.ideas_dict:
                        neighbours.update(self.ideas_dict[i].agents)
                        deg = self.ideas_dict[i].get_deg()
                        total -=  self.c * deg ** self.alpha
            total += len(neighbours)
            self.U = total
            return total
        elif self.model == 'mil01':
            total = 0
            self._system.shortest()
            for i in range(N):
                #print(f"shortest of {self.identifier, i} path is {self._system.dist_matrix[self.identifier, i]}")
                if i == self.identifier:
                    ad = 0
                else:
                    ad = 1/self._system.dist_matrix[self.identifier, i]
                #print(f"ad is {ad}")
                total += ad
            one_indices = [i for i, val in enumerate(self.hedges) if val == 1]
            for i in one_indices:
                if i in self.ideas_dict:
                    total -= self.c
            self.U = total
            return total
        return 0
    def another_util(self, dist_vector):
        total = 0
        if self._system is None:
            raise ValueError("Агент должен быть добавлен в GraphManager")
        N = self._system.N
        for i in range(N):
            if i == self.identifier:
                ad = 0
            else:
                ad = 1 / dist_vector[i]
            # print(f"ad is {ad}")
            total += ad
        one_indices = [i for i, val in enumerate(self.hedges) if val == 1]
        for i in one_indices:
            if i in self.ideas_dict:
                total -= self.c
        self.U = total
        return total
    def simultaneous_move(self, origin_adj):
        if self._system is None:
            raise ValueError("Агент должен быть добавлен в GraphManager")
        #self.utility()
        current_utility = deepcopy(self.U)
        best_move = None
        best_improvement = 0
        origin = deepcopy(self.hedges)
        #origin_adj = self._system.adj_matrix()
        changed = []

        # За 1 ход
        for i in range(self.M):
            new_value = 1 - origin[i]  # меняем 0 на 1 или 1 на 0

            # Временно изменяем значение
            self.hedges[i] = new_value
            self.ideas_dict[i].invert(self)
            adj = self._system.individual_adj(self, origin_adj)
            short = self._system.individual_shortest(self.identifier, adj)
            # Пересчитываем полезность с новым значением
            new_utility = self.another_util(short)
            # print(f"агент {self.identifier}, идея {i}, полезность {self.U}, {new_utility}")
            improvement = new_utility - current_utility

            # Проверяем, лучше ли это изменение
            if improvement > best_improvement:
                best_improvement = improvement
                changed = [i]
                best_move = (changed, improvement)

            self.hedges[i] = 1 - new_value
            self.ideas_dict[i].invert(self)

        # За 2 хода
        zero_indices = [i for i, val in enumerate(origin) if val == 0]
        one_indices = [i for i, val in enumerate(origin) if val == 1]
        for i in one_indices:
            for j in zero_indices:
                # Временно изменяем значение
                self.hedges[i], self.hedges[j] = 0, 1
                self.ideas_dict[i].invert(self)
                self.ideas_dict[j].invert(self)

                adj = self._system.individual_adj(self, origin_adj)
                short = self._system.individual_shortest(self.identifier, adj)
                # Пересчитываем полезность с новым значением
                new_utility = self.another_util(short)
                improvement = new_utility - current_utility

                # Проверяем, лучше ли это изменение
                if improvement > best_improvement:
                    best_improvement = improvement
                    changed = [i, j]
                    best_move = (changed, improvement)

                self.hedges[i], self.hedges[j] = 1, 0
                self.ideas_dict[i].invert(self)
                self.ideas_dict[j].invert(self)
        # print(best_move)
        #self.utility()

        #return best_move
        if best_move is None:
            #print('нет перемен к лучшему')
            return None
        #print(best_move)
        positions, improvement = best_move
        return positions

    def sys_upd(self, positions):
        # Обновляем систему
        for i in positions:
            self.hedges[i] = 1 - self.hedges[i]
            self.ideas_dict[i].invert(self)
    def find_best_move(self) -> tuple[list[int], float] | None:
        """
        Находит лучшее изменение в векторе hedges для улучшения полезности

        Returns:
            tuple: (список изменённых принадлежностей, прирост_полезности) или None если улучшения нет
        """
        if self._system is None:
            raise ValueError("Агент должен быть добавлен в GraphManager")
        self.utility()
        current_utility = deepcopy(self.U)
        best_move = None
        best_improvement = 0
        origin = deepcopy(self.hedges)
        changed = []

        # За 1 ход
        for i in range(self.M):
            new_value = 1 - origin[i]  # меняем 0 на 1 или 1 на 0

            # Временно изменяем значение
            self.hedges[i] = new_value
            self.ideas_dict[i].invert(self)

            # Пересчитываем полезность с новым значением
            new_utility = self.utility()
            #print(f"агент {self.identifier}, идея {i}, полезность {self.U}, {new_utility}")
            improvement = new_utility - current_utility

            # Проверяем, лучше ли это изменение
            if improvement > best_improvement:
                best_improvement = improvement
                changed = [i]
                best_move = (changed, improvement)

            self.hedges[i] = 1 - new_value
            self.ideas_dict[i].invert(self)

        # За 2 хода
        zero_indices = [i for i, val in enumerate(origin) if val == 0]
        one_indices = [i for i, val in enumerate(origin) if val == 1]
        for i in one_indices:
            for j in zero_indices:
                # Временно изменяем значение
                self.hedges[i], self.hedges[j] = 0, 1
                self.ideas_dict[i].invert(self)
                self.ideas_dict[j].invert(self)

                # Пересчитываем полезность с новым значением
                new_utility = self.utility()
                improvement = new_utility - current_utility

                # Проверяем, лучше ли это изменение
                if improvement > best_improvement:
                    best_improvement = improvement
                    changed = [i, j]
                    best_move = (changed, improvement)

                self.hedges[i], self.hedges[j] = 1, 0
                self.ideas_dict[i].invert(self)
                self.ideas_dict[j].invert(self)
        #print(best_move)
        #print(best_improvement)
        self.utility()

        return best_move

    def make_best_move(self) -> bool:
        """
        Выполняет лучшее изменение в векторе, если оно улучшает полезность

        Returns:
            bool: True если изменение было сделано, False если улучшения нет
        """
        best_move = self.find_best_move()

        if best_move is None:
            #print('нет перемен к лучшему')
            return False

        positions, improvement = best_move

        # Делаем изменение
        for i in positions:
            self.hedges[i] = 1 - self.hedges[i]
            self.ideas_dict[i].invert(self)
        prevu = self.U
        # Обновляем систему
        if self._system:
            self._system._update_ideas()
            self._system.update_utilities()
        #print(f'были изменены позиции {positions}, функция полезности возрасла на {self.U - prevu}')
        return True

    def hamming_distance(self, other: 'Agent') -> int:
        """Вычисляет расстояние Хэмминга между двумя бинарными векторами"""
        if len(self.hedges) != len(other.hedges):
            raise ValueError("Векторы должны быть одинаковой длины")

        return sum(a != b for a, b in zip(self.hedges, other.hedges))


class Idea:
    """Объекты из доли Y"""

    def __init__(self, identifier: int, all_agents: set[Agent] = None):
        """
        Инициализация объекта второго типа

        Args:
            identifier: позиция в бинарном векторе (индекс)
            all_agents: ссылка на все объекты первого типа для построения множества
        """
        self.identifier = identifier
        self.agents = set()
        if all_agents is not None:
            self.update_agents(all_agents)

    def update_agents(self, all_agents: set[Agent]):
        """Обновляет множество агентов"""
        self.agents = self._set_agents(all_agents)

    def invert(self, agent):
        if agent in self.agents:
            self.agents.remove(agent)
        else:
            self.agents.add(agent)

    def _set_agents(self, all_agents: set[Agent]) -> set[Agent]:
        """Возвращает множество объектов первого типа, у которых в позиции identifier стоит 1"""
        result = set()
        for obj in all_agents:
            # Проверяем, что индекс в пределах вектора и значение равно 1
            if (self.identifier < len(obj.hedges) and
                    obj.hedges[self.identifier] == 1):
                result.add(obj)
        return result

    def get_deg(self) -> int:
        """Возвращает количество объектов в множестве"""
        return len(self.agents)

    def get_average_utility(self) -> tuple:
        """Возвращает средние значения функции полезности по смежным агентам"""
        objects = self.agents
        if not objects:
            return 0
        total = sum(obj.U for obj in self.agents if obj.U is not None)
        count = len([obj for obj in self.agents if obj.U is not None])

        return total / count if count > 0 else 0

    def __str__(self):
        return f"IndexObject(id={self.identifier}, objects_count={self.get_deg()})"

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        """Делает объект итерируемым по содержащимся объектам первого типа"""
        return iter(self.agents)
