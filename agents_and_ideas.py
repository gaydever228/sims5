import numpy as np
from copy import deepcopy
import itertools



class Agent:
    """Класс для объектов X"""

    def __init__(self, hedges: list[int], identifier: int = 0, model='mil1', alpha=2, c1=0.2):
        """
        Инициализация объекта первого типа
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
        self.c1 = c1

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
        if self.model == 'mil1':
            total = 0
            for i in range(self.M):
                if self.hedges[i] == 1:
                    if i in self.ideas_dict:
                        deg = self.ideas_dict[i].get_deg()
                        total += deg
            self.U = total
            return total
        elif self.model == 'mil11':
            total = 0
            for i in range(self.M):
                if self.hedges[i] == 1:
                    if i in self.ideas_dict:
                        deg = self.ideas_dict[i].get_deg()
                        total += deg - self.c1 * deg ** self.alpha - 0.5
            self.U = total
            return total

        return 0

    def find_best_move(self) -> tuple[list[int], float] | None:
        """
        Находит лучшее изменение в векторе hedges для улучшения полезности

        Returns:
            tuple: (список изменённых принадлежностей, прирост_полезности) или None если улучшения нет
        """
        if self._system is None:
            raise ValueError("Агент должен быть добавлен в GraphManager")

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
            self.ideas_dict[i].invert(self.identifier)

            # Пересчитываем полезность с новым значением
            new_utility = self.utility()
            improvement = new_utility - current_utility

            # Проверяем, лучше ли это изменение
            if improvement > best_improvement:
                best_improvement = improvement
                changed = [i]
                best_move = (changed, improvement)

            self.hedges[i] = 1 - new_value
            self.ideas_dict[i].invert(self.identifier)

        # За 2 хода
        zero_indices = [i for i, val in enumerate(origin) if val == 0]
        one_indices = [i for i, val in enumerate(origin) if val == 1]
        for i in one_indices:
            for j in zero_indices:
                # Временно изменяем значение
                self.hedges[i], self.hedges[j] = 0, 1
                self.ideas_dict[i].invert(self.identifier)
                self.ideas_dict[j].invert(self.identifier)

                # Пересчитываем полезность с новым значением
                new_utility = self.utility()
                improvement = new_utility - current_utility

                # Проверяем, лучше ли это изменение
                if improvement > best_improvement:
                    best_improvement = improvement
                    changed = [i, j]
                    best_move = (changed, improvement)

                self.hedges[i], self.hedges[j] = 1, 0
                self.ideas_dict[i].invert(self.identifier)
                self.ideas_dict[j].invert(self.identifier)
        print(best_move)

        return best_move

    def make_best_move(self) -> bool:
        """
        Выполняет лучшее изменение в векторе, если оно улучшает полезность

        Returns:
            bool: True если изменение было сделано, False если улучшения нет
        """
        best_move = self.find_best_move()

        if best_move is None:
            print('нет перемен к лучшему')
            return False

        positions, improvement = best_move

        # Делаем изменение
        for i in positions:
            self.hedges[i] = 1 - self.hedges[i]
            self.ideas_dict[i].invert(self.identifier)

        # Обновляем систему
        if self._system:
            self._system._update_ideas()
            self._system.update_utilities()
        print(f'были изменены позиции {positions}, функция полезности возрасла на {improvement}')
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
        if all_agents is not None:
            self.update_agents(all_agents)

    def update_agents(self, all_agents: set[Agent]):
        """Обновляет множество агентов"""
        self.agents = self._set_agents(all_agents)

    def invert(self, agent_id):
        if agent_id in self.agents:
            self.agents.remove(agent_id)
        else:
            self.agents.add(agent_id)

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
