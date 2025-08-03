import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.animation as animation

from agents_and_ideas import Agent, Idea

def bip(agents: set[Agent]) -> nx.Graph:
    """
    Преобразует множество агентов в двудольный граф.
    Агенты и идеи — разные доли. Рёбра соединяют агента и идею, если вектор hedges содержит 1.

    Returns:
        G: двудольный граф NetworkX
    """
    G = nx.Graph()

    for agent in agents:
        agent_node = f"A{agent.identifier}"
        G.add_node(agent_node, bipartite=0, type="agent")

        for i, bit in enumerate(agent.hedges):
            if bit == 1:
                idea_node = f"I{i}"
                G.add_node(idea_node, bipartite=1, type="idea")
                G.add_edge(agent_node, idea_node)

    return G

def draw_bip(agents: set[Agent]):
    G = bip(agents)

    # Получим наборы узлов каждой доли
    agent_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    idea_nodes = set(G) - agent_nodes

    # Визуализация с разделением по долям
    pos = dict()
    pos.update((node, (1, i)) for i, node in enumerate(sorted(agent_nodes)))
    pos.update((node, (2, i)) for i, node in enumerate(sorted(idea_nodes)))

    plt.figure(figsize=(15, 10))
    nx.draw(G, pos, with_labels=True, node_color=["skyblue" if n in agent_nodes else "lightgreen" for n in G.nodes()])
    plt.title("Мысли и их Люди")
    plt.show()

def get_bipartite_pos(num_agents, num_ideas):
    # Создаём двудольный граф с числовыми узлами
    B = nx.complete_bipartite_graph(num_agents, num_ideas)

    # Генерируем позиции для этого графа
    pos_raw = nx.spring_layout(B, seed=42)

    # Преобразуем ключи в строковые имена узлов
    pos = {}
    for node, coords in pos_raw.items():
        if node < num_agents:
            pos[f"A{node}"] = coords
        else:
            pos[f"I{node - num_agents}"] = coords

    return pos

def animate(snapshots, edge_changes, filename="animation.mp4", interval=800):
    fig, ax = plt.subplots(figsize=(10, 8))

    num_agents, num_ideas = snapshots[0].shape
    agent_nodes = [f"A{i}" for i in range(num_agents)]
    idea_nodes = [f"I{j}" for j in range(num_ideas)]
    pos = get_bipartite_pos(num_agents, num_ideas)
    #pos = nx.spring_layout(nx.complete_bipartite_graph(len(agent_nodes), len(idea_nodes)), seed=42)
    fig, ax = plt.subplots(figsize=(10, 8))
    def update(frame):
        ax.clear()
        matrix = snapshots[frame]
        G = nx.Graph()

        for i in range(num_agents):
            G.add_node(f"A{i}", bipartite=0)
        for j in range(num_ideas):
            G.add_node(f"I{j}", bipartite=1)

        for i in range(num_agents):
            for j in range(num_ideas):
                if matrix[i][j]:
                    G.add_edge(f"A{i}", f"I{j}")

        # Узлы и рёбра
        nx.draw_networkx_nodes(G, pos, nodelist=agent_nodes, node_color="lightblue", node_size=600, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=idea_nodes, node_color="lightgreen", node_size=600, ax=ax)

        # Подписи: полезность агентов (просто сумма hedges), степень идеи
        labels = {f"A{i}": f"A{i}\n{matrix[i].sum():.2f}" for i in range(num_agents)}
        labels.update({f"I{j}": f"I{j}\n{matrix[:, j].sum()}" for j in range(num_ideas)})
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

        # Обычные рёбра
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray")

        # Подсветка изменённых рёбер
        changed = edge_changes[frame]
        if changed:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=list(changed),
                edge_color="red",
                width=2,
                style="dashed",
                ax=ax
            )

        ax.set_title(f"Step {frame}", fontsize=14)
        ax.axis("off")

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=interval)
    ani.save(filename, writer="ffmpeg")
    plt.close()