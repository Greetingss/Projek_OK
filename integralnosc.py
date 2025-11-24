import networkx as nx
import random
import numpy as np

def generate_random_greedy_connected_graph(n, k):
    """
    Generuje spójny graf z n wierzchołkami i k krawędziami
    przy użyciu zachłannego losowego algorytmu.
    """
    if k < n-1 or k > n*(n-1)//2:
        raise ValueError("Nieprawidłowa liczba krawędzi dla spójnego grafu")

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Tworzymy drzewo rozpinające
    nodes = list(G.nodes)
    random.shuffle(nodes)
    for i in range(1, n):
        j = random.randint(0, i-1)
        G.add_edge(nodes[i], nodes[j])

    # Dodajemy pozostałe krawędzie
    remaining_edges = k - (n-1)
    # all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n) if not G.has_edge(u,v)]
    
    all_possible_edges = []

    for u in range(n):
        for v in range(u + 1, n):
            if not G.has_edge(u, v): 
                all_possible_edges.append((u, v)) 

    random.shuffle(all_possible_edges)

    for edge in all_possible_edges:
        if remaining_edges == 0:
            break
        G.add_edge(*edge)
        remaining_edges -= 1

    return G

def is_integral_graph(G, tol=1e-8):
    """Sprawdza, czy graf jest integralny (wszystkie wartości własne całkowite)."""
    adj_matrix = nx.adjacency_matrix(G).todense()
    eigenvalues = np.linalg.eigvals(adj_matrix)
    
    for eig in eigenvalues:
        # Sprawdzamy, czy część urojona jest prawie zerowa
        if abs(eig.imag) > tol:
            return False
        # Sprawdzamy, czy część rzeczywista jest całkowita
        if not np.isclose(eig.real, round(eig.real), atol=tol):
            return False
    return True

def adjacency_hash(G):
    """Tworzy hash macierzy sąsiedztwa grafu, aby łatwo sprawdzić duplikaty."""
    adj_matrix = nx.adjacency_matrix(G).todense()
    # Zamieniamy na tuple, żeby hash był stabilny
    return hash(tuple(map(int, adj_matrix.flatten())))

def generate_integral_connected_graph(n, k, max_attempts=100000):
    """
    Próbuje wygenerować spójny graf całkowity.
    max_attempts ogranicza liczbę prób losowania.
    """
    attempt = 0
    seen_graphs = set()
    
    while True:
        
    # for attempt in range(max_attempts):
        G = generate_random_greedy_connected_graph(n, k)
        h = adjacency_hash(G)
        if h in seen_graphs:
            continue  # pomijamy powtórki
        seen_graphs.add(h)

        if is_integral_graph(G):
            print(f"Znaleziono graf całkowity po {attempt+1} próbach!")
            return G
    raise Exception(f"Nie znaleziono grafu całkowitego w {max_attempts} próbach.")

# --- Przykład użycia ---
n = 5   # liczba wierzchołków
k = 10   # liczba krawędzi (spójny graf musi mieć przynajmniej n-1 krawędzi)
G_integral = generate_integral_connected_graph(n, k)

print(f"Liczba wierzchołków: {G_integral.number_of_nodes()}")
print(f"Liczba krawędzi: {G_integral.number_of_edges()}")
print(f"Czy graf jest spójny? {nx.is_connected(G_integral)}")
print(f"Czy graf jest całkowity? {is_integral_graph(G_integral)}")
print("Lista krawędzi:")
print(list(G_integral.edges()))
