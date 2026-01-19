import networkx as nx
import random
import numpy as np
import math
import sys
import time
import subprocess

 
def generate_graph(n, k):
 
    if k < n-1 or k > n*(n-1)//2:
        raise ValueError("Nieprawidłowa liczba krawędzi")
 
    G = nx.Graph()
    G.add_nodes_from(range(n))
 
    nodes = list(G.nodes)
    random.shuffle(nodes)
    for i in range(1, n):
        j = random.randint(0, i-1)
        G.add_edge(nodes[i], nodes[j])
 
    remaining_edges = k - (n-1)
 
    all_possible_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not G.has_edge(u, v):
                all_possible_edges.append((u, v))
 
    random.shuffle(all_possible_edges)
 
    for edge in all_possible_edges:
        if remaining_edges == 0:
            break
        G.add_edge(edge[0], edge[1])
        remaining_edges -= 1
 
    return G
 
def integral_cost(G):
    adj = nx.adjacency_matrix(G).todense()
    eigvals = np.linalg.eigvals(adj)
 
    cost = 0
    for eig in eigvals:
        if abs(eig.imag) > 1e-9:
            cost += abs(eig.imag)
        cost += abs(eig.real - round(eig.real))
 
    return float(cost)
 
def random_graph_neighbor(G):
 
    n = G.number_of_nodes()
 
    edges = list(G.edges())
    non_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not G.has_edge(u, v):
                non_edges.append((u, v))
 
    if not edges or not non_edges:
        return None
 
    e_remove = random.choice(edges)
    e_add = random.choice(non_edges)
    G2 = G.copy()
    G2.remove_edge(*e_remove)
    G2.add_edge(*e_add)
 
    if not nx.is_connected(G2):
        return None
 
    return G2
 
def search(n, k, t, T_start=1.0, alpha=0.995, stagnation_limit=500):
    st = time.time()

    while True:
        if t is not None and time.time() - st > t:
            print(f"\nPrzekroczono limit czasu {t} sekund. Kończę działanie.")
            return 
        G = generate_graph(n, k)
        best_G = G.copy()
        cost = integral_cost(G)
        best_cost = cost
        T = T_start
        stagnation_counter = 0
 
        print(f"\nNowy start: start cost = {cost:.5f}")
 
        iter_count = 0
        while True:
            iter_count += 1
            candidates = []
            for _ in range(10):
                neighbor = random_graph_neighbor(G)
                if neighbor is None:
                    continue
                c = integral_cost(neighbor)
                candidates.append((neighbor, c))
            if not candidates:
                continue
 
            candidates.sort(key=lambda x: x[1])
            top3 = candidates[:3]
            G_new, cost_new = random.choice(top3)
 
            delta = cost_new - cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                G = G_new
                cost = cost_new
 
            if cost < best_cost:
                best_cost = cost
                best_G = G.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
 
            T *= alpha
 
            if iter_count % 200 == 0:
                print(f"Iter {iter_count}: cost = {cost:.5f}, best = {best_cost:.5f}")
 
            if stagnation_counter >= stagnation_limit:
                print(f"\nBrak poprawy przez {stagnation_limit} iteracji. Zapisuję najlepszy graf.")
 
                g6 = nx.to_graph6_bytes(best_G, header=False).decode("utf-8").strip()
 
                with open(file_all, "a") as f:
                    f.write(f"\nBest cost: {best_cost}\n")
                    adj_matrix = nx.to_numpy_array(best_G, dtype=int)
                    f.write("Adjacency matrix:\n")
                    for row in adj_matrix:
                        f.write(" ".join(map(str, row)) + "\n")
                    f.write(g6+'\n')    
 
 
                if cost < 1e-9:
                    print("\nZnaleziono graf całkowity!")
                    if g6 not in saved_graph6:
                        saved_graph6.add(g6)
                        with open(file_hash, "a") as f:
                            f.write(g6+'\n')
                            cmd = f"./nauty2_8_9/labelg {file_hash} | sort | uniq | tee final_{file_hash}"
                            subprocess.run(cmd, shell=True, check=True)
 
                break

 
n = int(sys.argv[1])
k = int(sys.argv[2])
t = int(sys.argv[3])
if(t==0): t=None
 
file_hash = "best_graph_graph6_"+str(n)+"_"+str(k)+".txt"
file_all = "best_graph_"+str(n)+"_"+str(k)+".txt"
 
saved_graph6 = set()
try:
    with open("best_graph_graph6_"+str(n)+"_"+str(k)+".txt", "r") as f:
        for line in f:
            saved_graph6.add(line.strip())
except FileNotFoundError:
    open("best_graph_graph6_"+str(n)+"_"+str(k)+".txt", "x").close()
 
search(n, k, t)
