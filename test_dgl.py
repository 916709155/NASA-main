import dgl

# 创建一个 DGL 图
g = dgl.graph(([0, 0, 0, 1, 1, 2, 3, 3], [1, 2, 3, 0, 3, 0, 0, 1]))

# 遍历所有边
print(g.edges())

# for u, v in zip(g.edges()[0],g.edges()[1]):
#     print("({}, {})".format(u, v))

def mine_triangle_motifs(graph):
    motifs = set()
    for u, v in zip(graph.edges()[0].numpy(), graph.edges()[1].numpy()):
        successors_u = set(graph.successors(u).numpy())
        successors_v = set(graph.successors(v).numpy())
        predecessors_u = set(graph.predecessors(u).numpy())
        predecessors_v = set(graph.predecessors(v).numpy())
        N_u = successors_u.union(predecessors_u)
        N_v = successors_v.union(predecessors_v)
        common_neighbors = N_u.intersection(N_v)
        if common_neighbors:
            for w in common_neighbors:
                if graph.has_edges_between(u, w) and graph.has_edges_between(v, w):
                    motif = frozenset([u, v, w])
                    motifs.add(motif)
    return motifs

modifs = mine_triangle_motifs(g)
count = len(modifs)
print(count)



