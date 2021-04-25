def dfs(v):
    visited[v] = True

    # print("Visited vertex V!")

    for u in graph[v]:
        if not visited[u]:
            dfs(u)
    # print("Finished exploring vertice V!")

graph = [
    [1, 2],
    [0],
    [0],
    [],
    [5],
    [4]
]

n = len(graph)
components = 0
visited = [False for i in range(n)]

for v in range(n):
    if not visited[v]:
        components += 1
        dfs(v)
print("Number of components", components)

graph = [
    [1],
    [],
    [0],
    [1, 2, 5],
    [5],
    []
]

def dfs_topol(v):
    visited[v] = True
    for u in graph[v]:
        if not visited[u]:
            dfs_topol(u)
    order.append(v)

n = len(graph)
visited = [False for _ in range(n)]
order = []

for v in range(n):
    if not visited[v]:
        dfs_topol(v)

order = order[::-1]
print(order)