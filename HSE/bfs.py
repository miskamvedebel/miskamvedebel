from queue import Queue

def bfs(start):
    q = Queue()
    q.put(start)
    visited[start] = True

    while not q.empty():

        v = q.get()
        #adjacency list
        for u in graph[v]:
            if not visited[u]:
                q.put(u)
                visited[u] = True
                #if we need to something with U
        #if we want to do something after visiting all the neighbours

#Application shortest part problem
def bfs_distance(start):
    q = Queue()
    q.put((0, start))
    visited[start] = True

    while not q.empty():

        d, v = q.get()
        print("Distance to vertice to", v, "from start equals", d)
        #adjacency list
        for u in graph[v]:
            if not visited[u]:
                q.put((d + 1, u))
                visited[u] = True

graph = [
    [1, 2],
    [0],
    [0, 3, 4],
    [2],
    [2, 5],
    [4]
]

n = len(graph)
visited = [False for _ in range(n)]
bfs_distance(0)