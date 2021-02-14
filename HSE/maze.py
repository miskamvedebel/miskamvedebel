import math

def fastest_escape_length(maze, i=0, j=0):
    # (i, j) is the starting position
    # maze[x][y] = 0 <=> (x, y) cell is empty
    # maze[x][y] = 1 <=> (x, y) cell contains a wall
    n = len(maze)
    m = len(maze[0])
    if i == n - 1 and j == m - 1:
        return 1
    maze[i][j] = 1
    result = math.inf
    for a, b in [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]:
        if 0 <= a < n and 0 <= b < m and maze[a][b] == 0:
            result = min(result, fastest_escape_length(maze, a, b) + 1)
    maze[i][j] = 0
    return result

def fastest_escapes(maze, i=0, j=0):
    pass

def weighted_escape_length(maze, w, i=0, j=0):
    # (i, j) is the starting position
    # maze[x][y] = 0 <=> (x, y) cell is empty
    # maze[x][y] = 1 <=> (x, y) cell contains a wall
    n = len(maze)
    m = len(maze[0])
    if i == n - 1 and j == m - 1:
        return 1
    val = maze[i][j]
    maze[i][j] = 'X'
    result = math.inf
    for a, b in [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]:
        if 0 <= a < n and 0 <= b < m and (maze[a][b] == 0 or maze[a][b] == 1):
            if maze[a][b] == 0:
                result = min(result, weighted_escape_length(maze, w, a, b)+1)
            elif maze[a][b] == 1:
                result = min(result, weighted_escape_length(maze, w, a, b)+w)
    maze[i][j] = val
    return result

def weighted_escapes(maze, w, i=0, j=0):
    pass


# some test code
if __name__ == "__main__":
    test_a = [
        [0, 0, 0],
        [1, 1, 0],
        [1, 1, 0]
    ]
    # should print 5
    print(fastest_escape_length(test_a))
    # should print 2
    print(weighted_escape_length(test_a, 0))
    test_b = [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ]
    # should print inf
    print(fastest_escape_length(test_b))
    # should print 5
    print(weighted_escape_length(test_b, 1))
    # should print 6
    print(weighted_escape_length(test_b, 2))

    # should print [[(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]]
    print(fastest_escapes(test_a))
    # should print []
    print(fastest_escapes(test_b))
    # should print [5, 5, 5, 5, 5, 5]
    #print(list(map(len, fastest_escapes([[0 for _ in range(3)] for _ in range(3)]))))

    # should print [[(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)]]
    print(weighted_escapes(test_b, 0))
    # should print [[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)], [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)], [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]]
    # the order of the paths within the list might be different
    print(weighted_escapes(test_b, 2))
