import math

def tj_cost(L, W):
    n = len(W)
    tbl = [ math.inf ] * (n + 1)
    tbl[0] = 0
    for i in range(1, n + 1):
        length = -1
        for j in range(i - 1, -1, -1):
            length += 1 + len(W[j])
            if length > L:
                P = math.inf
            else:
                if i == n: P = 0
                else: P = (L - length)**3
            tbl[i] = min(tbl[i], tbl[j] + P)
    return tbl[n]

def tj_cost_slow(L, W):
    n = len(W)
    tbl = [ math.inf ] * (n + 1)
    tbl[0] = 0
    for i in range(1, n + 1):
        for j in range(i):
            length = i - j - 1
            for k in range(j, i):
                length += len(W[k])
            if length > L:
                P = math.inf
            else:
                if i == n: P = 0
                else: P = (L - length)**3
                
            tbl[i] = min(tbl[i], tbl[j] + P)
    return tbl[n]

def tj(L, W):
    n = len(W)
    tbl = [ math.inf ] * (n + 1)
    split  = [0] * (n + 1) # NEW!
    tbl[0] = 0
    for i in range(1, n + 1):
        length = -1
        for j in range(i-1, -1, -1):
            length += 1 + len(W[j])
            if length > L:
                P = math.inf
            else:
                if i == n: P = 0
                else: P = (L - length)**3                
            if tbl[i] > tbl[j] + P:
                tbl[i] = tbl[j] + P
                split[i] = j
    
    result = []
    last = n
    while last > 0:
        result.append(' '.join(W[split[last] : last]))
        last = split[last]
    return '\n'.join(result[::-1])

if __name__ == "__main__":
    W_example = ["jars", "jaws", "joke", "jury", "juxtaposition"]
    L_example = 15
    # should print 432
    print(tj_cost(L_example, W_example))
    W_example = ["a", "a", "a", "a"]
    L = 5
    # should print 0
    
    print(tj_cost(L, W_example))

    W_example = ["one", "to", "three", "f"]
    L = 5
    # should print(35)
    print(tj_cost(L, W_example))
    W_example = ["one", "to", "three", "f"]
    L = 6
    # should print(1)
    print(tj_cost(L, W_example))
    
    # should print:
    #jars jaws
    #joke jury
    #juxtaposition
    W_example = ["jars", "jaws", "joke", "jury", "juxtaposition"]
    L_example = 15
    print(tj(L_example, W_example))

    W_example = ["one", "to", "three", "f"]
    L_example = 6
    print(tj(L_example, W_example))