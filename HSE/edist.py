import math

def edistance(A, B):
    m = len(A)
    n = len(B)
    
    tbl = [[0 for _ in range(m+1)] for _ in range(n+1)]
    
    for i in range(m+1):
        tbl[0][i] = i
    for j in range(n+1):
        tbl[j][0] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if A[i-1] == B[j-1]:
                tbl[j][i] = tbl[j-1][i-1]
            else:
                tbl[j][i] = min(tbl[j-1][i], tbl[j][i-1], tbl[j-1][i-1]) + 1
    
    return tbl[-1][-1]

def weighted_edistance(A, B, wdel, wins, wsub):
    m = len(A)
    n = len(B)
    
    tbl = [[0 for _ in range(m+1)] for _ in range(n+1)]
    
    for i in range(1, m+1):
        tbl[0][i] = wdel * i
    for j in range(1, n+1):
        tbl[j][0] = wins * j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if A[i-1] == B[j-1]:
                
                tbl[j][i] = tbl[j-1][i-1]
            else:
                tbl[j][i] = min(tbl[j-1][i] + wins, tbl[j][i-1]+wdel, tbl[j-1][i-1]+wsub)
#     print('\n'.join([str(l) for l in tbl]))
    return tbl[-1][-1]

def edistance_substring(A, B):
    m = len(A)
    n = len(B)
    
    tbl = [[0 for _ in range(m+1)] for _ in range(n+1)]
    
    # for i in range(1, m+1):
    #     tbl[0][i] = i
    for j in range(1, n+1):
        tbl[j][0] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if A[i-1] == B[j-1]:
                tbl[j][i] = tbl[j-1][i-1]
            else:
                tbl[j][i] = min(tbl[j-1][i], tbl[j][i-1], tbl[j-1][i-1]) + 1
    
    return tbl[-1][-1]

if __name__ == "__main__":
    # should print 3
    print(edistance("good", "bad"))
    # should print 7
    print(weighted_edistance("good", "bad", 1, 2, 5))
