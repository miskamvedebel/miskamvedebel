def sliding_window_min(a, k):
    q = []
    ans = []

    for i in range(k):
        while q and a[i] <= a[q[-1]]:
            q.pop()
        q.append(i)

    for i in range(k, len(a)):
        ans.append(a[q[0]])
        while q and q[0] <= i - k:
            q.pop(0)
        while q and a[i] <= a[q[-1]]:
            q.pop()
        q.append(i)
    ans.append(a[q[0]])
    return ans

# some test code
if __name__ == "__main__":
    test_a, test_k = [1, 3, 4, 5, 2, 7], 3
    # should print [1, 3, 2, 2]
    print(sliding_window_min(test_a, test_k))

    test_a, test_k = [5, 4, 10, 1], 2
    # should print [4, 4, 1]
    print(sliding_window_min(test_a, test_k))

    test_a, test_k = [10, 20, 6, 10, 8], 5
    # should print [6]
    print(sliding_window_min(test_a, test_k))

