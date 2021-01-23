def lines(a):
    
    deleted = 0
    counter = 1
    i = 0
    while i < len(a)-2:
        j = i + 1
        while j <= len(a)-1 and a[i] == a[j] :
            j += 1
            counter += 1
        if counter >= 3:
            indexes = list(range(i,j))
            a[:] = [a[k] for k in range(len(a)) if k not in indexes]
            deleted += len(indexes)
            i = 0
            counter = 1
        else:
            i += 1
            counter = 1
    return deleted


# some test code
if __name__ == "__main__":
    test_a = [2, 2, 1, 1, 1, 2, 1]
    # should print 6
    print(lines(test_a))

    test_a = [0, 0, 0, 0, 0]
    # should print 5
    print(lines(test_a))

    test_a = [2, 3, 1, 4]
    # should print 0
    print(lines(test_a))

