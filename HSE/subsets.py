#implementing subsets with reccurence

def subsets(elems):
    '''function to return subsets from the list of elements
    Example: [1, 2] --> [[], [1], [2], [1, 2]]
    
    :param elems: list of elements
    :type elems: list

    :returns: list of subsets
    :rtype: list
    '''
    #base case is when no elements
    if len(elems) == 0:
        return [[]]

    #taking a smaller problem for the list without last element
    results = subsets(elems[:-1])

    #appending to the result last element
    for i in range(len(results)):
        results.append(results[i] + [elems[-1]])

    return results

# some test code
if __name__ == "__main__":
    test = [1, 2]
    assert subsets(test) == [[], [1], [2], [1, 2]]

    test = [1, 2, 3]
    assert subsets(test) == [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]

