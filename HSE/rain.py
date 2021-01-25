def calc_rain_water(h):
    
    pos = []
    water = 0

    for i in range(len(h)):
        if not pos:
            pos.append(i)
        else:
            while pos and h[i] > h[pos[-1]]:
                prev = pos[-1]
                pos.pop()
                if not pos:
                    break
                ht = min(h[i], h[pos[-1]]) - h[prev]
                water += (i - pos[-1] - 1) * ht
            pos.append(i)
    return water

# some test code
if __name__ == "__main__":
    test_h = [2, 5, 2, 3, 6, 9, 1, 3, 4, 6, 1]
    # should print 15
    print(calc_rain_water(test_h))

    test_h = [2, 4, 6, 8, 6, 4, 2]
    # should print 0
    print(calc_rain_water(test_h))

    test_h = [8, 6, 4, 2, 4, 6, 8]
    # should print 18
    print(calc_rain_water(test_h))
    
    test_h = [4, 1, 2, 5, 1, 1, 3]
    # should print 9
    print(calc_rain_water(test_h))

    test_h = [4, 1, 2]
    # should print 1
    print(calc_rain_water(test_h))

    test_h = [4, 2, 1]
    # should print 0
    print(calc_rain_water(test_h))
    
    
    