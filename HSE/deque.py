class Deque:

    def __init__(self):
        self.deque = []

    def push_front(self, key):
        self.deque.insert(0, key)
        return 'ok'

    def push_back(self, key):
        self.deque.append(key)
        return 'ok'

    def pop_front(self):
        if not self.deque:
            return 'error'
        else:
            return self.deque.pop(0)

    def pop_back(self):
        if not self.deque:
            return 'error'
        else:
            return self.deque.pop()

    def front(self):
        if not self.deque:
            return 'error'
        else:
            return self.deque[0]

    def back(self):
        if not self.deque:
            return 'error'
        else:
            return self.deque[-1]

    def clear(self):
        if not self.deque:
            return 'error'
        else:
            self.deque.clear()
            return 'ok'

    def size(self):
        return len(self.deque)


def process_deque(commands):
    
    d = Deque()
    actions = {
                'push_back': d.push_back,
                'push_front': d.push_front,
                'pop_front': d.pop_front,
                'pop_back': d.pop_back,
                'front': d.front,
                'back': d.back,
                'clear': d.clear,
                'size': d.size}
    ans = []
    for cond in commands:
        r = cond.split()
        if len(r) > 1:
            resp = actions.get(r[0])(int(r[1]))
            ans.append(resp)
        else:
            ans.append(actions.get(r[0])())
    return ans
                

if __name__ == "__main__":
    test_cmd = ["push_front 1", "push_front 2", "push_back 6", "front", "back", "clear", "size", "back"]
    # should print ["ok", "ok", "ok", 2, 6, "ok", 0, "error"]
    print(process_deque(test_cmd))

    test_cmd = ["pop_front", "back", "push_back 2", "size"]
    # should print ["error", "error", "ok", 1]
    print(process_deque(test_cmd))

    test_cmd = ["push_back 1", "push_front 10", "push_front 4", "push_front 5", "back", "pop_back", "pop_back", "back"]
    # should print ["ok", "ok", "ok", "ok", 1, 1, 10, 4]
    print(process_deque(test_cmd))