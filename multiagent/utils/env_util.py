from collections import deque
from functools import wraps

def EventCounter(schedules, matters):
    if not (schedules and matters):
        raise ValueError("schedules and matters must be non-empty.")

    def wrap_func(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wrapper.schedules = deque(wrapper.schedules)
            wrapper.matters = deque(wrapper.matters)
            if wrapper.schedules and wrapper.counter == wrapper.schedules[0]:
                wrapper.schedules.popleft()
                if wrapper.matters and len(wrapper.matters) != 1: # keep the last matter
                    wrapper.matters.popleft()
            wrapper.schedule = wrapper.schedules[0] if wrapper.schedules else None
            wrapper.matter = wrapper.matters[0] if wrapper.matters else None
            wrapper.counter += 1
            return func(*args, **kwargs)

        wrapper.counter = 0  # Initialize wrapper.counter to zero before returning it
        wrapper.schedules = deque(schedules)
        wrapper.matters = deque(matters)
        return wrapper
    return wrap_func

class Test():
    @EventCounter(schedules=[2, 4, 8, 16], matters=[0, 1, 2, 3, 4])
    def reset(self):
        print(f'{self.reset.counter}: {self.reset.matter}')
        if self.reset.counter == 12:
            self.reset.__func__.counter = 0
            self.reset.__func__.schedules = [1,5,7,11]
            self.reset.__func__.matters = [10, 20, 30, 40, 50]

if __name__ == '__main__':
    @EventCounter([2, 4, 8, 16], [0, 1, 2, 3])
    def reset():
        print(f'{reset.counter}: {reset.matter}')
        if reset.counter == 12:
            reset.counter = 0
            reset.schedules = [1,5,7,11]
            reset.matters = [10, 20, 30, 40, 50]

    for _ in range(30):
        reset()

    t = Test()
    for _ in range(30):
        t.reset()

