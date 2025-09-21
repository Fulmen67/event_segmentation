from abc import abstractmethod

class base:
    @abstractmethod
    def method(self):
        return NotImplementedError


class one(base):
    def method(self):
        return 2


class two(base):
    pass

def func(object: base):
    print(object.method())

if __name__ == '__main__':
    object1 = one()
    object2 = two()

    func(object2)