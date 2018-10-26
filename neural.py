class Opertaion():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)
        
    def compute(self) :
        pass

class add(Opertaion):
    def __init__(self, x,y):
        super().__init__([x,y])

    def compute(self, x, y) :
        self.input = [x,y]
        return x+y


class multiply(Opertaion):
    def __init__(self, x,y):
        super().__init__([x,y])

    def compute(self, x, y) :
        self.input = [x,y]
        return x*y

class matmul(Opertaion):
    def __init__(self, x,y):
        super().__init__([x,y])

    def compute(self, x, y) :
        self.input = [x,y]
        return x.dot(y)