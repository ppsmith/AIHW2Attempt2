class LeftPin:
    def __init__(self, prob):
        self.prob = prob
        self.states = {}
    def getNumber(self):
        return self.prob

    def setStates(self, key, states):
        self.states[key] = states

    def getStates(self):
        return self.states

    def getProb(self, key):
        return self.getStates()[key]
