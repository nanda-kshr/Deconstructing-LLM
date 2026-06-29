class Neuron:
    
    def __init__(self):
        self.w = 0.3
        self.lr = 0.1
        self.gradient = 3
    
    def predict(self, x):
        return self.w * x
    
    def loss(self, predicted, actual):
        return (actual-predicted) ** 2
    
    def train_step(self, x, y):
        loss_current = self.loss(self.predict(x), y)
        current_w = self.w
        self.w = current_w -  self.lr
        loss_down = self.loss(self.predict(x), y)
        self.w = current_w + self.lr
        loss_up = self.loss(self.predict(x), y)


        best = min(loss_current, loss_down, loss_up)
        if best == loss_down:
            self.w = current_w - self.lr
        elif best == loss_up:
            self.w = current_w + self.lr
        else:
            self.w = current_w
data = [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8)
]


n = Neuron()

print(n.predict(4))
print(n.loss(n.predict(4), 8))
for epoch in range(100):
    total_loss = 0

    for x, y in data:
        n.train_step(x, y)   
        total_loss += n.loss(n.predict(x), y)

    print(epoch, n.w, total_loss)