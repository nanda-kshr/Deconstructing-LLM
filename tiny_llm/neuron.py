class Neuron:
    
    def __init__(self):
        self.w = 0.3
        self.lr = 0.0001
        self.gradient = 1
        self.epsilon = 0.0001
        self.b = 0
    
    def predict(self, x):
        return self.w * x + self.b
    
    def loss(self, predicted, actual):
        return (actual-predicted) ** 2
    
    def compute_gradient(self, actual_loss, predicted_loss):
        return (predicted_loss - actual_loss) / self.epsilon

    
    def train_step(self, x, y):
        loss_current = self.loss(self.predict(x), y)
        current_w = self.w 
        self.w += self.epsilon
        loss_new = self.loss(self.predict(x), y)
        self.w = current_w
        self.gradient = self.compute_gradient(loss_current, loss_new)
        self.w -= self.lr * self.gradient


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

    if epoch % 10 == 0:
        print(epoch, n.w, total_loss)