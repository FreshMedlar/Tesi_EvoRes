from bAlt.train import BAltNet, load_data, to_one_hot
import numpy as np
data, chars, c2i, i2c = load_data("tinyshakespeare.txt")
train_int = data[:5000].astype(int)
train_oh = to_one_hot(train_int, 65)
model = BAltNet(N=500, vocab_size=65, learning_rate=0.01)
for t in range(500):
    model.train_step(train_oh[t], train_oh[t+1])
print("Max |x|:", np.max(np.abs(model.x)), "Mean |x|:", np.mean(np.abs(model.x)))
print("Max |W|:", np.max(np.abs(model.W)), "Mean |W|:", np.mean(np.abs(model.W)))
