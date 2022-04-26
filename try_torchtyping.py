import torch
from model_definition import Model

data_x = torch.randn((1000, 2))

EPOCHS = 10
# fake a dataloader
data_x = torch.cat([
    data_x[torch.randperm(data_x.shape[0]), :]
    for _ in range(EPOCHS)
])


def xor(a, b):
    an = 0 if a < 0 else 1
    bn = 0 if b < 0 else 1
    res = bool(an) ^ bool(bn)
    if res > 0:
        return [0, 1]
    else:
        return [1, 0]


# would love to use `torch.vmap` but it is only avalable in nightly
# https://pytorch.org/tutorials/prototype/vmap_recipe.html
data_y = torch.FloatTensor([xor(*t) for t in data_x])

val_x = torch.randn((1000, 2))
val_y = torch.FloatTensor([xor(*t) for t in val_x])

model = Model()
criterion = torch.nn.BCEWithLogitsLoss()
LEARNING_RATE = 0.01
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

BATCH_SIZE = 16
for i in range(0, len(data_x), BATCH_SIZE):
    batch_x = data_x[i:i + BATCH_SIZE, :]
    batch_y = data_y[i:i + BATCH_SIZE, :]

    optim.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)        

    if i % 1024 == 0:
        acc_c = 0
        for vx, vy in zip(val_x, val_y):
            pred = model.xor_predict(vx)
            gt = 1 if vy[0] >= vy[1] else 0
            if pred == gt:
                acc_c += 1
        print(f"iter={i:5d} training loss = {loss.item():.4f} | validation accuracy = {acc_c/len(val_y):.2%}")

    loss.backward()
    optim.step()
