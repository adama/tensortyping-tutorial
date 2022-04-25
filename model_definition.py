import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard() # uncomment to allow checking at runtime

HIDDEN_DIM = 16


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(2, HIDDEN_DIM)
        self.relu1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(HIDDEN_DIM, 2)

    @typechecked
    def forward(self, x: TensorType["batch", 2]) -> TensorType["batch", 2]:
        x = self.lin1(x)
        x = self.relu1(x)
        x: TensorType["batch", 2] = self.lin2(x)  # lhs is optional
        return x

    # remember silent failure for no batch dim!
    @typechecked
    def xor_predict(self, x: TensorType[2]) -> int:
        x = x.unsqueeze(0)  # could also use einsum
        x = self.forward(x)
        x = torch.sigmoid(x)
        x = x.squeeze()
        x = 1 if x[0] >= x[1] else 0
        return x