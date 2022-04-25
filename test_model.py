import torch
import pytest
from model_definition import Model


@pytest.fixture
def model():
    return Model()

def test_forward_1dim(model):
    with pytest.raises(TypeError):
        model.forward(torch.randn((4,)))

def test_forward_2dims(model):
    model.forward(torch.randn((4,2)))

def test_forward_3dims(model):
    with pytest.raises(TypeError):
        model.forward(torch.randn((4,3,2)))

def test_forward_more_classes(model):
    with pytest.raises(TypeError):
        model.forward(torch.randn((4,9)))

def test_forward_error1(model):
    model.forward(torch.randn((4,9)))
    model.forward(torch.randn((4,2,3,4,1)))

def test_forward_error2(model):
    model.forward(torch.randn((4,2,3,4,1)))