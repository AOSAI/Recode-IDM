from script.image_train import main as train
from script.image_sample import main as sample
from script.image_nll import main as nll
from script.super_res_train import main as res_train
from script.super_res_sample import main as res_sample

if __name__ == "__main__":
    train()
    # sample(["image", "nparray"])
    # nll()

    # res_train()
    # res_sample()