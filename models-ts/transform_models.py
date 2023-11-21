import torch
import torch.jit

from psst.models import Inception3, Vgg13


def main():
    i3 = torch.load("models_test/Inception3/AridAgar.pt")
    bg, bth = Inception3(), Inception3()
    bg.load_state_dict(i3["bg_model"])
    bth.load_state_dict(i3["bth_model"])
    bg_script: torch.jit.RecursiveScriptModule = torch.jit.script(bg)
    bth_script: torch.jit.RecursiveScriptModule = torch.jit.script(bth)
    bg_script.save("models-ts/Inception3-AridAgar-Bg.pt")
    bth_script.save("models-ts/Inception3-AridAgar-Bth.pt")

    v13 = torch.load("models_test/Vgg13/AridAgar.pt")
    bg, bth = Vgg13(), Vgg13()
    bg.load_state_dict(v13["bg_model"])
    bth.load_state_dict(v13["bth_model"])
    bg_script: torch.jit.RecursiveScriptModule = torch.jit.script(bg)
    bth_script: torch.jit.RecursiveScriptModule = torch.jit.script(bth)
    bg_script.save("models-ts/Vgg13-AridAgar-Bg.pt")
    bth_script.save("models-ts/Vgg13-AridAgar-Bth.pt")


if __name__ == "__main__":
    main()
