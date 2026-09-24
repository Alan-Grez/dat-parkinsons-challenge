"""Verifica el entorno sin abrir ni inspeccionar datos de la competencia."""

import platform

import monai
import nibabel
import pandas
import sklearn
import torch
import torchvision


def main() -> None:
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"monai={monai.__version__}")
    print(f"nibabel={nibabel.__version__}")
    print(f"pandas={pandas.__version__}")
    print(f"scikit-learn={sklearn.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        result = torch.ones(3, device="cuda").sum().cpu().item()
        print(f"gpu={device}")
        print(f"gpu_smoke_test={result}")


if __name__ == "__main__":
    main()

