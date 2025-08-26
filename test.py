import torch
from PIL import Image
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f">$ Device : {device}")

im = Image.open(r"C:\research\ERA4\psoriasis_food_scout\imag1.jpg")
print(im)