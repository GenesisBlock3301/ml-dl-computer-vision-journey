from torchvision.transforms import ToTensor
from PIL import Image

img = Image.open("car.jpeg")
tensor_img = ToTensor()(img)

print(tensor_img.shape)  # e.g. torch.Size([1, 28, 28])
print(tensor_img)  # tensor(0.) tensor(1.)
