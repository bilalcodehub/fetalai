import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Common transforms for images
transform = T.Compose([
    T.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    T.ToTensor(),
    # Optionally, normalize here
    # T.Normalize(mean=[0.485, 0.456, 0.406], 
    #             std=[0.229, 0.224, 0.225])
])

# Common transforms for masks
transform_mask = T.Compose([
    T.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    T.ToTensor()
])
