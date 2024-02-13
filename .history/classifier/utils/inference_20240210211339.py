
from PIL import Image
import torch



def inference(model, image_path, data_transforms, device):
    model.eval()

    img = Image.open(image_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    return preds