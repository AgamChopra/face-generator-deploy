from deploy import build_model, img_to_bytes
import torch
from PIL import Image
from random import randint


class PythonPredictor:
    def __init__(self,config):
        self.model = build_model().cpu()

    def predict(self):
        resolution = 256
        N = 64
        with torch.no_grad(): 
            noise = torch.rand((N,1,10,10))
        opt = self.model(noise)
        i = randint(0,N-1)
        output = ((opt[i].detach().numpy().T)*255).astype('int')[:,:,::-1]
        img = Image.fromarray(output.astype('uint8'), 'RGB')
        img = img.resize((resolution,resolution))
        return img_to_bytes(img)
