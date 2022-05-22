import torch.nn as nn
import torch
from PIL import Image
from random import randint
import base64
from io import BytesIO


def img_to_bytes(img):
    """
    Convert PIL image to base64
    :param img : PIL Image object
    :return string : image in the form of base64
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


class GeneratorA(nn.Module): # (N,100,1,1) -> (N,3,128,128)
    def __init__(self,CH):
        super(GeneratorA, self).__init__()
        self.a,self.c_,self.c = 16,24,40
        
        self.fc = nn.Sequential(nn.Linear(100, self.c_ * self.a * self.a), nn.ReLU(inplace=False), nn.BatchNorm1d(self.c_ * self.a * self.a))
        
        self.E = nn.Sequential(nn.ConvTranspose2d(self.c_, self.c, kernel_size=4, padding=1, stride=2), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, stride=1), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, stride=1), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, stride=1), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.ConvTranspose2d(self.c, self.c, kernel_size=4, padding=1, stride=2), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, stride=1), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.ConvTranspose2d(self.c, self.c, kernel_size=4, padding=1, stride=2), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.Conv2d(self.c, CH, kernel_size=3, padding=1, stride=1), nn.Tanh())        
    
    def forward(self, x):
        y = self.fc(x.view(x.shape[0],100))
        y = self.E(y.view(x.shape[0],self.c_,self.a,self.a))
        output = (y + 1)/2
        return  output


def build_model():
    with torch.no_grad():    
        model = GeneratorA(3)
        model.load_state_dict(torch.load("Gen_temp.pt"))
    return model.eval()


def main():
    model = build_model()
    N = 64
    with torch.no_grad(): 
        noise = torch.rand((N,1,10,10))
    opt = model(noise)

    i = randint(0,N-1)
    output = ((opt[i].detach().numpy().T)*255).astype('int')[:,:,::-1]
    img = Image.fromarray(output.astype('uint8'), 'RGB')
    img = img.resize((256,256))
    img.show()
    

if __name__ == "__main__":
    main()