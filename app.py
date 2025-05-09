import os
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchcfm.models.unet import UNetModel
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

model = UNetModel(dim=(3, 256, 256), num_channels=32, num_res_blocks=1)

model.load_state_dict(torch.load("model_best.pth", map_location=torch.device("cpu"), weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.PILToTensor()
])

def bbox2mask(bbox, dtype='uint8'):
    """
    Generate mask in ndarray from bbox.
    bbox (tuple[int]): Configuration tuple, (top, left, height, width)
    """

    height, width = 256, 256

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1

    return mask
h, w = 256, 256
mask = bbox2mask((h//4, w//4, h//4, w//4))
mask = torch.from_numpy(mask).permute(2,0,1)

def euler_method(model, cond_image, t_steps, dt, mask):
    y = cond_image
    y_values = [y]
    with torch.no_grad():
        for t in t_steps[1:]:
            t = t.reshape(-1, )
            dy = model(t, y)
            y = y + dy * dt
            y = cond_image*(1. - mask) + mask*y
            y_values.append(y)
    return torch.stack(y_values)

def inference(gt_image, mask, model):
    gt_image = gt_image.unsqueeze(0)
    noise = torch.randn((256,256))
    mask = mask.unsqueeze(0)
    cond_image = gt_image*(1. - mask) + mask*noise
    
    # Time parameters
    t_steps = torch.linspace(0, 1, 5)  # Two time steps from 0 to 1
    dt = t_steps[1] - t_steps[0]  # Time step
    
    # Solve the ODE using Euler method
    traj = euler_method(model, cond_image, t_steps, dt, mask)
    return traj[-1, -1], cond_image

def main():
    st.title('Image Inpainting using Conditional Flow Matching')
    st.subheader('Model: Conditional Flow Matching. Dataset: CelebA-HQ')
    option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Run Example Image'))
    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image", type=["jpg", "png"])
        if file is not None:
            image = Image.open(file).convert("RGB").resize((256, 256))
            image = transform(image)
            pred_image, cond_image = inference(image, mask, model)
            grid = make_grid(
                show_imgs.view([-1, 3, 256, 256]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
            )
            img = ToPILImage()(grid)
            st.image(img)
          
    elif option == "Run Example Image":
        image = Image.open('example.png').convert("RGB").resize((256, 256))
        image = transform(image)
        pred_image, cond_image = inference(image, mask, model)
        show_imgs = torch.cat([cond_image.squeeze(), pred_image], dim=0)
        grid = make_grid(
            show_imgs.view([-1, 3, 256, 256]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
        )
        img = ToPILImage()(grid)
        st.image(img)

if __name__ == '__main__':
    main() 
