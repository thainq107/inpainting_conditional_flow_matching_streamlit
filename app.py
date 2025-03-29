import os
import torch
import streamlit as st
from PIL import Image
from torchcfm.models.unet import UNetModel

model = UNetModel(dim=(3, 256, 256), num_channels=32, num_res_blocks=1)

model.load_state_dict(torch.load("model_best.pt", map_location=torch.device("cpu"), weights_only=True))
model.eval()

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
            dy = model(t.to(device), y)
            y = y + dy * dt
            y = cond_image*(1. - mask) + mask*y
            y_values.append(y)
    return torch.stack(y_values)

def inference(gt_image, mask, model):
    gt_image = gt_image.unsqueeze(0)
    noise = torch.randn_like(gt_image)
    mask = mask.unsqueeze(0)
    cond_image = gt_image*(1. - mask) + mask*noise
    
    # Time parameters
    t_steps = torch.linspace(0, 1, 50, device=device)  # Two time steps from 0 to 1
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
            pred_image, cond_image = inference(image, mask, model)
            st.image(cond_image)
            st.image(pred_image)
          
    elif option == "Run Example Image":
        image = Image.open('example.png').convert("RGB").resize((256, 256))
        pred_image, cond_image = inference(image, mask, model)
        st.image(cond_image)
        st.image(pred_image)

if __name__ == '__main__':
    main() 
