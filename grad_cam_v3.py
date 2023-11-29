# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import warnings
warnings.filterwarnings("ignore")

# get parameters from command line. -m or --model is the model name, -im or --image_file is the image path, -id or --index_file is the index file path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model name", default="resnet50")
parser.add_argument("-im", "--image", help="image path", default="./images_with_labels/1_n03445924_golfcart.png")
parser.add_argument("-id", "--index", help="index file path", default="./images_with_labels/imagenet_class_index.json")

model_name = parser.parse_args().model
image_file = parser.parse_args().image
index_file = parser.parse_args().index


import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import json
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
as_numpy = lambda x: x.detach().cpu().numpy()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]

def show_result(img, saliency, label = ""):
    img = np.array(img, dtype = float) / 255.0

    saliency = F.interpolate(saliency, size = img.shape[:2], mode = "bilinear")
    saliency = as_numpy(saliency)[0, 0]
    saliency = saliency - saliency.min()
    saliency = np.uint8(255 * saliency / saliency.max())
    heatmap = jet_colors[saliency]
    plt.imshow(0.5 * heatmap + 0.5 * img)
    plt.axis("off")
    plt.title(label)
    
    save_path = "./png/" + label + ".png"
    plt.savefig(save_path)
    # plt.show()


# define the preprocessing transform

def get_pil_transform(): 
    transf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize
    ])    

    return transf    

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

# resize and take the center part of image to what our model expects
def get_input_transform():
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])       
    transf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize
    ])    

    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)



# index_file = "images_with_labels/imagenet_class_index.json"
with open(index_file) as f:
    indx2label = json.load(f)


def decode_predictions(preds, k=5):
    # return the top k results in the predictions
    return [
        [(*indx2label[str(i)], i, pred[i]) for i in pred.argsort()[::-1][:k]]
        for pred in as_numpy(preds)
    ]

class Probe:
    def get_hook(self,):
        self.data = []
        def hook(module, input, output):
            self.data.append(output)
        return hook


# load the image
print("loading the image...")
# image_file = "./images_with_labels/1_n03445924_golfcart.png"
img = Image.open(image_file)

# img 是一个 4 channel 的 RGBA image, 我们需要把它转换成 3 channel 的 RGB image
img = img.convert("RGB")

x = get_input_tensors(img).to(device)

print("loading the model...")
### You can change the model here.

# model_name = "resnet50"
model_func = getattr(torchvision.models, model_name)
model = model_func(pretrained=True)
model.eval()
model.to(device)

#add a probe to model
probe = Probe()
#probe will save the output of the layer4 during forward
handle = model.layer4.register_forward_hook(probe.get_hook())

logits = model(x)
preds = logits.softmax(-1)

print("the prediction result:")
# for tag, label, i, prob in decode_predictions(preds)[0]:
#     print("{} {:16} {:5} {:6.2%}".format(tag, label, i, prob))
for _, label, _, prob in decode_predictions(preds)[0]:
    print(" {:16} {:6.2%}".format(label, prob))


print("Calculating the saliency of the top prediction...")
target = preds.argmax().item()

### Grad_Cam
# get the last_conv_output
last_conv_output = probe.data[0]
handle.remove()

last_conv_output.retain_grad() #make sure the intermediate result save its grad

#backprop
logits[0, target].backward()
grad = last_conv_output.grad 
#taking average on the H-W panel
weight = grad.mean(dim = (-1, -2), keepdim = True)

saliency = (last_conv_output * weight).sum(dim = 1, keepdim = True)
#relu
saliency = saliency.clamp(min = 0)

label_name = indx2label[str(target)][1]
pltTitle = "{}_{}".format(label_name, model_name)
show_result(img, saliency, pltTitle)

# save the prediction result to a csv file
predictions = decode_predictions(preds)[0]
prediction_data = [{"Label": label, "Probability": prob} for _, label, _, prob in predictions]
df = pd.DataFrame(prediction_data)
df.to_csv("./csv/{}.csv".format(pltTitle), index=False)
