import argparse
import numpy as np
import pandas as pd
import torch
import json
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Load a model from a checkpoint and make a prediction on an image")
    parser.add_argument('path_to_image', type=str, help='set the image path')
    parser.add_argument('--category_names', type=str, help='path to the JSON file containing category names')
    parser.add_argument('--top_k', default=5, type=int, help='set the num of topk')
    parser.add_argument('--gpu', default=False, type=bool, help='set the gpu mode')
    parser.add_argument('checkpoint', default='checkpoint.pth', type=str, help='set the ckpt path')
    args = parser.parse_args()
    return args

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    return model

def process_image(image):
    im = Image.open(image)
    im = im.resize((256, 256))
    im = im.crop((16, 16, 240, 240))
    np_image = np.array(im)
    np_image_norm = ((np_image / 255) - ([0.485, 0.456, 0.406])) / ([0.229, 0.224, 0.225])
    np_image_norm = np_image_norm.transpose((2, 0, 1))
    return np_image_norm

def predict(image_path, model, device, topk):
    image = torch.from_numpy(process_image(image_path))
    image = image.unsqueeze(0).float()
    model, image = model.to(device), image.to(device)
    model.eval()
    model.requires_grad = False
    outputs = torch.exp(model.forward(image)).topk(topk)
    probs, classes = outputs[0].data.cpu().numpy()[0], outputs[1].data.cpu().numpy()[0]
    idx_to_class = {key: value for value, key in model.class_to_idx.items()}
    classes = [idx_to_class[classes[i]] for i in range(classes.size)]
    return probs, classes

def main():
    args = parse_args()
    img_path = args.path_to_image
    gpu_mode = args.gpu
    topk = args.top_k
    cat_to_name = args.category_names
    ckpt_path = args.checkpoint
    
    print('Image path:       {}'.format(img_path))
    print('Load model from:  {}'.format(ckpt_path))
    print('GPU mode:         {}'.format(gpu_mode))
    print('TopK:             {}'.format(topk))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    model = load_checkpoint(ckpt_path)
    probs, classes = predict(img_path, model, device, topk)
    
    if cat_to_name is None:
        name_classes = classes
    else:
        with open(cat_to_name, 'r') as f:
            cat_to_name_data = json.load(f)
        name_classes = [cat_to_name_data[i] for i in classes]
    pd_dataframe = pd.DataFrame({
        'classes': pd.Series(data = name_classes),
        'values': pd.Series(data = probs, dtype='float64')
    })
    print(pd_dataframe)

    
if __name__ == '__main__':
    main()