import json
import logging
import argparse
import torch
from model.hw_with_style import HWWithStyle
import cv2
from utils import string_utils
import numpy as np

logging.basicConfig(level=logging.INFO, format='')

def main(img_path, message, resume ,saveDir  ,gpu=None,config=None,addToConfig=None, fromDataset=False, test=False, arguments=None,style_loc=None):
    np.random.seed(1234)
    torch.manual_seed(1234)
    if resume is not None:
        checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
        
        keys = list(checkpoint['state_dict'].keys())
         
        if config is None:
            config = checkpoint['config']
        else:
            config = json.load(open(config))
        
    config['model']['RUN']=True    

    if gpu is None:
        config['cuda']=False

    if checkpoint is not None:
        if 'state_dict' in checkpoint:
            model = eval(config['arch'])(config['model'])
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint['model']
    model.eval()

    model.count_std=0
    model.dup_std=0

    char_set_path = config['data_loader']['char_file']
    
    with open(char_set_path) as f:
        char_set = json.load(f)
    char_to_idx = char_set['char_to_idx']

    with torch.no_grad():
        while True:
            if arguments is None:
                action = True
                    
            if action == True: 
                if arguments is None:
                    path1 =  img_path
                    path2 = img_path
                    text_gen = message

                img_height=64

                image1 = cv2.imread(path1,0)
                if image1.shape[0] != img_height:
                    percent = float(img_height) / image1.shape[0]
                    image1 = cv2.resize(image1, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
                image1 = image1[...,None]
                image1 = image1.astype(np.float32)
                image1 = 1.0 - image1 / 128.0
                image1 = image1.transpose([2,0,1])
                image1 = torch.from_numpy(image1)
                if gpu is not None:
                    image1=image1.to(gpu)

                image2 = cv2.imread(path2,0)
                if image2.shape[0] != img_height:
                    percent = float(img_height) / image2.shape[0]
                    image2 = cv2.resize(image2, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
                image2 = image2[...,None]
                image2 = image2.astype(np.float32)
                image2 = 1.0 - image2 / 128.0
                image2 = image2.transpose([2,0,1])
                image2 = torch.from_numpy(image2)
                if gpu is not None:
                    image2=image2.to(gpu)

                min_width = min(image1.size(2),image2.size(2))
                style = model.extract_style(torch.stack((image1[:,:,:min_width],image2[:,:,:min_width]),dim=0),None,1)
                if type(style) is tuple:
                    style1 = (style[0][0:1],style[1][0:1],style[2][0:1])
                    style2 = (style[0][1:2],style[1][1:2],style[2][1:2])
                else:
                    style1 = style[0:1]
                    style2 = style[1:2]

                images,stylesInter=interpolate(model,style1,style2, text_gen,char_to_idx,gpu)

                for b in range(images[0].size(0)):
                    for i in range(len(images)):
                        genStep = ((1-images[i][b].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                        path = 'static/generated.png'
                        cv2.imwrite(path,genStep)
                    exit(0)                        


# generates a series of images interpolating between the styles
def interpolate(model,style1,style2,text,char_to_idx,gpu,step=0.05):
    if type(style1) is tuple:
        batch_size = style1[0].size(0)
    else:
        batch_size = style1.size(0)

    label = string_utils.str2label_single(text, char_to_idx)
    label = torch.from_numpy(label.astype(np.int32))[:,None].expand(-1,batch_size).to(gpu).long()
    label_len = torch.IntTensor(batch_size).fill_(len(text))
    results=[]
    styles=[]
    for alpha in np.arange(0,1.0,step):
        if type(style1) is tuple:
            style = (style2[0]*alpha+(1-alpha)*style1[0],style2[1]*alpha+(1-alpha)*style1[1],style2[2]*alpha+(1-alpha)*style1[2])
        else:
            style = style2*alpha+(1-alpha)*style1
        gen = model(label,label_len,style)
        results.append(gen)
        if type(style) is tuple:
            styles.append((style[0].cpu().detach(),style[1].cpu().detach(),style[2].cpu().detach()))
        else:
            styles.append(style.cpu().detach())
    return results, styles

 
 

if __name__ == '__main__':

    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Interactive script to generate images from trained model')
    parser.add_argument("--image", type=str,default=None, help="Path to the input image")
    parser.add_argument("--message", type=str,default="Type in handwriting", help="Message string")
    parser.add_argument('-c', '--checkpoint', default='.\saved_model\checkpoint-iteration175000.pth', type=str, help='path to training snapshot (default: None)')
    parser.add_argument('-d', '--savedir', default= '.\static', type=str,
                        help='path to directory to save result images (default: None)')
     
    parser.add_argument('-T', '--test', default=False, action='store_const', const=True,
                        help='Run test set')
     
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn"')
    parser.add_argument('-r', '--run', default=None, type=str,
                        help='command to run')
    args = parser.parse_args()

    addtoconfig=[]
    if args.addtoconfig is not None:
        split = args.addtoconfig.split(',')
        for kv in split:
            split2=kv.split('=')
            addtoconfig.append(split2)
    
    arguments = None
    
    main(args.image,args.message,  args.checkpoint, args.savedir, addToConfig=addtoconfig, test =args.test,arguments=arguments,  )