import sys
import os
import openslide
import glob
import numpy as np
import cv2
import pandas as pd
import tqdm
import argparse
from sampler import Sampler
import unet_studio
import network
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_checkpoint_path", type=str, default='./seg_checkpoint.pth')
    parser.add_argument("--classifier_checkpoint_path", type=str, default='./classifier_checkpoint.pth')
    parser.add_argument("--image_quality", type=int, default=0)
    parser.add_argument("--slides_path", type=str)
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--image_ext", type=str, default="mrxs")    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    with torch.no_grad():
        seg_net = unet_studio.Unet(factor=0.25, kernel_size=3, input_size=3, num_outputs=3)
        seg_net.eval()
        seg_net = torch.nn.DataParallel(seg_net)
        checkpoint = torch.load(args.seg_checkpoint_path,map_location='cpu')['state_dict']
        seg_net.load_state_dict(checkpoint)
        seg_net = seg_net.cuda()

        classifier_model = network.Resnet50Classifier()
        classifier_model.eval()
        classifier_checkpoint = torch.load(args.classifier_checkpoint_path, map_location='cpu')
        classifier_model = torch.nn.DataParallel(classifier_model)
        classifier_model.load_state_dict(classifier_checkpoint)
        classifier_model = classifier_model.cuda()
        
        
        paths = glob.glob(os.path.join(args.slides_path, f"*.{args.image_ext}"))
        results_data = pd.read_csv('CSVTEMPLATE.csv')
        for path in tqdm.tqdm(paths):
            name = path.split("/")[-1].replace(f".{args.image_ext}","")

            sampler = Sampler(path, seg_net)
            images_for_classifier = []

            scores = []

            got_one_image = False
            for x, y, image in sampler.sample(image_quality=args.image_quality):
                got_one_image = True
                images_for_classifier.append(image)
                
                if len(images_for_classifier) % 128 == 0:

                    images_for_classifier = np.float32(images_for_classifier).transpose((0, 3, 1, 2)) / 255
                    images_for_classifier = torch.from_numpy(images_for_classifier).cuda()
                    classifier_outputs = classifier_model(images_for_classifier)
                    scores.extend(torch.nn.Softmax(dim=-1)(classifier_outputs).cpu().data.numpy()[:, 1])
                    images_for_classifier = []

            if len(images_for_classifier) > 0:
                images_for_classifier = np.float32(images_for_classifier).transpose((0, 3, 1, 2)) / 255
                images_for_classifier = torch.from_numpy(images_for_classifier).cuda()
                classifier_outputs = classifier_model(images_for_classifier)
                scores.extend(torch.nn.Softmax(dim=-1)(classifier_outputs).cpu().data.numpy()[:, 1])

            if not got_one_image:
                soft_prediction = 0
                hard_prediction = 0
            else:
                num_pos = len([score for score in scores if score>0.5])
                ratio_pooling = num_pos / len(scores)
                print("slide", name, "score", ratio_pooling, "number of tiles", len(scores), flush=True)

                soft_prediction = ratio_pooling
                hard_prediction = soft_prediction > 0.6


            results_data['soft_prediction'][results_data['caseID'] == int(name)] = soft_prediction
            results_data['hard_prediction'][results_data['caseID'] == int(name)] = int(hard_prediction)

        # save result to excel
        output_filename = os.path.join(args.output_path, 'irisai.csv')
        results_data.to_csv(output_filename)