from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import argparse
import numpy as np
import torch
import skimage.io
from torchvision import transforms as trn


preprocess = trn.Compose([
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
import misc.resnet as resnet


def main(params):
    net = getattr(resnet, params['model'])()
    net.load_state_dict(torch.load(os.path.join(params['model_root'], params['model'] + '.pth')))
    my_resnet = myResnet(net)

    my_resnet.cuda()
    my_resnet.eval()

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)  # 图片个数

    dir_att = params['output_dir'] + '_res152_7_14'

    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    for i, img in enumerate(imgs):
        # load the image  skimage.io.imread,读取图片是RGB格式，读取和保存是numpy格式，io.imread读取后的格式uint8(unsigned int)
        # skimage是H W　Ｃ　
        # cv2读取和存储格式是BGR，也是numpy格式
        I = skimage.io.imread(os.path.join(params['images_root'], img['filename']))
        # handle grayscale input images
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]  # np.newaxis插入新维度
            I = np.concatenate((I, I, I), axis=2)
        I = I.astype('float32') / 255.0
        I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
        I = preprocess(I)  # 数据归一化处理
        with torch.no_grad():
            tmp_fc, tmp_att = my_resnet(I, params['att_size'])
        np.savez_compressed(os.path.join(dir_att, img['filename']), feat=tmp_att.data.cpu().float().numpy())

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N))
    print('wrote ', params['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='/media/dmd/ours/mlw/rs/RSICD_Captions/dataset_rsicd.json')
    parser.add_argument('--output_dir', default='/media/dmd/ours/mlw/rs/RSICD_Captions/scene_feature/RSICD')

    # options
    parser.add_argument('--images_root', default='/media/dmd/ours/mlw/rs/RSICD_Captions/RSICD_images',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--att_size', default=7, type=int, help='14x14 or 7x7')
    parser.add_argument('--model', default='resnet152', type=str, help='resnet101, resnet152, vgg16')
    parser.add_argument('--model_root', default='/media/dmd/ours/mlw/project/m2/models', type=str,
                        help='model root')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
