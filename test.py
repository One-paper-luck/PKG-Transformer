import random
from data import ImageDetectionsField, TextField, RawField
from data import Sydney, UCM, RSICD, DataLoader
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, PriorKnowledgeAugmentedAttention
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
random.seed(6)
torch.manual_seed(6)
np.random.seed(6)

def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            detections = images[0].to(device)
            detections_gl = images[1].to(device)

            with torch.no_grad():
                out, _ = model.beam_search(detections, detections_gl, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i

            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='PKG-Transformer')
    parser.add_argument('--exp_name', type=str, default='Sydney')
    # sydney
    parser.add_argument('--annotation_folder', type=str,
                        default='./datasets/Sydney_Captions')
    parser.add_argument('--scene_features_path', type=str,
                        default='./datasets/Sydney_Captions/features/scene_feature/Sydney_res152_7_14')
    parser.add_argument('--object_features_path', type=str,
                        default='./datasets/Sydney_Captions/features/object_feature/sydney50')

    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)

    args = parser.parse_args()

    print('PKG-Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(max_detections=50, load_in_tmp=False,
                                       scene_detections_path=args.scene_features_path,
                                       object_detections_path=args.object_features_path)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset  Sydney，UCM，RSICD
    if args.exp_name == 'Sydney':
        dataset = Sydney(image_field, text_field, 'Sydney/images/', args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'UCM':
        dataset = UCM(image_field, text_field, 'UCM/images/', args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'RSICD':
        dataset = RSICD(image_field, text_field, 'RSICD/images/', args.annotation_folder, args.annotation_folder)

    train_dataset, val_dataset, test_dataset = dataset.splits

    text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))


    # Model and dataloaders

    encoder = MemoryAugmentedEncoder(3, 0, attention_module=PriorKnowledgeAugmentedAttention,
                                     attention_module_kwargs={'m': args.m})
    decoder = MeshedDecoder(len(text_field.vocab), 127, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load('./saved_models/Sydney_best.pth')

    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, text_field)
    print(scores)
