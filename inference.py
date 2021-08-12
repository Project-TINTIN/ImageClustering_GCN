import argparse, time, os, pickle
import numpy as np
import pandas as pd
import pathlib

import dgl
import torch
import torch.optim as optim

import cv2

from models import LANDER
from dataset import LanderDataset
from utils import evaluation, decode, build_next_level, stop_iterating

###########
# ArgParser
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--model_filename', type=str, default='lander.pth')
parser.add_argument('--faiss_gpu', action='store_true')
parser.add_argument('--num_workers', type=int, default=0)

# HyperParam
parser.add_argument('--knn_k', type=int, default=10)
parser.add_argument('--levels', type=int, default=1)
parser.add_argument('--tau', type=float, default=0.5)
parser.add_argument('--threshold', type=str, default='prob')
parser.add_argument('--metrics', type=str, default='pairwise,bcubed,nmi')
parser.add_argument('--early_stop', action='store_true')

# Model
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--num_conv', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--gat', action='store_true')
parser.add_argument('--gat_k', type=int, default=1)
parser.add_argument('--balance', action='store_true')
parser.add_argument('--use_cluster_feat', action='store_true')
parser.add_argument('--use_focal_loss', action='store_true')
parser.add_argument('--use_gt', action='store_true')

# Subgraph
parser.add_argument('--batch_size', type=int, default=4096)

args = parser.parse_args()
print(args)

###########################
# Environment Configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

##################
# Data Preparation
root_folder = "D:/Data/GE2020/embed/"
test_data = "D:/Data/GE2020/EmbedList.txt"
data_labels = pd.read_csv(test_data)
num_data = len(data_labels)
labels = np.empty(num_data)
features = np.empty((num_data, 512))

for idx in range(num_data):
    labels[idx] = data_labels.iloc[idx, 1]
    embed_file = root_folder + data_labels.iloc[idx, 0]
    if idx%1000 == 0:
        print("Loading " + embed_file)
    curr_feature = np.load(embed_file)
    features[idx] = curr_feature.flatten()
    
global_features = features.copy()
dataset = LanderDataset(features=features, labels=labels, k=args.knn_k,
                        levels=1, faiss_gpu=args.faiss_gpu)
g = dataset.gs[0]
g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
global_labels = labels.copy()
ids = np.arange(g.number_of_nodes())
global_edges = ([], [])
global_peaks = np.array([], dtype=np.int64)
global_edges_len = len(global_edges[0])
global_num_nodes = g.number_of_nodes()

fanouts = [args.knn_k-1 for i in range(args.num_conv + 1)]
sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
# fix the number of edges
test_loader = dgl.dataloading.NodeDataLoader(
    g, torch.arange(g.number_of_nodes()), sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers
)

##################
# Model Definition
if not args.use_gt:
    feature_dim = g.ndata['features'].shape[1]
    model = LANDER(feature_dim=feature_dim, nhid=args.hidden,
                   num_conv=args.num_conv, dropout=args.dropout,
                   use_GAT=args.gat, K=args.gat_k,
                   balance=args.balance,
                   use_cluster_feat=args.use_cluster_feat,
                   use_focal_loss=args.use_focal_loss)
    model.load_state_dict(torch.load(args.model_filename))
    model = model.to(device)
    model.eval()

# number of edges added is the indicator for early stopping
num_edges_add_last_level = np.Inf
##################################
# Predict connectivity and density
for level in range(args.levels):
    if not args.use_gt:
        total_batches = len(test_loader)
        for batch, minibatch in enumerate(test_loader):
            input_nodes, sub_g, bipartites = minibatch
            sub_g = sub_g.to(device)
            bipartites = [b.to(device) for b in bipartites]
            with torch.no_grad():
                output_bipartite = model(bipartites)
            global_nid = output_bipartite.dstdata[dgl.NID]
            global_eid = output_bipartite.edata['global_eid']
            g.ndata['pred_den'][global_nid] = output_bipartite.dstdata['pred_den'].to('cpu')
            g.edata['prob_conn'][global_eid] = output_bipartite.edata['prob_conn'].to('cpu')
            torch.cuda.empty_cache()
            if (batch + 1) % 10 == 0:
                print('Batch %d / %d for inference' % (batch, total_batches))

    new_pred_labels, peaks,\
        global_edges, global_pred_labels, global_peaks = decode(g, args.tau, args.threshold, args.use_gt,
                                                                ids, global_edges, global_num_nodes,
                                                                global_peaks)
    ids = ids[peaks]
    new_global_edges_len = len(global_edges[0])
    num_edges_add_this_level = new_global_edges_len - global_edges_len
    if stop_iterating(level, args.levels, args.early_stop, num_edges_add_this_level, num_edges_add_last_level, args.knn_k):
        break
    global_edges_len = new_global_edges_len
    num_edges_add_last_level = num_edges_add_this_level

    # build new dataset
    features, labels, cluster_features = build_next_level(features, labels, peaks,
                                                          global_features, global_pred_labels, global_peaks)
    # After the first level, the number of nodes reduce a lot. Using cpu faiss is faster.
    dataset = LanderDataset(features=features, labels=labels, k=args.knn_k,
                            levels=1, faiss_gpu=False, cluster_features = cluster_features)
    g = dataset.gs[0]
    g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
    g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
    test_loader = dgl.dataloading.NodeDataLoader(
        g, torch.arange(g.number_of_nodes()), sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

res_folder_root = "D:/Data/GE2020/results/"
image_folder_root = "D:/Data/GE2020/" 
test_result_txt = "D:/Data/GE2020/test_results.txt"
test_result_arr_txt = "D:/Data/GE2020/test_results_arr.txt"
for peakNum in range(len(global_peaks)):
    res_folder = res_folder_root + "cluster_" + str(peakNum)
    pathlib.Path(res_folder).mkdir(parents=True, exist_ok=True)

count_arr = np.zeros(len(global_peaks))
with open(test_result_txt, 'w') as f:
    count = 0
    for pred in global_pred_labels:
        embed_file = data_labels.iloc[count, 0]
        image_file = embed_file
        ext = embed_file.rfind(".jpg")
        if ext == -1:
            ext = embed_file.rfind(".png")
        if ext != -1:
            box_str = image_file[ext+4:]
            image_file = image_file[: ext+4]
            image = cv2.imread(image_folder_root + image_file)
            
            bbox = np.zeros(5)
            temp = box_str.rfind(".npy")
            if temp == -1:
                print(box_str)
                break
            box_str = box_str[:temp]
            for i in range(5):
                temp = box_str.rfind("_")
                if temp == -1:
                    print(box_str)
                    break
                bbox[4-i] = float(box_str[temp+1:])
                box_str = box_str[:temp]
            
            color = (0, 0, 255)
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(image, str(bbox[4]), (int(bbox[0])-1, int(bbox[1])-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)
            
            output_path = res_folder_root + "cluster_" + str(pred) + "/" + image_file.replace("/", "_")
            if count%100 == 0:
                print(output_path)
            cv2.imwrite(output_path, image)
        
        f.write(image_file + ", " + str(pred) + "\n")
        count = count + 1
        count_arr[pred] = count_arr[pred] + 1

with open(test_result_arr_txt, 'w') as f:
    for idx in range(len(global_peaks)):
        f.write(str(count_arr[idx]) + "\n")

