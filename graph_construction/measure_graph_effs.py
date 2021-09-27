"""
Measures graph purity, efficiency, and number of particles before and after pt cut for all the graphs at a given pt cut
Writes the results to a csv file 
""" 



import os
import sys
import math
import argparse
import pickle
from os import listdir
from os.path import isfile, join
import multiprocessing as mp  
from multiprocessing import Process, Manager 
import csv 

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

import trackml
from trackml.dataset import load_event
from trackml.dataset import load_dataset
import itertools
from itertools import combinations

# break down input files into event_ids 
data_dir = '/home/lhv14/validationNtuple/ntuple_PU200/new_simids/'

# extract all event ids from the file names of the built graphs 
evt_ids = [f.split('ntuple_PU200_event')[1].split('.')[0] for f in listdir(data_dir)]

N_avg = 995
endcaps = True

parser = argparse.ArgumentParser(description='Measure Graph Construction Efficiencies')
parser.add_argument('--pt-min',type=str, default=str(2), 
                    help='pt_min for graph construction')
parser.add_argument('--construction', type=str, default='geometric', 
                    help='construction method: geometric, pre-clustering, or data-driven')
parser.add_argument('--train-sample', type=int, default=1,
                    help='train_<train-sample> used to generate graphs')
parser.add_argument('--N', type=int, default=995,
                    help='number of graphs use in calculation')
args = parser.parse_args()
print(' ... using args:\n', args)



# set up for writing to a csv 
graph_dir = '../output/'+args.pt_min+'GeV/'
print(" ... reading graphs from", graph_dir, '\n')
filename = "graph_measurement_result_"+args.pt_min+"GeV.csv" 
file_exists = os.path.isfile(filename)
result_file =open(filename, 'a')
column_names = ['efficiency', 'purity', 'graph_edges','n_true_edges', 'graph_nodes', 'n_particles_before_cut', 'n_particles_after_cut', 'graph_size']
writer = csv.DictWriter(result_file, delimiter=',', lineterminator='\n',fieldnames=column_names)

if not file_exists:
    print("file does not exist, create header")
    writer.writeheader()  # file doesn't exist yet, write a header

pt_cut = float(args.pt_min)
method = args.construction


# building the layer geometry, just like in the graph build script 
n_det_layers = 5
l = np.arange(1,n_det_layers)
layer_pairs = np.stack([l[:-1], l[1:]], axis=1)
n_det_layers = 18
EC_L = np.arange(5, 17)
EC_L_pairs = np.stack([EC_L[:-1], EC_L[1:]], axis=1)
layer_pairs = np.concatenate((layer_pairs, EC_L_pairs), axis=0)
EC_R = np.arange(17, 29)
EC_R_pairs = np.stack([EC_R[:-1], EC_R[1:]], axis=1)
layer_pairs = np.concatenate((layer_pairs, EC_R_pairs), axis=0)
barrel_EC_L_pairs = np.array([(4,5), (1,5), (2,5), (3,5)])
barrel_EC_R_pairs = np.array([(4,17), (1,17), (2,17), (3,17)])
layer_pairs = np.concatenate((layer_pairs, barrel_EC_L_pairs), axis=0)
layer_pairs = np.concatenate((layer_pairs, barrel_EC_R_pairs), axis=0)
valid_layer_pairs = layer_pairs 



truth = {}
purities, efficiencies = [], []
sizes, nodes, edges = [], [], []
counter = 0

def evaluate_graph(evtid): 
    """ Is applied to one built graph
    Writes efficiency, purity, number of nodes and edges, number of particles before and after cut, to csv" 
    """ 

    print("processing event ", evtid)  
    graph_path = graph_dir + 'event'+evtid + '_g000.npz'
    if not os.path.isfile(graph_path):
        print('didnt find graph')
        #continue
            
    with np.load(graph_path) as f:
        # load graph info 
        x = torch.from_numpy(f['x'])
        edge_attr = torch.from_numpy(f['edge_attr'])
        edge_index = torch.from_numpy(f['edge_index'])
        y = torch.from_numpy(f['y'])
        pid = torch.from_numpy(f['pid'])
        data = Data(x=x, edge_index=edge_index,
                    edge_attr=torch.transpose(edge_attr, 0, 1),
                    y=y, pid=pid)
        data.num_nodes = len(x)
        
        # memory size of the graph 
        size = sys.getsizeof(data.x) + sys.getsizeof(data.edge_attr) 
        size += sys.getsizeof(data.edge_index) + sys.getsizeof(data.y)
            
        n_edges, n_nodes = edge_index.shape[1], x.shape[0]
        
        # read the hit info  
        hits = pd.read_hdf(data_dir+'ntuple_PU200_event'+evtid+'.h5')
        n_particles_before = len(hits['particle_id'].unique())
        hits = hits[hits['sim_pt'] > pt_cut]

#        particles = hits[hits['sim_pt'] > pt_cut]
        
        
        r = np.sqrt(hits.x**2 + hits.y**2)
        phi = np.arctan2(hits.y, hits.x)
        
        
        particle_ids = hits.particle_id.unique()
        n_particles_after = particle_ids.shape[0]

        truth_edges_per_particle = {}
        truth_edges =  0
        hit_ids = np.unique(hits.hit_id)
        n_layers_hit = len(hit_ids)
        
#         need to find the number of actual true edges in the data  
        for particle in hits.particle_id.unique():
            particle_data = hits[hits['particle_id']==particle]
        #    # find all combination of layers possible for this particle 
            layer_combos = np.array(list(itertools.combinations(particle_data.layer_id, 2)))
            layers = np.array(particle_data.layer_id.values)
            lo, li = layers[:-1], layers[1:]
            layer_pairs = np.column_stack((lo, li))
            for lp in layer_pairs:
                if (lp==valid_layer_pairs).all(1).any():
                    truth_edges += 1
            truth_edges_per_particle[particle] = n_layers_hit-1
#            truth_edges += n_layers_hit-1 

        





        # number of edges labelled as true compared to actual true number of edges 
        efficiency = torch.sum(data.y).item()/truth_edges
    #    print('number of labelled true edges', torch.sum(data.y), 'number of counted true edges', truth_edges)
        # number of true edges to total number of edges 
        purity = torch.sum(data.y).item()/len(data.y)
        if (torch.sum(y).item()/truth_edges > 1.0): print('\nERROR: PURITY>1!\n')

        purities.append(purity)
        efficiencies.append(efficiency)
        sizes.append(size)
        nodes.append(n_nodes)
        edges.append(n_edges)

    result = {'efficiency':efficiency, 'purity':purity, 'graph_edges':n_edges, 'n_true_edges': truth_edges, 'graph_nodes':n_nodes, 'n_particles_before_cut':n_particles_before,  'n_particles_after_cut':n_particles_after, 'graph_size':size}
    writer.writerow(result)
    # we will be writing to a file during multiprocessing, which is bad practice
    # make sure to flush the file each time, otherwise it won't write 
    result_file.flush()

# multiprocessing 
pool = mp.Pool(processes=4) 
[pool.apply(evaluate_graph, args=(evtid,)) for evtid in evt_ids]
