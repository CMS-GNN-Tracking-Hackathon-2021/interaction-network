"""
Data preparation script for GNN tracking.

This script processes h5 files of the ntuple and produces graph data on disk.
Will also save a csv file of the time to build each graph
The differences between Savannah's code is: 
    - CMS geometry instead of TrackML geometry (main difference is just the layers that connect to each other and their numbering)  
    - Intersecting lines capability removed (should be added back in) 
"""

# System
import os
import sys
import time
import argparse
import logging
import multiprocessing as mp
from functools import partial
sys.path.append("../")

# Externals
import yaml
import pickle
import numpy as np
import pandas as pd
import csv 

# Locals
from collections import namedtuple


Graph = namedtuple('Graph', ['x', 'edge_attr', 'edge_index', 'y', 'pid', 'pt', 'eta'])



# the following will create a list of accepted layer transistions

# there are 4 inner barrel layers 
l = np.arange(1,5)
# creates cobinations (1,2), (2,3) etc. 
layer_pairs = np.stack([l[:-1], l[1:]], axis=1)

n_det_layers = 18
# left_side endcap, creates (5,6), (6,7) etc. 
EC_L = np.arange(5, 17)
EC_L_pairs = np.stack([EC_L[:-1], EC_L[1:]], axis=1)
layer_pairs = np.concatenate((layer_pairs, EC_L_pairs), axis=0)
# right side endcap 
EC_R = np.arange(17, 29)
EC_R_pairs = np.stack([EC_R[:-1], EC_R[1:]], axis=1)
layer_pairs = np.concatenate((layer_pairs, EC_R_pairs), axis=0)

# transitions between any barrel layer and nearest endcap layer also allowed 
barrel_EC_L_pairs = np.array([(1,5), (2,5), (3,5), (4,5)])
barrel_EC_R_pairs = np.array([(1,17), (2,17), (3,17), (4,17)])
layer_pairs = np.concatenate((layer_pairs, barrel_EC_L_pairs), axis=0)
layer_pairs = np.concatenate((layer_pairs, barrel_EC_R_pairs), axis=0)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/geometric.yaml')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--task', type=int, default=0)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    add_arg('--start-evt', type=int, default=1000)
    add_arg('--end-evt', type=int, default=3000)
    return parser.parse_args()



# Construct the graph
def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

def select_segments(hits1, hits2, phi_slope_max, z0_max,
                    layer1, layer2):
    """
    Constructs a list of selected segments from the pairings
    between hits1 and hits2, filtered with the specified
    phi slope and z0 criteria.

    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """
    
    # Start with all possible pairs of hits
    hit_pairs = hits1.reset_index().merge(hits2.reset_index(), on='evt', suffixes=('_1', '_2'))

    #print(hit_pairs)
    # Compute line through the points
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    eta_1 = calc_eta(hit_pairs.r_1, hit_pairs.z_1)
    eta_2 = calc_eta(hit_pairs.r_2, hit_pairs.z_2)
    deta = eta_2 - eta_1
    dR = np.sqrt(deta**2 + dphi**2)
    phi_slope = dphi / dr
    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr


    
    # Filter segments according to phi slope and z0 criteria
    good_seg_mask = ((phi_slope.abs() < phi_slope_max) & 
                     (z0.abs() < z0_max))

    dr = dr[good_seg_mask]
    dphi = dphi[good_seg_mask]
    dz = dz[good_seg_mask]
    dR = dR[good_seg_mask]
    
    return hit_pairs[good_seg_mask], dr, dphi, dz, dR

def construct_graph(hits, layer_pairs, phi_slope_max, z0_max,
                    feature_names, feature_scale, evt="-1"):
    """Construct one graph (i.e. from one event)
        The graph contains: 
        - Node information: r, 
        - Edge information: dr, dR, dz, dphi 
        - Particle: Particle id, momentum and eta  
        - y label: 1 if a true edge, 0 otherwise 
    """
    
    t0 = time.time()
    # Loop over layer pairs and construct segments
    segments = []
    seg_dr, seg_dphi, seg_dz, seg_dR = [], [], [], []
    
    # for all accepted layer combinations construct segments 
    for (layer1, layer2) in layer_pairs:
        # Find and join all hit pairs for one combo of layers at a time 
        try:
            hits1 = hits[hits['layer_id']==layer1]
            hits2 = hits[hits['layer_id']==layer2]
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair
        except KeyError as e:
            logging.info('skipping empty layer: %s' % e)
            continue
        # Construct the segments
        selected, dr, dphi, dz, dR = select_segments(hits1, hits2, phi_slope_max, z0_max,
                                                     layer1, layer2)
        
    
        segments.append(selected)
        seg_dr.append(dr)
        seg_dphi.append(dphi)
        seg_dz.append(dz)
        seg_dR.append(dR)
        
    # Combine segments from all layer pairs
    #segmetns contains the index in the hit data frame of the two hits that may be connected 
    segments = pd.concat(segments)
    seg_dr, seg_dphi = pd.concat(seg_dr), pd.concat(seg_dphi)
    seg_dz, seg_dR = pd.concat(seg_dz), pd.concat(seg_dR)

    #print("hits", hits)
    #print("segments", segments)
    # Prepare the graph matrices
    n_hits = hits.shape[0]
    n_edges = segments.shape[0]
    
    # node information
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    # edge information 
    edge_attr = np.stack((seg_dr/feature_scale[0], 
                         seg_dphi/feature_scale[1], 
                          seg_dz/feature_scale[2], 
                          seg_dR))
    # initialise as zeros 
    y = np.zeros(n_edges, dtype=np.float32)
    
    # pytorch expects edge connections numbered from 0 to the number of edges
    # right now they are labelled by the hit id, so we need to convert
    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
    #seg start is a list of all the segment starts where the number is the edge as enumerated above  
    seg_start = hit_idx.loc[segments.index_1].values
    seg_end = hit_idx.loc[segments.index_2].values
    # connect starts and ends  
    edge_index = np.stack((seg_start, seg_end))
    pid = hits.particle_id
    pt = hits.sim_pt
    eta = hits.sim_eta
    
    # is is 1 if the segments have the same particle, otherwise 0 
    y = [int (x) for x in segments.particle_id_1 == segments.particle_id_2]
    
    print("... completed in {0} seconds".format(time.time()-t0))
    return Graph(X, edge_attr, edge_index, y, pid, pt, eta)

def select_hits(hits, pt_min=0):
    """Subset hits based on particle momentum and drop duplicate hits""" 
    hits = hits[hits['sim_pt'] > pt_min]
    # consider the row a duplicate if the particle has the same id and the hit has the same position and another row 
    hits = hits.drop_duplicates(subset=['particle_id', 'layer_id', 'x', 'y', 'z'])
    # if multiple hits in one layer, this selects the one with the smallest dxdy_sig value 
    # if the simdxy is the same for the same layer hits, it'll just select the first value. Should be changed to e.g. min r value 
    hits = hits.loc[hits.groupby(['particle_id', 'layer_id']).sim_dxy_sig.idxmin().values]
    return hits



def process_event(file, output_dir, pt_min, eta_range, phi_range, phi_slope_max, z0_max):
    """ 
    Calls all necessary functions to build graph 
    Inputs: file name, directory to write result to, range of eta and phi, max phi slope and z0 
    Returns: Savens the built graphs to output directory 
    """ 

    # Load the data
    data = pd.read_hdf(file)
    #subset by useful columns 
    data = data[['evt', 'hit_id', 'x', 'y', 'z', 'r', 'sim_pt', 'sim_eta', 'sim_phi', 'particle_id', 'volume_id', 'layer_id', 'sim_dxy_sig']]
    #calculate phi of hit 
    data['phi'] = np.arctan2(data.y, data.x)


    # extract the event number from file name 
    evt = int(file.split("ntuple_PU200_event")[1].split('.h5')[0].strip())
    #evt = int(evt_number)
    #evt = file 
    logging.info('Event %i, loading data' % evt)

    # Apply hit selection
    logging.info('Event %i, selecting hits' % evt)
    #apply pt cut and remove duplciates. Add new column for evt 
    hits = select_hits(data, pt_min=pt_min)
    

    # Graph features and scale
    feature_names = ['r', 'phi', 'z']
    
    #  
    feature_scale = np.array([1000., np.pi, 1000.])
    

    logging.info('Event %i, constructing graphs' % evt)
    graph = construct_graph(hits, layer_pairs=layer_pairs,
                              phi_slope_max=phi_slope_max, z0_max=z0_max,
                              feature_names=feature_names,
                              feature_scale=feature_scale,
                              evt=evt) 
    # Write these graphs to the output directory
    try:
        filename = os.path.join(output_dir,'event%s_g000' % (evt))

    except Exception as e:
        logging.info(e)
   
    logging.info('Event %i, writing graphs', evt)    
    np.savez(filename, ** dict(x=graph.x, edge_attr=graph.edge_attr,
                                   edge_index=graph.edge_index, 
                                   y=graph.y, pid=graph.pid, pt=graph.pt, eta=graph.eta))
        

def main():
    """Main function"""

    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)
    if args.task == 0:
        logging.info('Configuration: %s' % config)
    # Find the input files
    input_dir = config['input_dir']
    # list all files in input directory 
    all_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]

    # Prepare output
    output_dir = os.path.expandvars(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    logging.info('Writing outputs to ' + output_dir)

    # Process input files with a worker pool
    t0 = time.time()
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_event, output_dir=output_dir,
                               phi_range=(-np.pi, np.pi), **config['selection'])
        pool.map(process_func, all_files)
    t1 = time.time()
    print("Finished in", t1-t0, "seconds")

    
    # write timing results to a file 
    column_names = ['pt_min', 'z0_max', 'phi_max', 'total time', 'number of events', 'mean time per event']

    times = {'pt_min': config['selection']['pt_min'],
            'z0_max': config['selection']['z0_max'], 
            'phi_max': config['selection']['phi_slope_max'],  
            'total time': t1-t0,
            'number of events': len(all_files), 
            'mean time per event': (t1-t0)/len(all_files)} 
   
    timing_file = 'graph_building_timing.csv' 
    with open (timing_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=column_names)
        
        if not os.path.isfile(timing_file):
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow(times)


    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
