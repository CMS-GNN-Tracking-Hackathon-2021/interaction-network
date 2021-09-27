# quick start
```
mkdir cmsgnn 
cd cmsgnn 
git clone https://github.com/CMS-GNN-Tracking-Hackathon-2021/interaction-network

#create virtual env 
python3 -m venv cmsgnn_venv
source cmsgnn_venv/bin/activate

#install requirements 
pip install -r requirements.txt
pip install -r requirements-cpu-linux.txt
#pip install -r requirements-gpu-linux.txt

# build graph 
cd graph_construction 
# in some of the config files, you will have to change the directory to the eos space where the data is 
python build_graph.py config/select config file 

# run interaction network
python run_interaction_network.py --pt=2


``` 

## interaction-network

This repo contains a version of the interaction network developed by Gage DeZoort, Savannah Thais et.al in their paper [Charged particle tracking via edge-classifying interactionnetworks](https://arxiv.org/abs/2103.16701). This was created for the [TrackML dataset](https://www.kaggle.com/c/trackml-particle-identification), and thei repository is available [here](https://github.com/GageDeZoort/interaction_network_paper). The code in this repo is slightly simplified and adapted to CMSSW data. The main difference is an adaptation to fit the CMSSW Phase 2 geometry. 


This repo is split into the following sections: 

## graph_construction
Builing the graphs. Beware that the parameters in the config files have been optimised for the TrackML data, not CMS data. 

## models
Code for the graph neural net 

## plotting 
Evaluation plots 

## Eos space
The EOS space for the hackathon, with prepared ntuples, csv files, built graphs, trained neural nets: 
/eos/cms/store/group/ml/GNNTrackingHackathon

## Building graphs in CMSSW
You can refer to this code as a starting point for porting the graph builidng to CMSSW. It does graph building in C++. https://github.com/leviBlinder/Graph_Construction_for_TrackML-C/blob/main/graph_construction/build_geometric.cc

