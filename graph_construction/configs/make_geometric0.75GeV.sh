
#!bin/sh
cd ~ 
source gnnenv/bin/activate
cd /vols/cms/liv/cmsgnn/graph_construction/ 
#for file in configs/heptrkx_classic*
#do 
python build_graph.py configs/geometric0.75GeV.yaml --n-workers=8
#done
