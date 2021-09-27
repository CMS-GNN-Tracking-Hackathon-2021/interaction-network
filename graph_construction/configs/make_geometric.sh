
#!bin/sh
cd ~ 
source gnnenv/bin/activate
cd /vols/cms/liv/cmsgnn  
#for file in configs/heptrkx_classic*
#do 
python build_graph.py configs/geometric0.5GeV.yaml --n-workers=16
#done
