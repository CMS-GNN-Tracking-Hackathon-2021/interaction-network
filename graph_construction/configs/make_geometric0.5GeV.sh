
#!bin/sh
cd ~ 
source gnnenv/bin/activate
cd /vols/cms/liv/cmsgnn/graph_construction/  
#for file in configs/heptrkx_classic*
#do 
start=$(date +%s)
python build_graph.py configs/geometric0.5GeV.yaml --n-workers=4
end=$(date +%s)

echo "runtime was $[end - start] seconds" 
#done
