#!/usr/bin/env bash

if [ -z "$STRUCTURE_DATA_DIR" ]; then
  echo "STRUCTURE_DATA_DIR not set"
  exit
fi
if ! type wget > /dev/null; then
  echo "wget not installed"
  exit
fi
if ! type unzip > /dev/null; then
  echo "unzip not installed"
  exit
fi

cd $STRUCTURE_DATA_DIR

# TOUGH-M1 dataset
mkdir TOUGH-M1
wget https://zenodo.org/record/3687317/files/dt_tough.zip?download=1 -O dt_tough.zip && unzip dt_tough.zip
rm dt_tough.zip
wget https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/L7H7JJ/UFO5CB -O official_tough_m1.tar.gz && tar -xvzf official_tough_m1.tar.gz -C TOUGH-M1
rm official_tough_m1.tar.gz
wget https://osf.io/tmgne/download -O TOUGH-M1/TOUGH-M1_positive.list
wget https://osf.io/6dn5s/download -O TOUGH-M1/TOUGH-M1_pocket.list
wget https://osf.io/3aypv/download -O TOUGH-M1/TOUGH-M1_negative.list

# Vertex dataset
mkdir Vertex
wget https://zenodo.org/record/3687317/files/dt_vertex.zip?download=1 -O dt_vertex.zip && unzip dt_vertex.zip
rm dt_vertex.zip
wget http://pubs.acs.org/doi/suppl/10.1021/acs.jcim.6b00118/suppl_file/ci6b00118_si_002.zip && unzip ci6b00118_si_002.zip -d Vertex
rm ci6b00118_si_002.zip

# ProSPECCTs
mkdir prospeccts
for FILE in kahraman_structures.tar.gz identical_structures.tar.gz identical_structures_similar_ligands.tar.gz barelier_structures.tar.gz decoy.tar.gz review_structures.tar.gz NMR_structures.tar.gz
do
    wget www.ewit.ccb.tu-dortmund.de/ag-koch/prospeccts/ --post-data "file=${FILE}&licenseagreement=accept&action=Download" -O $FILE && tar -xvzf $FILE -C prospeccts
    rm $FILE
done
wget https://zenodo.org/record/3687317/files/dt_prospeccts.zip?download=1 -O dt_prospeccts.zip && unzip dt_prospeccts.zip
rm dt_prospeccts.zip


