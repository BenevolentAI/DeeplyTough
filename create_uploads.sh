DEST=/data/datab/dtout

cp -r ~/DeeplyTough/datasets/TOUGH-M1 $DEST
for FN in $DEST/TOUGH-M1/TOUGH-M1_dataset/*
do
	find $FN -maxdepth 1 -type f -delete
done
rm -f $DEST/TOUGH-M1/*.list
find $DEST/TOUGH-M1/TOUGH-M1_dataset -type f -name "*_full.pdb" -delete
find $DEST/TOUGH-M1/TOUGH-M1_dataset -type f -name "*_site_*.pdb" -delete

cp -r ~/DeeplyTough/datasets/Vertex $DEST
rm $DEST/Vertex/protein_pairs.tsv

cp -r ~/DeeplyTough/datasets/prospeccts $DEST
for FN in $DEST/prospieccts/*
do
	 rm -f $FN/*
done
find $DEST/prospeccts/ -type f -name "?????.pdb" -delete
cp ~/DeeplyTough/datasets/prospeccts/decoy/decoy_structures5.csv $DEST/prospeccts/decoy
cp ~/DeeplyTough/datasets/prospeccts/kahraman_structures/kahraman_structures80.csv $DEST/prospeccts/kahraman_structures
cp ~/DeeplyTough/datasets/prospeccts/kahraman_structures/phosphate_filter.sh $DEST/prospeccts/kahraman_structures


cp -r ~/DeeplyTough/datasets/processed $DEST


cd $DEST
mkdir -p zipVertex/processed/htmd
mv Vertex zipVertex
mv processed/htmd/Vertex zipVertex/processed/htmd
cd zipVertex
zip -r -9 ../dt_vertex.zip *


cd $DEST
mkdir -p zipTough/processed/htmd
mv TOUGH-M1 zipTough
mv processed/htmd/TOUGH-M1 zipTough/processed/htmd
cd zipTough
zip -r -9 ../dt_tough.zip *


cd $DEST
mkdir -p zipPro/processed/htmd
mv prospeccts zipPro
mv processed/htmd/prospeccts zipPro/processed/htmd
cd zipPro
zip -r -9 ../dt_prospeccts.zip *
