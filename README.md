# DeeplyTough

This is the official PyTorch implementation of our paper *DeeplyTough: Learning Structural Comparison of Protein Binding Sites*, available from <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00554>.

![DeeplyTough overview figure](overview.png?raw=true "DeeplyTough overview figure.")

## Setup

### Code setup

The software is ready for Docker: the image can be created from `Dockerfile` with the `docker build` command (you may have to increase the disk image size available to the docker engine). The DeeplyTough tool is then accessible within `deeplytough` conda environment inside the container.

Alternatively, environment `deeplytough` can be created inside local [conda](https://conda.io/en/latest/miniconda.html) by executing the following steps from the root of this repository: 

```bash
conda create -y -n deeplytough python=3.6
conda install -y -n deeplytough -c acellera -c psi4 -c conda-forge htmd=1.13.10
apt-get -y install openbabel
conda create -y -n deeplytough_mgltools python=2.7
conda install -y -n deeplytough_mgltools -c bioconda mgltools

conda activate deeplytough
pip install --upgrade pip
pip install -r requirements.txt
git clone https://github.com/mariogeiger/se3cnn && cd se3cnn && git reset --hard 6b976bea4ea17e1bd5655f0f030c6e2bb1637b57 && mv experiments se3cnn; sed -i "s/exclude=\['experiments\*'\]//g" setup.py && python setup.py install && cd .. && rm -rf se3cnn
git clone https://github.com/AMLab-Amsterdam/lie_learn && cd lie_learn && python setup.py install && cd .. && rm -rf lie_learn
```

### Dataset setup

#### Training and benchmark datasets

The tool comes with built-in support for three datasets: TOUGH-M1 (Govindaraj and Brylinski, 2018), Vertex (Chen et al., 2016), and ProSPECCTs (Ehrt et al., 2018). These datasets must be downloaded if one wishes to either retrain the network or evaluate on one of these benchmarks. The datasets can be prepared in two steps:

1. Set `STRUCTURE_DATA_DIR` environment variable to a directory that will contain the datasets (about 27 GB): `export STRUCTURE_DATA_DIR=/path_to_a_dir`
2. Run `datasets_downloader.sh` from the root of this repository and get yourself a coffee

This will download PDB files, extracted pockets and pre-process input features. It will also download lists of pocket pairs provided by the respective dataset authors. By downloading Prospeccts, you accept their [terms of use](http://www.ccb.tu-dortmund.de/ag-koch/prospeccts/license_en.pdf).

Note that this is a convenience and we also provide code for data pre-processing: in case one wishes to start from the respective base datasets, pre-processing may be triggered using the `--db_preprocessing 1` flag when running any of our training and evaluation scripts. For the TOUGH-M1 dataset in particular, fpocket2 is required and can be installed as follows:
```bash
curl -O -L https://netcologne.dl.sourceforge.net/project/fpocket/fpocket2.tar.gz && tar -xvzf fpocket2.tar.gz && rm fpocket2.tar.gz && cd fpocket2 && sed -i 's/\$(LFLAGS) \$\^ -o \$@/\$\^ -o \$@ \$(LFLAGS)/g' makefile && make && mv bin/fpocket bin/fpocket2 && mv bin/dpocket bin/dpocket2 && mv bin/mdpocket bin/mdpocket2 && mv bin/tpocket bin/tpocket2
```

#### Custom datasets

The tool also supports an easy way of computing pocket distances for a user-defined set of pocket pairs. This requires providing i) a set of PDB structures, ii) pockets in PDB format (extracted around bound ligands or detected using any pocket detection algorithm), iii) a CSV file defining the pairing. A toy custom dataset example is provided in `datasets/custom`. The CSV file contains a quadruplet on each line indicating pairs to evaluate: `relative_path_to_pdbA, relative_path_to_pocketA, relative_path_to_pdbB, relative_path_to_pocketB`, where paths are relative to the directory containing the CSV file and the pdb extension may be omitted. `STRUCTURE_DATA_DIR` environment variable must be set to the parent directory containing the custom dataset (in the example `/path_to_this_repository/datasets`).

### Environment setup

To run the evaluation and training scripts, please first set the `DEEPLYTOUGH` environment variable to the directory containing this repository and then update the `PYTHONPATH` and `PATH` variables respectively:
```bash
export DEEPLYTOUGH=/path_to_this_repository
export PYTHONPATH=$DEEPLYTOUGH/deeplytough:$PYTHONPATH
export PATH=$DEEPLYTOUGH/fpocket2/bin:$PATH
```

## Evaluation

We provide pre-trained networks in the `networks` directory in this repository. The following commands assume a GPU and a 4-core CPU available; use `--device 'cpu'` if there is no GPU and set `--nworkers` parameter accordingly if there are fewer cores available.

* Evaluation on TOUGH-M1: 
```bash
python $DEEPLYTOUGH/deeplytough/scripts/toughm1_benchmark.py --output_dir $DEEPLYTOUGH/results --device 'cuda:0' --nworkers 4 --net $DEEPLYTOUGH/networks/deeplytough_toughm1_test.pth.tar
```

* Evaluation on Vertex: 
```bash
python $DEEPLYTOUGH/deeplytough/scripts/vertex_benchmark.py --output_dir $DEEPLYTOUGH/results --device 'cuda:0' --nworkers 4 --net $DEEPLYTOUGH/networks/deeplytough_vertex.pth.tar
```

* Evaluation on ProSPECCTs: 
```bash
python $DEEPLYTOUGH/deeplytough/scripts/prospeccts_benchmark.py --output_dir $DEEPLYTOUGH/results --device 'cuda:0' --nworkers 4 --net $DEEPLYTOUGH/networks/deeplytough_prospeccts.pth.tar
```

* Evaluation on a custom dataset, located in `$STRUCTURE_DATA_DIR/some_custom_name` directory: 
```bash
python $DEEPLYTOUGH/deeplytough/scripts/custom_evaluation.py --dataset_subdir 'some_custom_name' --output_dir $DEEPLYTOUGH/results --device 'cuda:0' --nworkers 4 --net $DEEPLYTOUGH/networks/deeplytough_toughm1_test.pth.tar
```
Note that networks `deeplytough_prospeccts.pth.tar` and `deeplytough_vertex.pth.tar` may also be used, producing different results.

Each of these commands will output to `$DEEPLYTOUGH/results` a CSV file with the resulting similarity scores (negative distances) as well as a pickle file with more detailed results (please see the code). The CSV files are already provided in this repository for conveniency.


## Training

Training requires a GPU with >=11GB of memory and takes about 1.5 days on recent hardware. In addition, at least a 4-core CPU is recommended due to volumetric input pre-processing being an expensive task.

* Training for TOUGH-M1 evaluation: 
```bash
python $DEEPLYTOUGH/deeplytough/scripts/train.py --output_dir $DEEPLYTOUGH/results/TTTT_forTough --device 'cuda:0' --seed 4
```

* Training for Vertex evaluation:
```bash
python $DEEPLYTOUGH/deeplytough/scripts/train.py --output_dir $DEEPLYTOUGH/results/TTTT_forVertex --device 'cuda:0' --db_exclude_vertex 'uniprot' --db_split_strategy 'none'
```

* Training for ProSPECCTs evaluation:
```bash
python $DEEPLYTOUGH/deeplytough/scripts/train.py --output_dir $DEEPLYTOUGH/results/TTTT_forProspeccts --device 'cuda:0' --db_exclude_prospeccts 'uniprot' --db_split_strategy 'none' --model_config 'se_4_4_4_4_7_3_2_batch_1,se_8_8_8_8_3_1_1_batch_1,se_16_16_16_16_3_1_2_batch_1,se_32_32_32_32_3_0_1_batch_1,se_256_0_0_0_3_0_2_batch_1,r,b,c_128_1'
```

Note that due to non-determinism inherent to the currently established process of training deep networks, it is nearly impossible to exactly reproduce the pre-trained networks in `networks` directory.

Also note the convenience of an output directory containing "TTTT" will afford this substring being replaced by the current `datetime`.

## Changelog

- 23.02.2020: Updated code to follow our revised [JCIM paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00554), in particular away moving from UniProt-based splitting strategy as in our [BioRxiv](https://www.biorxiv.org/content/10.1101/600304v1) paper to sequence-based clustering approach whereby protein structures sharing more than 30% sequence identity are always allocated to the same testing/training set. We have also made data pre-processing more robust and frozen the versions of several dependencies. The old code is kept in `old_bioarxiv_version` branch, though note the legacy splitting behavior can be turned on also in the current `master` by setting `--db_split_strategy` command line argument in the scripts to `uniprot_folds` instead of `seqclust`.

## License Terms

(c) BenevolentAI Limited 2019. All rights reserved.<br>
For licensing enquiries, please contact hello@benevolent.ai
