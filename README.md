# Learning to segment from misaligned and partial labels
This repo supports the implementation of this paper:

@inproceedings{fobi2020learning,
  title={Learning to segment from misaligned and partial labels},
  author={Fobi, Simone and Conlon, Terence and Taneja, Jayant and Modi, Vijay},
  booktitle={Proceedings of the 3rd ACM SIGCAS Conference on Computing and Sustainable Societies},
  pages={286--290},
  year={2020}
}

## Getting Started

### Install Environment 
conda env create -f environment.yml

### Generate Image Patches
python dataprep/generate_patches.py

### Running Alignment Correction
python learning/alignment_correction.py

### Running Point Segmentation
python learning/point_segmentation.py



