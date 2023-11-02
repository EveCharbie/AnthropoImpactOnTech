# Introduction
This repository contains the code generating the optimal control problems (OCP) and results of a study entitled "Should all athletes use the same twisting strategy? The role of anthropometry in the personalization of optimal acrobatic techniques" (currently available in [preprint](https://doi.org/10.51224/SRXIV.337)). The OCP generates optimal kinematics of a double forward pike somersault ending with 1.5 or 2.5 twists for models representing the anthropometry of 18 athletes. The goal is then to compare the optimal kinematics between anthropometry. Uniquely, we also introcude a new metric assessing the twist potential of atheltes using forward simulations.

# How to cite
If you use part of the code available here, please cite:
```bibtex
@misc{Charbonneau2023,
  author = {Charbonneau, Eve and Sechoir, Lisa and Pascoa, Francisco and Begon, Mickael},
  title = {The role of anthropometry in the personalization of optimal acrobatic techniques},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://zenodo.org/records/10067374}}
}
```

# Requirements
In order to run the OCP and analysis of the results, you need to install the environment:
```bash
conda env create -f environment.yml
```

# Contact
If you would like to discuss research or have any question do not hesitate to communicate with me through email :)
[eve.charbonneau.1@umontreal.ca]
