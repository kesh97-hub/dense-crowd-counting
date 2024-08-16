# DATASETS

1. UCF-QNRF dataset:
   The UCF-QNRF is a publicly available dataset and can be downloaded from the official [website](https://www.crcv.ucf.edu/data/ucf-qnrf/).
   
2. ShanghaiTech dataset:
   The ShanghaiTech dataset is also a publicly available dataset and can be downloaded from the official [GitHub](https://github.com/desenzhou/ShanghaiTechDataset?tab=readme-ov-file).
   
3. UCF-CC-50 dataset:
   The UCF-CC-50 dataset is also a publicly available dataset and can be downloaded from the official [website](https://www.crcv.ucf.edu/data/ucf-cc-50/).

# Organizing the dataset for this project.
1. Download the datasets from the respective official websites.
2. Extract the parent folder and place it inside this "Datasets" folder.

This project folder is organized as follows:

```markdown
Datasets/
├── UCF-QNRF_ECCV18/
│   └── Train/
│       ├── img_0001.jpg
│       ├── img_0001_ann.mat
│       └── ...
│   └── Test/
│       ├── img_0001.jpg
│       ├── img_0001_ann.mat
│       └── ...
├── ShanghaiTech_Crowd_Counting_Dataset/
│   └── part_A_final/
│       └── train_data/
│           └── images/
│               ├── IMG_1.jpg
│               └── ...
│           └── ground_truth/
│               ├── GT_IMG_1.mat
│               └── ...
│       └── test_data/
│           └── images/
│               ├── IMG_1.jpg
│               └── ...
│           └── ground_truth/
│               ├── GT_IMG_1.mat
│               └── ...
│   └── part_A_final/
│       └── train_data/
│           └── images/
│               ├── IMG_1.jpg
│               └── ...
│           └── ground_truth/
│               ├── GT_IMG_1.mat
│               └── ...
│       └── test_data/
│           └── images/
│               ├── IMG_1.jpg
│               └── ...
│           └── ground_truth/
│               ├── GT_IMG_1.mat
│               └── ...
├── UCF_CC_50 /
│   ├── 1.jpg
│   ├── 1_ann.mat
└── README.md
```

