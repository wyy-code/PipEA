# Potential Isomorphism Propagation Entity Alignment
This is the code of PipEA introduced in our paper, which is based on PEEA encoder.

## Datasets

The dataset can be unzipped by data.zip in your root.

* ref_ent_ids: testing entity pairs;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG;


## Just run PIPEA.py for the results in our paper

The environment is provided in requirements.txt.

## Update 2024.2.2: Large scale Dataset with special strategy

We also provide faster large-scale strategy for 100K and larger datasets. If you want to apply PipEA on large-scale KG, you can modify and run test.py

You can also apply 15K datasets based on the test.py, which only needs to generate embeddings firstly. The test.py will faster than original code.

Notably, the application on the large-scale KG need tf 2.x, which is different from original environment.

Our large scale environment is:
* fbpca
* tensorflow == 2.4.1
* Python == 3.6.5

If you find this repo useful, please cite us:
```bigquery
@article{sun2025understanding,
  title={Understanding and guiding weakly supervised entity alignment with potential isomorphism propagation},
  author={Sun, Haifeng and Wang, Yuanyi and Li, Han and Tang, Wei and Zhuang, Zirui and Qi, Qi and Wang, Jingyu},
  journal={ACM Transactions on Knowledge Discovery from Data},
  volume={19},
  number={7},
  pages={1--28},
  year={2025},
  publisher={ACM New York, NY}
}
```
