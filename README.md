# Potential Isomorphism Propagation Entity Alignment
Paper for 2024

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

## Acknowledgement
We appreciate [PEEA](https://github.com/OceanTangWei/PEEA) for their open-source contributions and the enocder hyperparameters follow their paper.
