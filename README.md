# Potential Isomorphism Propagation Entity Alignment
Paper for 2024

This is the code of PipEA introduced in our paper, which is based on PEEA encoder.

## Datasets

The dataset can be unzipped by data.zip in your root.

* ref_ent_ids: testing entity pairs;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG;


## Just run PIPEA.py for the results in our paper

## Large scale Dataset

We also provide sparse strategy for 100K and larger datasets. The main code is based on " refina_tf_batch " in gutils.py. If you want to apply PipEA on large-scale KG, you can modify and run test.py

