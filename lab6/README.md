# Lab 6: Spark MLlib Clustering and Classification

The lab uses a reproducible synthetic customer-profile dataset with numeric and categorical features.

Pipeline:

- generate CSV dataset with 3000 records;
- preprocess features with Spark ML `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `StandardScaler`;
- compare K-Means, Bisecting K-Means and Gaussian Mixture clustering;
- interpret clusters;
- train Logistic Regression, Random Forest and GBT classifiers to predict membership in the largest cluster.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python spark_mllib_lab.py --out lab6_results
```

Generated datasets, metrics and plots are ignored by git.
