import data_reader as reader
import helper
import clustering
import pandas as pd
import decision_tree as dtree
from sklearn.preprocessing import LabelEncoder

data = reader.get_data_csv()

numeric_names = ["Year", "PriceUsd", "RaceKmThs"]
categorical_names = ["MarkName", "TypeId", "FuelId", "GearId", "BodyId", "DriveId", "HasDamage"]
categorical_indexes = [0, 3, 5, 6, 7, 8, 9]

def describe_means():
    for name in ["Year", "PriceUsd", "RaceKmThs"]:
        count, min, max, mean, sd, quantile1, quantile2, quantile3 = helper.get_means(data, name)
        print(name)
        print("count:" + str(count) + "\n" +
              "min:" + str(min) + "\n" +
              "max:" + str(max) + "\n" +
              "mean:" + str(mean) + "\n" +
              "sd:" + str(sd) + "\n" +
              "quantile1:" + str(quantile1) + "\n" +
              "quantile2:" + str(quantile2) + "\n" +
              "quantile3:" + str(quantile3) + "\n")


def fuel_id_distribution():
    res = data.groupby(['FuelId'])['FuelId'].count()

    print(res)


def type_id_distribution():
    res = data.groupby(['TypeId'])['TypeId'].count()

    print(res)


def gear_id_distribution():
    res = data.groupby(['GearId'])['GearId'].count()

    print(res)


def body_id_distribution():
    res: pd.Series = data.groupby(['BodyId'])['BodyId'].count()

    print(res)


def drive_id_distribution():
    res: pd.Series = data.groupby(['DriveId'])['DriveId'].count()

    print(res)


def has_damage_distribution():
    res: pd.Series = data.groupby(['HasDamage'])['HasDamage'].count()

    print(res)


def model_distribution():
    res: pd.Series = data.groupby(['MarkName'])['MarkName'].count() \
                             .reset_index(name='count') \
                             .sort_values(['count'], ascending=False) \

    print(res)


# describe_means()
# fuel_id_distribution()
# type_id_distribution()
# gear_id_distribution()
# body_id_distribution()
# drive_id_distribution()
# has_damage_distribution()
# model_distribution()

# clustering.umap_embedding(data, numeric_names, categorical_names)

# clustering.eblow_cluster_numbers(data, numeric_names, categorical_names)

# clustering.k_means_clustering(data, numeric_names, categorical_names)

# clustering.k_prototypes_clustering(data, numeric_names, categorical_indexes, numeric_names)

# clustering.k_means_evaluation(data, numeric_names, categorical_names)

# clustering.k_prototypes_evaluation(data, numeric_names, categorical_names, categorical_indexes)

clusters = clustering.k_means_clustering_values(data, numeric_names, categorical_names)
tree_data = data.copy()
tree_data = pd.get_dummies(tree_data, columns=categorical_names, prefix=categorical_names)
dtree.create_decision_tree(tree_data, clusters)
