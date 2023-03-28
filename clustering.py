import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.model_selection import cross_val_score
import lightgbm as ltb
import shap

CLUSTER_NUMBER = 12


def clustering(df: pd.DataFrame):
    encoded_df = encoding_categorical_variables(df, ["MarkName", "TypeId", "FuelId", "GearId", "BodyId", "DriveId"])


    return df


def encoding_categorical_variables(df: pd.DataFrame, names):
    res = pd.get_dummies(df, columns=names, prefix=names)
    test = pd.get_dummies(df)
    return res


def umap_embedding(df: pd.DataFrame, numeric_names, categoric_names):
    numerical = df[numeric_names]

    for c in numerical.columns:
        pt = PowerTransformer()
        numerical.loc[:, c] = pt.fit_transform(np.array(numerical[c]).reshape(-1, 1))

    categorical = df[categoric_names]
    categorical = pd.get_dummies(categorical, columns=categoric_names, prefix=categoric_names)

    categorical_weight = len(df[categoric_names].columns) / df.shape[1]

    fit1 = umap.UMAP(metric='l2').fit(numerical)
    fit2 = umap.UMAP(metric='dice').fit(categorical)

    intersection = umap.umap_.general_simplicial_set_intersection(fit1.graph_, fit2.graph_, weight=categorical_weight)
    intersection = umap.umap_.reset_local_connectivity(intersection)

    # embedding_test = fit1 * fit2

    embedding = umap.umap_.simplicial_set_embedding(fit1._raw_data, intersection, fit1.n_components,
                                                    fit1._initial_alpha, fit1._a, fit1._b, fit1.repulsion_strength,
                                                    fit1.negative_sample_rate, 200, 'random', np.random, fit1.metric, fit1._metric_kwds, False,
                                                    densmap_kwds={},
                                                    output_dens=False)
    plt.clf()

    return embedding[0]


def eblow_cluster_numbers(X, numeric, categorical):
    X = pd.get_dummies(X, columns=categorical, prefix=categorical)

    # Pre-processing
    for c in X[numeric].columns:
        pt = PowerTransformer()
        X.loc[:, c] = pt.fit_transform(np.array(X[c]).reshape(-1, 1))


    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(2, 50)

    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)

        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_

    figure, axis = plt.subplots(2)

    axis[0].plot(K, inertias, 'bx-')
    axis[0].set_title('The Elbow Method using Inertia')

    axis[1].plot(K, distortions, 'bx-')
    axis[1].set_title('The Elbow Method using Distortion')

    plt.show()

def k_means_clustering_values(data: pd.DataFrame, numeric, categorical):
    origin_data = data.copy()

    data = pd.get_dummies(data, columns=categorical, prefix=categorical)

    # Pre-processing
    for c in data[numeric].columns:
        pt = PowerTransformer()
        data.loc[:, c] = pt.fit_transform(np.array(data[c]).reshape(-1, 1))

    kmeans = KMeans(n_clusters=CLUSTER_NUMBER).fit(data)
    return kmeans.labels_


def k_means_clustering(data: pd.DataFrame, numeric, categorical):
    origin_data = data.copy()

    data = pd.get_dummies(data, columns=categorical, prefix=categorical)

    # Pre-processing
    for c in data[numeric].columns:
        pt = PowerTransformer()
        data.loc[:, c] = pt.fit_transform(np.array(data[c]).reshape(-1, 1))

    kmeans = KMeans(n_clusters=CLUSTER_NUMBER).fit(data)
    kmeans_labels = kmeans.labels_

    embedding = umap_embedding(origin_data, numeric, categorical)

    fig, ax = plt.subplots()
    fig.set_size_inches((20, 10))

    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=2, c=kmeans_labels, cmap='tab20b', alpha=1.0)

    legend1 = ax.legend(*scatter.legend_elements(num=CLUSTER_NUMBER),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)

    plt.show()


def k_means_evaluation(data:pd.DataFrame, numeric_names, categorical_names):
    lgbm_data = data.copy()

    data = pd.get_dummies(data, columns=categorical_names, prefix=categorical_names)

    # Pre-processing
    for c in data[numeric_names].columns:
        pt = PowerTransformer()
        data.loc[:, c] = pt.fit_transform(np.array(data[c]).reshape(-1, 1))

    kmeans = KMeans(n_clusters=CLUSTER_NUMBER).fit(data)
    kmeans_labels = kmeans.labels_

    for c in lgbm_data[categorical_names]:
        lgbm_data[c] = lgbm_data[c].astype('category')

    clf_km = ltb.LGBMClassifier(colsample_by_tree=0.8)
    cv_scores_km = cross_val_score(clf_km, lgbm_data, kmeans_labels, scoring='f1_weighted')

    clf_km.fit(lgbm_data, kmeans_labels)

    explainer_km = shap.TreeExplainer(clf_km)
    shap_values_km = explainer_km.shap_values(lgbm_data)
    shap.summary_plot(shap_values_km, lgbm_data, plot_type="bar", plot_size=(15, 10))


def k_prototypes_evaluation(data:pd.DataFrame, numeric_names, categorical_names, categorical_indexes):
    lgbm_data = data.copy()

    kprot_data = data.copy()
    for c in data[numeric_names].columns:
        pt = PowerTransformer()
        kprot_data[c] = pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))

    kproto = KPrototypes(n_clusters=CLUSTER_NUMBER, init='Cao', n_jobs=4)
    clusters = kproto.fit_predict(kprot_data, categorical=categorical_indexes)

    for c in lgbm_data[categorical_names]:
        lgbm_data[c] = lgbm_data[c].astype('category')

    clf_km = ltb.LGBMClassifier(colsample_by_tree=0.8)
    cv_scores_km = cross_val_score(clf_km, lgbm_data, clusters, scoring='f1_weighted')

    clf_km.fit(lgbm_data, clusters)

    explainer_km = shap.TreeExplainer(clf_km)
    shap_values_km = explainer_km.shap_values(lgbm_data)
    shap.summary_plot(shap_values_km, lgbm_data, plot_type="bar", plot_size=(15, 10))


def k_prototypes_clustering(data: pd.DataFrame, numeric_names, categorical_indexes, categorical_names):
    kprot_data = data.copy()
    for c in data[numeric_names].columns:
        pt = PowerTransformer()
        kprot_data[c] = pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))

    kproto = KPrototypes(n_clusters=CLUSTER_NUMBER, init='Cao', n_jobs=4)
    clusters = kproto.fit_predict(kprot_data, categorical=categorical_indexes)

    embedding = umap_embedding(data, numeric_names, categorical_names)

    fig, ax = plt.subplots()
    fig.set_size_inches((20, 10))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=2, c=clusters, cmap='tab20b', alpha=1.0)

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(num=CLUSTER_NUMBER),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)

    plt.show()


def k_prototypes_clustering_values(data: pd.DataFrame, numeric_names, categorical_indexes, categorical_names):
    kprot_data = data.copy()
    for c in data[numeric_names].columns:
        pt = PowerTransformer()
        kprot_data[c] = pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))

    kproto = KPrototypes(n_clusters=CLUSTER_NUMBER, init='Cao', n_jobs=4)
    clusters = kproto.fit_predict(kprot_data, categorical=categorical_indexes)

    return clusters