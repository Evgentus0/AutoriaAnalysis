from sklearn import tree
from graphviz import Source
import pandas as pd
import numpy as np
from sklearn.tree import _tree



def create_decision_tree(data:pd.DataFrame, clusters):
    """
    !dot -Tpng tree.dot -o tree.png
    """

    dt_model = tree.DecisionTreeRegressor()
    dt_model.fit(data.values, clusters)
    score = dt_model.score(data.values, clusters)

    rules = get_rules(dt_model, data.columns, clusters);
    write_to_file("rules.txt", rules)

    # dotfile = open("tree2.dot", 'w')
    # tree.export_graphviz(dt_model, out_file=dotfile, feature_names=data.columns)
    # dotfile.close()
    # graph = Source(tree.export_graphviz(dt_model, out_file=None, feature_names=data.columns))
    # graph.format = 'png'
    # graph.render('dtree_render', view=True)


def get_rules(model, feature_names, class_names):

    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


def write_to_file(file_name, rules):
    with open(file_name, 'w') as f:
        for index, line in enumerate(rules):
            f.write(str(index + 1) + '. ' + line)
            f.write('\n')