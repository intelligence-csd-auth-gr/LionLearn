import math


def roundup(a, digits=0):
    """roundup function is used to round up a float number by specific number of digits. Ex: roundup(0.005435,3)->0.006
    Args:
        a: The float number
        digits: the number of digits of the
    Return:
        upper rounded float number
    """
    n = 10 ** -digits
    return round(math.ceil(a / n) * n, digits)

def path_similarity(path_1, path_2, feature_names, min_max_feature_values):
    """path_similarity function computes the similarity of two paths (rules)
    Args:
        path_1: the first path
        path_2: the second path
        feature_names: the list of features
        min_max_feature_values: the min and max possible values of each feature
    Return:
        similarity: the similarity of the paths
    """
    similarity = 0
    for i in feature_names:
        if i in path_1 and i in path_2:
            if len(path_1[i]) == 2:
                l1 = path_1[i][1][1]
                u1 = path_1[i][0][1]
            else:
                if path_1[i][0][0] == '<=':
                    u1 = path_1[i][0][1]
                    l1 = min_max_feature_values[i][0]
                else:
                    l1 = path_1[i][0][1]
                    u1 = min_max_feature_values[i][1]
            if len(path_2[i]) == 2:
                l2 = path_2[i][1][1]
                u2 = path_2[i][0][1]
            else:
                if path_2[i][0][0] == '<=':
                    u2 = path_2[i][0][1]
                    l2 = min_max_feature_values[i][0]
                else:
                    l2 = path_2[i][0][1]
                    u2 = min_max_feature_values[i][1]
            if u1 <= l2 or u2 <= l2:
                similarity = similarity
            else:
                inter = min(u1, u2) - max(l1, l2)
                union = max(u1, u2) - min(l1, l2)
                if union != 0:
                    similarity = similarity + inter / union
        elif i not in path_1 and i not in path_2:
            similarity = similarity + 1
    similarity = similarity / len(feature_names)
    return similarity

def path_distance(path_1, path_2, feature_names, min_max_feature_values):
    """path_distance function computes the distance of two paths (rules)
    Args:
        path_1: the first path
        path_2: the second path
        feature_names: the list of features
        min_max_feature_values: the min and max possible values of each feature
    Return:
        distance: the distance of the paths
    """
    distance = 0
    feature_count = 0
    for i in feature_names:
        if i in path_1 and i in path_2:
            if len(path_1[i]) == 2:
                l1 = path_1[i][1][1]
                u1 = path_1[i][0][1]
            else:
                if path_1[i][0][0] == '<=':
                    u1 = path_1[i][0][1]
                    l1 = min_max_feature_values[i][0]
                else:
                    l1 = path_1[i][0][1]
                    u1 = min_max_feature_values[i][1]
            if len(path_2[i]) == 2:
                l2 = path_2[i][1][1]
                u2 = path_2[i][0][1]
            else:
                if path_2[i][0][0] == '<=':
                    u2 = path_2[i][0][1]
                    l2 = min_max_feature_values[i][0]
                else:
                    l2 = path_2[i][0][1]
                    u2 = min_max_feature_values[i][1]
            distance = distance + (1 / 2) * (abs(l1 - l2) + abs(u1 - u2))
            feature_count = feature_count + 1
        elif i in path_1 or i in path_2:
            distance = distance + 1
            feature_count = feature_count + 1
    if feature_count != 0:
        distance = distance / feature_count
    else:
        distance = 0
    return distance