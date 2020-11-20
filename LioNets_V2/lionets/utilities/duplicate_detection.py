from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances
import numpy as np

def double_distance_detector(X, distance_1_type = manhattan_distances, distance_2_type = cosine_distances):
	temp_X = X.copy()
	dict_distance_1 = {}
	distance_1 = distance_1_type([temp_X[0]],temp_X)
	for i in range(0,len(distance_1[0])):
		dict_distance_1.setdefault(distance_1[0][i], []).append(i)
	del distance_1
	possible_duplicates = []
	possible_duplicates_index = []
	for i in dict_distance_1:
		if len(dict_distance_1[i]) > 1:
			for j in dict_distance_1[i]:
				possible_duplicates.append(temp_X[j])
				possible_duplicates_index.append(j)
	del dict_distance_1
	dict_distance_2 = {}
	if not possible_duplicates_index:
		return X #There are no duplicates
	distance_2 = distance_2_type([temp_X[0]],possible_duplicates)
	for i in range(0,len(distance_2[0])):
		dict_distance_2.setdefault(distance_2[0][i], []).append(possible_duplicates_index[i])
	del distance_2, possible_duplicates, possible_duplicates_index
	to_delete = []
	for i in dict_distance_2:
		for j in range(1,len(dict_distance_2[i])):
			to_delete.append(dict_distance_2[i][j])
	del dict_distance_2
	temp_X = np.delete(temp_X, to_delete, axis=0)
	return temp_X     