# -*- coding: utf-8 -*-

# This file is part of music-autotagging-demo.
#
# music-autotagging-demo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# music-autotagging-demo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with music-autotagging-demo.  If not, see <http://www.gnu.org/licenses/>.

# Written by Mohamed Sordo (@neomoha)
# Email: mohamed ^dot^ sordo ^at^ gmail ^dot^ com
# Website: http://msordo.weebly.com

import argparse
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

_ALL_ = ["random", "decision_trees", "linear_svm", "gaussian_naive_bayes"]

def precision_recall_f1score(GT, Y_pred):
	precisions, recalls, f1scores = [], [], []
	for i in range(GT.shape[0]):
		precisions.append(precision_score(GT[i][:], Y_pred[i][:]))
		recalls.append(recall_score(GT[i][:], Y_pred[i][:]))
		f1scores.append(f1_score(GT[i][:], Y_pred[i][:]))
	precisions, recalls, f1scores = np.array(precisions), np.array(recalls), np.array(f1scores)
	precision = (np.mean(precisions), np.median(precisions), np.std(precisions))
	recall = (np.mean(recalls), np.median(recalls), np.std(recalls))
	f1score = (np.mean(f1scores), np.median(f1scores), np.std(f1scores))
	return precision, recall, f1score

def annotation_evaluation(collection, algorithm, model="track_based"):
	print "\n\n####Evaluating %s annotation predictions by algorithm '%s'####" % (model, algorithm)
	evaluation = {}
	filename = "classification/%s/confusion_matrix__%s.out" % (collection, algorithm)
	Y_pred = np.loadtxt(open(filename))
	GT = np.loadtxt(open("classification/%s/ground_truth.out" % collection))
	if model == "tag_based":
		GT = GT.transpose()
		Y_pred = Y_pred.transpose()
	precision, recall, f1score = precision_recall_f1score(GT, Y_pred)
	print "-----------------------------------------------------------"
	print "Precision: %s" % str(precision)
	print "Recall: %s" % str(recall)
	print "F1score: %s" % str(f1score)
	print "-----------------------------------------------------------"
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluate annotation predictions using track_based and tag_based measures')
	parser.add_argument('collection', help='Collection name (e.g.: majorminer)')
	parser.add_argument('-a', '--algorithm', nargs='?', default="all", help='algorithm to evaluate (default="all")')
	args = parser.parse_args()
	
	if args.algorithm == "all":
		for algorithm in _ALL_:
			annotation_evaluation(args.collection, algorithm, "track_based")
			annotation_evaluation(args.collection, algorithm, "tag_based")
	else:
		annotation_evaluation(args.collection, args.algorithm, "track_based")
		annotation_evaluation(args.collection, args.algorithm, "tag_based")
