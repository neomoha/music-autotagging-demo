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

import os, glob, json, pickle, argparse
import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA

def load_features(collection):
	files_features = []
	trackids = []
	for filename in sorted(glob.glob("preprocessed_features/%s/*.json" % collection)):
		features = json.load(open(filename))
		files_features.append(features)
		trackid = filename[filename.rfind("/")+1:filename.rfind(".")]
		trackids.append(trackid)
	dv = DictVectorizer(sparse=False)
	X = dv.fit_transform(files_features)
	#print dv.get_feature_names()
	print X.shape
	return dv, X, trackids

def pca(X, n_components=25):
	pca = PCA(n_components=n_components)
	return pca.fit_transform(X)

def load_metadata(collection):
	f = open("metadata/%s.tsv" % collection)
	metadata = {}
	for line in f:
		data = line.strip().split("\t")
		if len(data) == 7:
			trackid, tag, weight = data[0], data[-2], data[-1]
			if not metadata.has_key(trackid):
				metadata[trackid] = []
			metadata[trackid].append((tag, int(weight)))
	f.close()
	return metadata

def get_numerical_labels(metadata, X, trackids):
	L = []
	L2idx = {}
	idx = 1
	for i in range(X.shape[0]):
		row = []
		for tag,weight in metadata[trackids[i]]:
			if not L2idx.has_key(tag):
				L2idx[tag] = idx
				idx += 1
			row.append(L2idx[tag])
		L.append(row)
	return L, L2idx

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Feature selection and multilabel binarization')
	parser.add_argument('collection', help='Collection name (e.g.: majorminer)')
	args = parser.parse_args()
	
	# feature selection
	dv, X, trackids = load_features(args.collection)
	X = pca(X, 25)
	
	# metadata (labels) to numeric binarized matrix
	metadata = load_metadata(args.collection)
	L, L2idx = get_numerical_labels(metadata, X, trackids)
	Y = LabelBinarizer().fit_transform(L)
	
	feature_model = {
		"X": X,
		"Y": Y,
		"label2idx": L2idx,
		"trackids": trackids
	}
	
	if not os.path.exists("feature_models"):
		os.mkdir("feature_models")
	pickle.dump(feature_model, open("feature_models/%s.pickle" % args.collection, "w"))
