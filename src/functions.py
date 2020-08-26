#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:07:39 2019

@author: abhinavkaushik
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import warnings
from random import *
import itertools
import fcsparser
import umap
import faiss
import pickle
import joblib
import time
import datetime
import re
import math
import collections
from itertools import compress
import numpy as np
from collections import Counter
import pandas as pa
from src.fastKDE import *
from scipy import mean
from scipy import stats
from scipy import nanstd
from scipy import nanmean
from scipy.stats import sem, t
from scipy import spatial
from scipy.spatial import Delaunay
from scipy.sparse import *
from matplotlib.pyplot import *
from itertools import chain
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns; sns.set(color_codes=True)
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture as GMM
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
##from thundersvm import *
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from xgboost import plot_tree
'''/////// Function ////////'''

class Sample:
    X = None
    y = None
    z = None
    l = None
    o = None
    p = None
    q = None
    def __init__(self, X, y = None, z = None, l = None, o = None, p = None, q=None):
        self.X = X ## expression (and meta data)
        self.y = y ## orignal index
        self.z = z ## column header (marker and meta data)
        self.l = l
        self.o = o
        self.p = p
        self.q = q

def method0(handgated,filterCells,relevantMarkers,cellCount,head,ic, Findungated,ProjectName,Fileinfo):
    ## this function read the handgated file info and load the csv/fcs files from the path given and creates a dic with key as sample_id and value as xpression set and celltypes for each row in expression set
    ct2expr = {}
    TotalCount = {}
    md = pa.read_csv(handgated,header=0)
    PersampleAllGated = {}
    inputs = md.iloc[:,0]
    labels = md.iloc[:,1]
    sample_ids = md.iloc[:,2]
    fl = open(ProjectName + "/output.log", "a")
    fl.write("**** Starting Cyanno ****\n[Arguments]:\n")
    fl.write("Handgated file: " + str(handgated) + '\n')
    fl.write("Live cells file: "+ str(Fileinfo) + '\n')
    fl.write("cell count: "+ str(cellCount) + '\n')
    fl.write("Relevant Markers: " + str(relevantMarkers) + '\n')
    fl.write("filter cells: " + str(filterCells) + '\n')
    fl.write("Find ungated cells: " + str(Findungated) + '\n')
    fl.write("\nNow loading handgated cells for model training\n")
    ct2indx = {CT:(i+1) for i, CT in enumerate(list(np.unique(labels)))} ## +1 ensures that first celltype will not get 0 as cell typen name ; this is helpful in SVM
    if 'Unknown' in ct2indx:
        ct2indx['Unknown'] = 0
    if 'Ungated' in ct2indx:
        ct2indx['Ungated'] = 0
    if 'Other' in ct2indx:
        ct2indx['Other'] = 0
    if 'Undefined' in ct2indx:
        ct2indx['Undefined'] = 0
    if 'Unclassified' in ct2indx:
        ct2indx['Unclassified'] = 0
    indx2ct = {value:key for key, value in ct2indx.items() } ## get celltype name back from its index
    print ("Reading handgated cells...")
    PIDs = []
    if Findungated:
        PIDs = getPIDnames(md) ## list of sampleIDs that are needed to be analyzed for ungated cell identification
        UNKcount = dict(zip(sample_ids,[0]* md.shape[0])) ## a dict in which key is sample_id and value is zero which will be updated by the total number of ungated cells present in this sample ID 
    for c,f in enumerate(inputs):
        if os.path.exists(f):
            match = re.search(r'fcs$', f)
            match2 = re.search(r'csv$', f)
            key=md.iloc[c,2]
            #cellType = labels[c]
            cellType = ct2indx[labels[c]] ## instead of actual celltypr name use a number to
            if match: ## if file is FCS
                panel, exp = fcsparser.parse(f, reformat_meta=True)
                desc = panel['_channels_']['$PnS'] ## marker name instead of meta name
                desc = desc.str.replace('.*_', '', regex=True)
                desc = desc.str.replace('-', '.', regex=True).tolist()
                exp.columns = desc
                print(key,labels[c])
            elif match2: ## if file is CSV
                exp = pa.read_csv(f, delimiter=",", header=head, index_col=ic)
                print(key,labels[c])
            else:
                print(str("[ERROR] Line: " + c + "- Unknown File format, must be CSV or FCS only!!!"))
                fl.write("[ERROR] Line: " + str(c) + "- Unknown File format, must be CSV or FCS only!!!\n")
                sys.exit(0)
            exp = exp.loc[:,relevantMarkers]
            if Findungated is True and (key in PIDs): ## for these samples ungated cells needs to be calculated so we are storing their all gated cells expression values  
                if key in PersampleAllGated:
                    PersampleAllGated[key] = pa.concat([PersampleAllGated[key],exp])
                else:
                    PersampleAllGated[key] = exp
            if Findungated is True and (labels[c] in ["Ungated","Unknown","Unclassified","Other","Undefined"]): ## this will be used later to plot the proportions of ungated cells present in each sample used in training dataset 
                UNKcount[key] = UNKcount[key] + exp.shape[0] ## updating the number of ungated cells observed in this sample_id, if found 
            TotalCount.update({key: TotalCount.get(key, 0) + exp.shape[0] }) ## Total cells found in each sample_id
            indxrm = None
            if filterCells == True:
                indxrm = filterCells(exp) ## this will give indexes to remove bad quality cell;
                exp.drop(indxrm, axis = 0, inplace=True) ## this will remove the bad quality cells
            if cellType in ct2expr.keys(): ## checking if this celltype has already been used for gathering expression from other sample_id
                exp = pa.concat([ct2expr[cellType].X,exp]) ## concatenate (rbind) with existsing expression dataframe
                ct2expr[cellType] = Sample(X=exp)
            else:
                ct2expr[cellType] = Sample(X=exp)                  
        else:
            print(str("[ERROR] File " + f + " does not exists"))
            fl.write(str("[ERROR] File " + f + " does not exists\n"))
            
    tmp= {}
    for ct,expr in ct2expr.items():
        if expr.X.shape[0] >= cellCount: ## removing celltypes with poor cell count; less equal to than cellCount
            tmp[ct] = expr
        else:
            print("[Warning]" , str(indx2ct[ct]), "has low number of cells (" + str(expr.X.shape[0]) + " cells). To include this cell type reduce the value 'cellCount' argument")
            fl.write("[Warning]" + str(indx2ct[ct]) + "has low number of cells (" + str(expr.X.shape[0]) + " cells). To include this cell type reduce the value of 'cellCount' argument\n")
    ct2expr = tmp
    fl.close()
    #if umap == True:
    #    plotUMAP(ct2expr,indx2ct)
    sns.set(style="whitegrid")
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    if Findungated is True and sum(UNKcount.values()) > 1: ## plotting the proportion of ungated samples in each sample_id from which training dataset arose
        X = [k for k in TotalCount.keys()] ## all sample_ids used in training dataset
        T = [TotalCount[k] for k in TotalCount.keys()] ## total number of cells in all sample_ids used in training dataset 
        Y = [UNKcount[k] for k in TotalCount.keys()] ## number of ungated cells observed each of the sample_id used in training dataset 
        P = [(Y[i]/T[i])*100 for i in range(len(Y))] ## proportion of ungated cells observed 
        df = pa.DataFrame({"X":X,"Y":P})
        ax = sns.barplot(x="X", y="Y", data=df)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
        ax.set_title('Percentage of ungated cells in each sample ID ')
        ax.set_ylabel('Percentage of ungated cells')
        ax.set_xlabel('Sample ID')
    return Sample(X=ct2expr,y=indx2ct,z=PersampleAllGated,l=PIDs)

def SelectMD(md,ProjectName):
    ### This function is written to specifically deal with POISED data ###
    new_md = []
    batches = np.unique(md.loc[:, 'Batch'])
    for b in batches:
        subset1 = md.query("Batch == @b")
        TimePoints = np.unique(subset1.loc[:, 'Time'])
        for t in TimePoints:
            subset2 = subset1.query("Time == @t")
            Stims = np.unique(subset2.loc[:, 'Stim'])
            for s in Stims:
                subset3 = subset2.query("Stim == @s")
                PIDs =  np.unique(subset3.loc[:, 'PID'])
                selectedPID = np.random.choice(PIDs,1).tolist() ## this is the randonly selected patient that will be used for this batch b for time t with stim s ## suppose there are multiple samples in a PeaStim in agiven batch for time t; then it picks all cell types of only one sample randomly
                subset4  = subset3.query("PID == @selectedPID")
                new_md.append(subset4)
    new_md = pa.concat(new_md)
    new_md.index = list(range(new_md.shape[0]))
    name = str(ProjectName) + "/NewMD.csv"
    new_md.to_csv(name)
    return new_md

def method1(Fileinfo,normalizeCell,relevantMarkers,header,ic,ungated,train,ProjectName,cofactor):
    ''' This can be used to load panda dataframe 50 times faster'''
    '''https://blog.esciencecenter.nl/irregular-data-in-pandas-using-c-88ce311cb9ef'''
    test = {}
    #header="infer"
    md = pa.read_csv(Fileinfo,header=header)
    md.index = list(range(md.shape[0]))
    inputs = md.iloc[:,0]
    sample_ids = md.iloc[:,1]
    if ungated is True and len(train.l) > 0: ##Check if PIDs are non-zeros, i.e. there are sample IDs that are required to be scanned for ungated cellsn
        fnd = list(set(list(train.l)) & set(list(sample_id))) ## check if 
        if len(fnd) != len(train.l): ## check if all of these samples exists as live cells 
            notfound = list(set(train.l) - set(fnd)) ## some or all samplesIDs do not have live ungated datasets available. find their sampleIDs 
            print("Error!!! The live cells belonging to these sampleIDs: " + str(notfound) + " not found. These samples are required for finding their ungated cell population. Either set ungated=False or read manual")
            exit
    print ("Reading live cells for annotation...")
    counter = 0
    for c,f in enumerate(inputs):
        if os.path.exists(f):
            match = re.search(r'fcs$', f)
            match2 = re.search(r'csv$', f)
            key=md.loc[c,"sample_id"] #+ "_" + str(counter) ## sampleID
            if match: ## if file is FCS
                panel, exp = fcsparser.parse(f, reformat_meta=True)
                desc = panel['_channels_']['$PnS'] ## marker name instead of meta name
                desc = desc.str.replace('.*_', '', regex=True)
                desc = desc.str.replace('-', '.', regex=True).tolist()
                exp.columns = desc
                #_, exp = fcsparser.parse(f, reformat_meta=True)
                print("FCS file: " + str(c) + "...loaded")
            elif match2: ## if file is CSV
                exp = pa.read_csv(f, delimiter=",", header=header, index_col=ic)
                print(f)
            if 'labels' in exp.columns: ## when the test CSV/FCS file already have labels for every cell ; the column name must be labels 
                CT=exp.loc[:,'labels'] ## this is those cases when labels are already available for test dataset; good for benchmarking
                exp.drop(['labels'], axis=1, inplace=True) ## remove this from expression matrix as we will be filling it with predicted values 
                test[key] = Sample(X=exp,y=CT) ## key the sample.y as labels 
            else:
                test[key] = Sample(X=exp) ## keep is plain no labels are available 
            counter += 1
        else:
            print(str("*** Error ***: Line: " + str(c) + " or " + str(f) + "- Unknown File !!!"))
            exit
    if ungated is True and len(train.l) > 0: ## there are sample_IDs for which ungated cells have to be calculated 
        train = labelUngated(test,train,relevantMarkers) ## train.z is expression values of gated cells in train.l sample_ids from which ungated cells needs to be calculated 
    if normalizeCell is True: ## normalize only after ungated cells are predicted; if required 
        test = preprocess(test,relevantMarkers,cofactor)
        train.X = preprocess(train.X,relevantMarkers,cofactor)
    print ("Saving Train and test objects")
    os.mkdir(ProjectName + "/others_")
    testsave = ProjectName + "/others_/test.PyData"
    joblib.dump(test, testsave)
    path = ProjectName + "/others_/train.PyData"
    joblib.dump(train, path)
    #return Sample(X=test, y=train)

def preprocess(test,relevantMarkers,cofactor):
    for key, value in test.items():
        print("Normalizing..." + str(key))
        value.X[value.X < 0] = 0 ## if expression is negative then make it zero as cell cannot be negatively expressed also it will create problem in normalizarion
        if len(value.X.columns) == len(relevantMarkers):
            exp = np.arcsinh((value.X - 1.0)/cofactor)
            test[key].X = exp
        elif len(value.X.columns) > len(relevantMarkers):
            expr1 = value.X.drop(columns=relevantMarkers, inplace=False) ## separating out other columns, if any 
            expr2 = value.X.loc[:,relevantMarkers] ## normalizing only the expression of relevant markers 
            exp = np.arcsinh((expr2 - 1.0)/cofactor)
            test[key].X = pa.concat([exp,expr1],axis=1) ## again adding the columns excluded from normalization 
        else:
            print("[ERROR] Number of relevant markers are greater than markers in expression matrix of live cells")
            os.exit(0)
    return test

def getPIDnames(md):
    UNKcount = {}
    sortedPID = [] ## list of sampleIDs that are needed to be analyzed for ungated cell identification
    print ("Finding the samples for which ungated cells needs to be determined...")
    for rid in range(md.shape[0]):
        key = md.iloc[rid,2]
        labels = md.iloc[rid,1]
        if key not in UNKcount:
            UNKcount[key] = 0
        if labels in ["Ungated","Unknown","Unclassified","Other","Undefined"]:
            UNKcount[key] = 1
    for pid, cnt in UNKcount.items():
        if cnt == 0:
            sortedPID.append(pid)
    return sortedPID
    
def labelUngated(test,train,relevantMarkers):
    persamplegatedCells = train.z
    PIDs = train.l
    key = 0
    train.y[key] = "Unknown"
    for sample_id in PIDs:
        print("Finding Ungated cells in ... " + sample_id)
        gatedcells = persamplegatedCells[sample_id].loc[:,relevantMarkers]
        Uniques, unique_count = np.unique(gatedcells.values,return_counts = True , axis=0)
        cleangated = pa.DataFrame(Uniques[unique_count==1], columns=relevantMarkers)
        allLiveCells = test[sample_id].X.loc[:,relevantMarkers]
        #cells_DF = pa.concat([allLiveCells,cleangated], axis=0) ## merging gated and all live cells 
        d = len(relevantMarkers) 
        index = faiss.IndexFlatL2(d) ## initiating faiss database
        xb = np.ascontiguousarray(np.float32(allLiveCells.values)) ## making faiss database of epoch-specific marker expression matrix
        index.add(xb) ## indexing the database
        xq = np.ascontiguousarray(np.float32(cleangated.values)) ## getting from the dict ; should be panda data frame of landmark cells with column in same order as orignal expr dataframe ## for each celltype get the landmarks cells (as expression values of all markers from 'eset') ##
        D, I = index.search(xq,1) ##  ## get neighbours of all the landmarks cells for a given cell type ## I variable has rowindexes of cells that should belong to same distribution to the
        indexes = list(np.unique(I.ravel())) ## these are the indexes of gated population in all live cells
        Ungatedcells = allLiveCells.drop(indexes)
        prop = (Ungatedcells.shape[0]/allLiveCells.shape[0]) * 100
        expected = allLiveCells.shape[0] - cleangated.shape[0]
        observed = Ungatedcells.shape[0]
        percentError = abs(((expected - observed)/observed)) * 100
        print("Found " + str(prop) + "% of ungated cells in " + str(sample_id) + " with " + str(percentError) + "% error rate")
        if percentError > 2.0:
            print("*** Too Much error rate in this sample ID. Ungated cells will not be considered for analysis")
        else:
            if key in train.X:
                train.X[0] = Sample(X=pa.concat([train.X[0].X,pa.DataFrame(Ungatedcells)],axis=0))
            else:
                train.X[0] = Sample(X=pa.DataFrame(Ungatedcells))
    train.z = None
    train.l = None
    return(train)

def ReduceDFbyLabel(ct2expr,limit):
    print("Reducing expression matrix")
    data = pa.DataFrame()
    labels = pa.DataFrame()
    for ct, V in ct2expr.items():
        Xi = V.X
        print("for CellType : " + str(ct) + "::" + str(Xi.shape[0]) + " Cells")
        if Xi.shape[0] > limit:
            Xi = Xi.sample(n=limit)
        data = pa.concat([data,Xi])
        ytmp = pa.DataFrame([ct] * Xi.shape[0])
        labels = pa.concat([labels,ytmp])
    return Sample(X=data, y=labels)

def plotUMAP(ct2expr,ct2indx): ## upto 30 different celltype can be plotted, if more cell types are there then add the number of colors accordingly 
    dataSample=ReduceDFbyLabel(ct2expr,3000)
    embedding = umap.UMAP(n_neighbors=15, min_dist=.25).fit_transform(dataSample.X.values)
    target = np.array([ct2indx[x] for x in dataSample.y.values.ravel()])
    df = pa.DataFrame({"UMAP1":embedding[:,0] ,"UMAP2": embedding[:,1], "Cell_Type":list(target)})
    df["Cell_Type"] = df["Cell_Type"].astype('category')
   
    from matplotlib import pyplot as plt
    plt.figure(figsize=(16, 10))
    
    colors =['#6C8387', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
             '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
             '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
             '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
             '#ffffff', '#000000', '#A65656', '#A69556', '#56E042',
             '#BEB8FA', '#4439BA', '#0BE982', '#EAB14B', '#73AFF5' ]
             
    for val, ct in ct2indx.items():
        idxs = np.where(df.iloc[:,2].values == ct)[0]
        sub = df.iloc[idxs,:]
        if ct == "Unknown":
            colorc= "#FF0013"
        else:
            colorc = colors[val]
        plt.scatter(sub.iloc[:,0], sub.iloc[:,1],c = colorc, label=ct, alpha=0.7, s = 10)
    #plt.legend()
    plt.show()
    
def plotSNE(ct2expr):
    dataSample=ReduceDFbyLabel(ct2expr,3000)
    tsne = TSNE(n_components=2, random_state=0)
    vis_data = tsne.fit_transform(dataSample.X)
    # Visualize the data
    classes = list(ct2expr.keys())
    ct2idx = {key:idx for idx, key in enumerate(classes)}
    target = np.array([ct2idx[x] for x in dataSample.y.values.ravel()])

    ### Plotting ####
    plt.title('Hangated tSNE')
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(*vis_data.T, s=0.1, c=target, cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(25)-0.5)
    cbar.set_ticks(np.arange(24))

def findrange(data,confidenceInt=0.80):
    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidenceInt) / 2, n - 1)
    here = m - h
    return abs(here)

def CountFrequency(y):
    freq = {} # Creating an empty dictionary
    for item in y:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return freq

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return(a_set & b_set)
    else:
        return("Error: No common elements")

def index2ct(idx2ct,y_pred_index):
    ct_pred = [0] * len(y_pred_index)
    for idx, ct in enumerate(idx2ct):
        #print(str(idx) + " " + str(ct))
        found =  (y_pred_index == idx).tolist()
        res = [i for i, val in enumerate(found) if val]
        for get in res:
            ct_pred[get] = ct
    return ct_pred

def filterCells(exp): ##TO DO LIST : instaed of removing it should just give index to be removed in each matrix
    #for key,value in eset.items():
    #    exp = value.X
    markers  = exp.columns
    print('removing cell events with very high intensity in any given marker')
    to_remove = []
    for i in range(len(markers)):
        print("reading..."+ markers[i])
        marker_expr = exp.loc[:,markers[i]]
        threshold = np.percentile(marker_expr, 99.99) ## change it to 99.99
        cell_index = marker_expr[marker_expr >= threshold].index
        to_remove.append(cell_index.values.tolist())
    merged = list(itertools.chain(*to_remove)) ## reduce list of lists to singel list
    remove = np.unique(merged) ## these cell indeces have atleast one marker that is outlier and these cells will be discarded from analysis
    per_cells_removed = (len(remove) / exp.shape[0]) * 100
    print("Level 1: Cell can be removed " , per_cells_removed, "%(" ,len(remove), ")")
    min_max_scaler = preprocessing.MinMaxScaler() ## scaling column wise expression between 0 to 1
    x_scaled = min_max_scaler.fit_transform(exp)
    x=pa.DataFrame(x_scaled, columns=exp.columns) ## converting numpy array to panda df
    info = x.sum(axis=1) ## sum of each row or cell(info score)
    threshold = 0.05 ## INFOi threshold ## cells with score less than this value have no use
    to_remove = [] ## this list will hold indexes of all those cells that are needed to be removed
    tmp = info[info <= threshold].index.tolist() ## these cells have INFO score very less and these cells contribute nothing so removed
    to_remove.append(tmp)
    df = x.max(axis=1)
    tmp = df[(info * 0.5)< df].index.tolist()
    to_remove.append(tmp)
    merged = list(itertools.chain(*to_remove,remove.tolist())) ## merging the index from level 1 of filtering and level2
    remove = np.unique(merged)
    #exp.drop(remove.tolist(), axis = 0, inplace=True) ## removing the cells from main matrix ## role of scaled matrix is over
    print("Level 2: Cell can be removed " , per_cells_removed, "%(" ,len(remove), ")")
    #eset[key].X = exp
    #return eset ## this is a dict with key as filname and value has sample class. The sample class contains refrence/pointers to the corresponding expression matrix (.X) ; marker columns (.y) and sample info (.z)
    return remove.tolist() ## list of indexes that can be removed

def countEpoch(blockInfo,allowed):
    counts = []
    for key,value in blockInfo.items():
        blocksCount = len(blockInfo[key]) ## blockInfo is hash of hash so its value is also an hash whose length is the
        counts.append(blocksCount)
    epoch = max(counts) * allowed ## each block in the sample with maximum number of blocks must cover 'allowed' iterations to generate reliable probability
    return epoch

#def select_block(blockinfo):
#    vals = blockinfo.values()
#    minVal = min(vals)
#    minVals = []
#    for k in blockinfo.keys():
#        if blockinfo[k] == minVal:
#            minVals.append((k, minVal))
#        #elif blockinfo[k] == maxVal:
#        #    maxVals.append((k, maxVal))
#    ind = np.random.choice(range(len(minVals)),1)
#    return(minVals[ind[0]][0]) ## return the randonly selected block ID with lowest freq

def getModel(models,counter,ongoing, blocknumber,lda):
    if counter == blocknumber:
        mod = models[ongoing].X
        ct = models[ongoing].y
        counter = 0
        ongoing = ongoing + 1
    else:
        mod = lda
        ct = models[ongoing].y
    return Sample(X=mod,y=counter,z=ongoing,l=ct)

def makePrediction(E,M,lda,CellTypes,postProb):
    X = E.X ## this is the expression matrix
    X = X.loc[:,M] ## M = relevantMarkers
    pred = lda.predict(X)
    pred_prob = lda.predict_proba(X) ## Posterior probability lower threshold, below which the prediction will be 'unknown',
    max_prob = np.amax(pred_prob, axis=1) ## maximum probability associated with each cell (row) for a given cell type among all celltypes
    #indexes = np.argmax(pred_prob, axis=1) ## indexes of the column to which maximum prob is associated; each column ~ a celltype; the cell type with maximum prob is the one we assigned to each cell
    unk_index = np.nonzero(max_prob < postProb) ## getting index of cells when maximum post. prob of cell to be associated with a given celltype is less than threshold
    pred[unk_index] = "Unknown" ## replacing the assigningment of cells as "unknwon" if their post prob. is less than threshold
    unassignedCellsProp  = (np.shape(unk_index)[0]/np.shape(pred)[0])*100 ##%age of cells that are likely to be of unknown category
    ## TO DO: compute the cluster size; average distance; average silhouette ; euclidean distance
    return Sample(X=pred,y=max_prob,z=unassignedCellsProp)

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

#def findcelltype(blck2ct,rowsInfo,epoch,f1):
#    indx2ct ={}
#    for b in blck2ct.keys(): ## 'b' is the bloc number
#        df = blck2ct[b] ## the cell types predicted for each cell in a given block (remember this is running for each file)
#        ##df.groupby('a').count() ## counting the number of
#        x = epoch - df.apply(pa.value_counts, axis=1).fillna(0) ## for each cell count the number of times this cell type occuered in all possibel cell types i.e. in each row calculate the freq of all known cell types
#        x = x/epoch ## This x has the score of each cell in all possible CTs ; score = (epoch - number of times CT found across all epoch) / epoch ; lower the better
#        ct = x.idxmin(axis=1) ## extract the name of cell type with min score
#        ct_pval = x.min(axis=1) ## extarct the min score
#        varianc = df.var(axis=1)
#        df = pa.DataFrame({'Celltype':ct,'Minimum_P-value':ct_pval,'Cell type Variance':varianc})
#        cell_index = rowsInfo[f1][b]
#        indx2ct = dict(zip(cell_index, df)) ## a dict where key is the cell index and value is the 3-column df data frame
#    return(Sample(indx2ct))
#
#def FindCelltypePerCell(dicti,rd,counter,Cellimit,rowsInfo,Runcollection,result,result2):
#    start = 0
#    CTs = dicti.X ## get the cell type labels for the dataset in this run
#    PostProb = dicti.y ## posterior prob. associated with each cell for the predicted cell type.
#    tmp=rd.y ## which block is used in which file during this epoch number
#    acc_pred = []
#    for f1 in Runcollection.l: ## fileorder: reading the filenames in the same order as per they exists in the expression matrix ; this has file names (see makerun function)
#        print(f1)
#        df1 = pa.DataFrame()
#        if f1 in result.keys(): ## if this file f1 has already been processd in someother epoch number
#            blck2ct = result[f1] ## read the block number of each file: key2 and gives the cell types of all cells if block already run under any epoch(s) before
#            blck2pp = result2[f1] ## read the block number of each file: key2 and gives the posterior prob. if block already run under any epoch(s) before
#        else:
#            blck2ct = {}
#            blck2pp = {}
#        block = tmp[f1] ## for this file whats the block used
#        itsIndecies = rowsInfo[f1][block] ## rowinfo will tell what are the indexes available in this block
#        end = start + len(itsIndecies)
#        region = range(start,end)
#        en = ('e'+str(counter)) ## epochnumber
#        ct = pa.DataFrame({en:CTs[region]}) ## a dataframe in which column name is the epoch number (e1, e2..en) and value is the cell type predicted
#        pp = pa.DataFrame({en:PostProb[region]})
#        cellsFound = ct.shape[0]
#        ct[en] = ct[en].astype('category') ## converting the cell type to categorical data
#        print(str(CTs.shape[0]) + " / " + str(start) + " / " + str(end)  + " / " + str(cellsFound))
#        if block in blck2ct.keys(): ## if blck2ct[block] exists then update the vlues else
#            df1 = blck2ct[block]
#            blck2ct[block] = pa.concat([df1, ct], axis=1, sort=False) ## keep on appending the df cell types (column wise) according to the order of cells used in expression matrix
#            ref = blck2ct[block].iloc[:,0]
#            acc_pred.append(accuracy_score(ref, ct)) ## calculate accuracy at this point and keeping first column as ref and see the fluction of accuracy after all runs
#            df1 = blck2pp[block]
#            blck2pp[block] = pa.concat([df1, pp], axis=1, sort=False)
#        else:
#            blck2ct[block] = ct ## value (cell type) is the panda datafarame
#            blck2pp[block] = pp
#        result[f1]=blck2ct ##result{filename}{blocknumber}:celltye labels
#        result2[f1]=blck2pp ##result2{filename}{blocknumber}:celltye labels post. prob.
#        start = end
#        if cellsFound != Cellimit:
#            print ("Error 1: Row count doesnt match. Dont know what to do :( \n" +
#            "epoch: " + str(counter) + "\n" +
#            "expected rows: "+ str(Cellimit) + "\n" +
#            "Found rows:" + str(np.shape(ct)[0]) + "\n")
#            #exit()
#    acc_overall = sum(acc_pred)/len(acc_pred) ## average accuracy per  block in each run
#    return Sample(X=result,y=result2,z=acc_overall)
#
#def makeBlocks(cells,limit,expr):
#    blockscore = {}
#    block = {}
#    items = range(cells)
#    if items == limit: ## for this file only one block can be made
#        blockNumber =  ('block' + str(1))
#        block[blockNumber] = [items]
#        blockscore[blockNumber] = 0
#    else : ## basicially it will keep on randomly collecting index from the file and start assigning them to block; if indexes are over and a block has remianed empty by > 1 cell, them remaining cell was assigned from any other block randomly
#        count = int(cells/limit) + 1## total number of blocks to be made for ongoing file
#        #items_copy = items.copy() ## making a copy for later use
#        for i in range(count):  ## +1 beacuse we used int
#            blockNumber =  ('block' + str(i))
#            if len(items) < limit: ## if you have less number of cells available in dataset for this block than required; anyways we will fill this block to desired number of cells by randomly selecting the cells
#                blck_indx = np.random.choice(items,len(items), replace=False)
#                req = limit - len(blck_indx) ##if in the final block the number of cells are less than limits than add required number of cells ##
#                subse = list(set(list(range(cells)))^set(list(blck_indx))) ## small subset of indexes from which remaining cells need to be identified randomly
#                blck_indx = np.append(blck_indx, np.random.choice(subse,req, replace=False)) ## randomly capturing required number of cells from subset of cells indexes
#            else :
#                blck_indx = np.random.choice(items,limit, replace=False)
#            exprlist = expr.X.iloc[blck_indx,:].values.tolist() ## this is the expression matrix made up of the indexes of the given block 'blockNumber' for the ongoing file
#            block[blockNumber] = Sample(X=exprlist,y=blck_indx) ## updating block dict for the expression matrix and indexes that becomes the part of this block for the ongoing file
#            blockscore[blockNumber] = 0 ## initiating frequency: number of times thois block has been used during iteration; this dict is imp and will keep on updating itself during the process
#            items = list(set(items)^set(blck_indx)) ## now items hold only those indexes not used before
#    blocks = Sample(block, blockscore)
#    return blocks
#
#def getBlockNum(blockInfo):
#    blcounts=[]
#    for samples in blockInfo:
#        blcounts.append(len(blockInfo[samples]))
#    blcounts.sort()
#    return(blcounts[0])
#
#def performLDA(ExprSet,relevantMarkers,shrinkage,nlandMarks,LandmarkPlots,ct2expr,indx2ct):
#    #counter = 1
#    #skf = StratifiedKFold(n_splits=2, shuffle = True, random_state=random.randint(1,100000))
#    print("Q/LDA: Splitting into train (80%) and test set (20%)")
#    skf = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random.randint(1,100000))
#    model={}
#    # X is the feature set and y is the target
#    for train_index, test_index in skf.split(ExprSet.X,ExprSet.y):
#        X_train = ExprSet.X.iloc[train_index,:]
#        X_test = ExprSet.X.iloc[test_index,:]
#        y_train = ExprSet.y.iloc[train_index,:].values.ravel()
#        y_test = ExprSet.y.iloc[test_index,:].values.ravel()
#        print("Fitting QDA")
#        qda = QDA(store_covariance=True)
#        qda.fit(X_train, y_train)
#        classes = qda.classes_ ## this is the list of cell types
#        #y_train_pred = qda.predict(X_train) ## train
#        y_test_pred = qda.predict(X_test)
#        #accu1 = accuracy_score(y_train_pred, y_train) ## train
#        accu1 = accuracy_score(y_test_pred, y_test)
#        accu11 = f1_score(y_test, y_test_pred, average='weighted')
#        prob = qda.predict_proba(X_test)
#        y_test_Postprob = pa.DataFrame(prob, columns=classes)
#        #CellTypeAcc(y_train_pred,y_train,"QDA", None)
#        Dplot = CellTypeAcc(y_test_pred,y_test,"QDA",y_test_Postprob,indx2ct)
#        #model[accu1] = Sample(X=qda, Y=)
#        model[accu11] = Sample(X=qda,y=Dplot)
#        #qda.predict_proba()
#        print('QDA Model: ' + ' // Accuracy : {:.2f}'.format(accu1) + ' // Accuracy : {:.2f}'.format(accu11))
#        print("Fitting LDA")
#        if shrinkage== None:
#            lda = LDA(n_components = len(relevantMarkers) -1)
#        else:
#            lda = LDA(solver='lsqr', shrinkage='auto')
#        lda.fit(X_train, y_train)
#        #y_train_pred = lda.predict(X_train)
#        y_test_pred = lda.predict(X_test)
#        y_test_Postprob = pa.DataFrame(lda.predict_proba(X_test), columns=classes)
#        #accu2 = accuracy_score(y_train_pred, y_train)
#        accu2 = accuracy_score(y_test_pred, y_test)
#        accu22 = f1_score(y_test, y_test_pred, average='weighted')
#        #CellTypeAcc(y_train_pred,y_train,"LDA",None)
#        Dplot = CellTypeAcc(y_test_pred,y_test,"LDA",y_test_Postprob,indx2ct)
#        print('LDA Model: ' + ' // Accuracy : {:.2f}'.format(accu2) + ' // F1-score : {:.2f}'.format(accu22))
#        #model[accu2] = Sample(X=lda) ## enable this if you want train prediction to be also used for final model selection
#        model[accu22] = Sample(X=lda, y=Dplot) ##<========
#        #counter = counter + 1
#        clf = GradientBoostingClassifier(n_estimators=130, learning_rate=0.1, max_depth=2, random_state=0).fit(X_train, y_train)
#        y_test_pred = clf.predict(X_test)
#        y_test_Postprob = pa.DataFrame(clf.predict_proba(X_test), columns=classes)
#        #accu2 = accuracy_score(y_train_pred, y_train)
#        accu2 = accuracy_score(y_test_pred, y_test)
#        accu22 = f1_score(y_test, y_test_pred, average='weighted')
#        #CellTypeAcc(y_train_pred,y_train,"LDA",None)
#        Dplot = CellTypeAcc(y_test_pred,y_test,"LDA",y_test_Postprob,indx2ct)
#        print('GB Model: ' + ' // Accuracy : {:.2f}'.format(accu2) + ' // F1-score : {:.2f}'.format(accu22))
#        lmarks = findLandmarks(ct2expr,X_test,y_test,classes,nlandMarks,LandmarkPlots,relevantMarkers,indx2ct)
#    scores = list(model.keys()) ## getting all the accuracu during LDA/QDA modelling
#    scores.sort(reverse = True) ## soring the accuracy in decending order
#    mod= model[scores[0]].X ## model with highest accuracy either LDA or QDA
#    thresholds = model[scores[0]].y ## celltype specific threshold (from TR prediction of test set) from the selected model mod
#    X_train=None; X_test=None; y_test_pred=None; y_test_Postprob=None;ExprSet=None; prob=None;scores=None;
#    return Sample(X=mod,y=thresholds,z=lmarks,l=lmarks)

def fitGMM(ct2expr,Maxlimit,totalCells,indx2ct): ## this will create a separate 2-component GMM model for each cell type; this will help us in shortlisting bad neighbours preicted by faiss
    #Maxlimit = 100000 ## Maximum number of cells to make GMM fitting per cell type
    models = {}
    freqCT = {}
    print ("Fitting GMM and finding Landmarks (may be time consuming for some cell types) ")
    start_time = time.time()
    for cellType, V in ct2expr.items():
        Xi=V.X
        ct = indx2ct[cellType]
        print("for CellType : " + str(ct) + "::" + str(Xi.shape[0]) + " Cells")
        if Xi.shape[0] > Maxlimit:
            Xi = Xi.sample(n=Maxlimit)
        gmm = GMM(n_components=2).fit(Xi)
        models[cellType] = gmm
        freqCT[cellType] = V.X.shape[0]/totalCells
    print("--- %s seconds used ---" % (time.time() - start_time))
    return Sample(X=models,y = freqCT)

#def makerun2(rowsInfo,blockInfo,relevantMarkers):
#    runInfo = {}
#    #expr = np.empty((0, dim))
#    expr=[]
#    orig = []
#    sampl = []
#    #annotation = pa.DataFrame() ## tmp varibale ; will be removed
#    for f1 in rowsInfo.keys(): ## read each file one by one
#        #print("fILENAME..",f1)
#        blockinfo = blockInfo[f1] ## get the frequency info of each block of cells. this stores the information of how many times a given block has been used for analysis
#        #selected_block1='block0'
#        selected_block1 = select_block(blockinfo) ## this will give the block ID with minimum frequency (randomly chosen block)
#        blockinfo[selected_block1] = blockinfo[selected_block1] + 1 ## update the frequency of block once its used so that other block woulg get preference
#        blockInfo[f1] = blockinfo # update main dict
#        indeces = rowsInfo[f1][selected_block1].y ## get the index of these cells in a given file f1 ## there can only be one block froma file so no worry about the order until filename order is intact
#        runInfo[f1] = selected_block1 ## update which block ID was used for which file in this run
#        expr += rowsInfo[f1][selected_block1].X ## a list of expression values from the selected block of this file
#        orig += list(indeces) ## its orignal indexes
#        sampl += ([f1] * len(indeces)) ## a list of samplenames with respect to the indexes
#    IndexDict = dict(zip(range(len(sampl)),orig))
#    sampleDict = dict(zip(range(len(sampl)),sampl))
#    ## the speed can further be increased if somehow I can resolve X=pa.DataFrame(expr,columns=relevantMarkers) with some alternative
#    runs = Sample(X=expr,y=runInfo,z=blockInfo,l=IndexDict,o=sampleDict) ## sample class object;
#    return runs

def farthest_search(df, top):
    out = []
    print(".", end = '')
    #print(df)
    for _ in range(top-1):
        dist_mat = spatial.distance_matrix(df.values, df.values)         # get distances between each pair of candidate points
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)         # get indices of candidates that are furthest apart
        z = [i] + [j] ## these two indexes are fathest from each other
        out += list(df.index[z]) ## getiing the orignal rowname/index of those indexes found to be farthest from teach other in this run
        df.drop(df.index[z], inplace=True)
    return out

def checkLM(TP1,TP2):
    ##there are 2 criteria to call TP1 good than TP2
    ret = TP2 ## by deafult do not remove/change TP2 from dict
    if TP2 == None:
        ret= None ## already None; so let it be None; TP1 is good; TP2 will be automatically ignored from dict
    else:
        common  = set(TP1) & set(TP2) ## common True Postives predicted by these two LMs
        if len(common) / len(TP2) >= 0.98:  ## more than 95% of common TP exits within TP1; why would i NEED TP2 THEN; its a repetitive landmark. i dont need it
            ret = None ## REMOVING ITS PREDICTIONS ; so that this index (TP2 index) will be ignored for subsequent analysis; whereas TP1 will be used
    return ret

def plot_XGboostLogLoss(model):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.show()

def run_CATboost(X_train, y_train,X_test, y_test, ct,threads):
    eval_set = [(X_train, y_train), (X_test, y_test)]
    catboostmod = CatBoostClassifier(depth = 10, learning_rate=0.1 , l2_leaf_reg=3, random_seed = 164530, thread_count=threads, od_wait = 20, od_type='IncToDec', eval_metric='F1', use_best_model=True).fit(X_train, y_train,eval_set=eval_set)
    y_test_pred = catboostmod.predict(X_test)
    accu1 = precision_score(y_test, y_test_pred, average='micro')
    accu11 = f1_score(y_test, y_test_pred, average='weighted')
    print('CATBoosting for cell type: ' + str(ct) + ' // Precision : {:.2f}'.format(accu1) + ' // F1-score : {:.2f}'.format(accu11))
    return Sample(X=catboostmod,y=accu11)

def runEnsemble(X_train,y_train,X_test,y_test,my_test,ct,obj0,tree,plotlogLoss,SVM,threads):
    eval_set = [(X_train, y_train), (X_test, y_test)]
    from mlens.ensemble import SuperLearner
    ensemble = SuperLearner(scorer=f1, random_state=8888, n_jobs=10, verbose=3)
    ensemble.add([XGBClassifier(learning_rate=0.3, subsample=0.80, n_estimators=2000, max_depth=6,
                                objective='binary:logistic', gamma=2, colsample_bytree=0.85, reg_alpha=0.005,
                                min_child_weight=3, nthread=threads, verbose=3),
                  SVC(C=0.001, verbose=3),
                  QDA(store_covariance=True)])
    ensemble.add([SVC(C=0.1, verbose=3)])
    ensemble.add([SVC(verbose=3)])
    ensemble.add_meta(LogisticRegression())
    ensemble.fit(X_train, y_train, eval_metric=["error", "logloss"], early_stopping_rounds=10, eval_set=eval_set,
                 verbose=3)
    y_test_pred = ensemble.predict(X_test)
    ensemblePrecision = precision_score(y_test, y_test_pred, average='micro')
    ensembleF1 = f1_score(y_test, y_test_pred, average='weighted')
    print("**********************************")
    print('Ensemble Model Result cell type: ' + str(ct) + ' // Precision : {:.2f}'.format(
        ensemblePrecision) + ' // F1-score : {:.2f}'.format(ensembleF1))
    print("**********************************")
    return ensemble

def XGboosTree(xgbmdl):
    plot_tree(xgbmdl, num_trees=2, rankdir='LR')
    plt.show()
    plot_importance(xgbmdl)
    plt.show()

def CellTypeAcc(y_test_pred,y_test,text,y_test_Postprob,indx2ct):
    ## read all celltypes one by one ## get it from unique
    y_pred = np.array(y_test_pred) ## predicted labels
    y_test = np.array(y_test) ## orignal labels
    mlist = y_test_Postprob.columns.values
    Dplot={}
    mod = {}
    temp = {}
    meanAcc = 0
    for ct in list(mlist):
        if ct != 0:
            probs = pa.DataFrame(y_test_Postprob.loc[:,ct])
            ind1 = np.where(y_pred == ct)[0] ## predicted lables
            ind2 = np.where(y_test == ct)[0] ## orignal labels for this ct
            ind11 = np.where(y_pred != ct)[0] ## predicted lables not having this CT
            ind22 = np.where(y_test != ct)[0]  ## orignal lables not having this CT
            common = list(set(ind1).intersection(ind2)) ## true positive prediction; the indexes that are predicted to be this cell type (in y_pred) are also the same cell type in y_test
            common2 = list(set(ind11).intersection(ind22))  ## true negative prediction; the indexes that are predicted to be this cell type (in y_pred) are also the same cell type in y_test
            probs = probs.iloc[common,:].values## prosterior probs observed in TP pedictions by LDA or QDA
            errorRate = (len(ind1) - len(common)) / len(ind1)
            Dplot[ct] = {'thesh':np.quantile(probs, 0.01),
                 'minimum': np.amin(probs),
                 'maximum': np.amax(probs),
                 'Mean':np.mean(probs),
                 'errorRate': errorRate}
            accu = len(common) / len(ind2) ## found among all the expected lables
            TP = len(common)
            FP = len(ind1) - len(common)
            FN = len(ind2) - len(common)
            TN = len(common2)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1 = 2 * ((precision * recall) / (precision + recall ))
            #F1 = (2*TP) / ((2*TP) + FP + FN)
            accuracy = (TP + TN)/(TP + TN + FP + FN)
            cELLt = indx2ct[ct]
            mod[ct] = accu
            print ('TP for celltype : ' + str(cELLt) + ' with ' + str(len(ind2)) + ' cells is : {:.2f}'.format(accu) + '; F1 score: ' + str(F1) + '; accuracy: ' + str(accuracy) + '; method: ' + text)
            print ('###############################################################################')
            meanAcc += accuracy
            temp[ct] = F1
    threshold = pa.DataFrame(Dplot).T
    threshold.plot.line()
    print(threshold)
    meanAcc = meanAcc/len(list(mlist))
    print("Mean accuracy" + str(meanAcc))
    return Sample(X=mod,y=temp)

def AllvsAllQDA(X_train, y_train,X_test,y_test,indx2ct,relevantMarkers,UnkKey):
    #qda = QDA(store_covariance=True)
    #qda.fit(X_train, y_train)
    #classes = qda.classes_ ## this is the list of cell types
    #y_test_pred = qda.predict(X_test)
    X_train = X_train[y_train != UnkKey] ## UnkKey is code for unknown cells
    y_train = y_train[y_train != UnkKey] ## here we are removing unkown cells from training. though we will keep them for testing (as randomly scattered among gated cell types)
    #X_test = X_test[y_test != UnkKey] ## UnkKey is code for unknown cells
    #y_test = y_test[y_test != UnkKey] ## here we are removing unkown cells from training. though we will keep them for testing (as randomly scattered among gated cell types)
    lda = LDA(n_components = len(relevantMarkers))
    lda.fit(X_train, y_train)
    y_test_pred = lda.predict(X_test)
    classes = lda.classes_
    accu1 = accuracy_score(y_test_pred, y_test)
    accu11 = f1_score(y_test, y_test_pred, average='weighted')
    #prob = qda.predict_proba(X_test)
    prob = lda.predict_proba(X_test)
    y_test_Postprob = pa.DataFrame(prob, columns=classes)
    obj = CellTypeAcc(y_test_pred,y_test,"LDA",y_test_Postprob,indx2ct)
    mod = obj.X
    temp = obj.y
    print('LDA Model: ' + ' // Accuracy : {:.2f}'.format(accu1) + ' // F1-score : {:.2f}'.format(accu11))
    ### lets plot F1-score (Y-axis) vs log(population size)
    popsize = dict(Counter(y_test))
    logpopsize = {key:math.log(values) for key,values in popsize.items()}
    df1 = pa.DataFrame([logpopsize], index=["Log population Size"]).T
    df2 = pa.DataFrame([temp], index= ["F1-score"]).T
    df = pa.concat([df1,df2], axis=1)
    ax2 = df.plot.scatter(x='Log population Size',y='F1-score',c='F1-score', colormap='viridis')
    return Sample(X=lda,y=mod)

def plot_LM_cells_in_PCA(Xi,rows,col,name):
    lmcells = pa.DataFrame(rows, columns=col)
    merged = pa.concat([pa.DataFrame(Xi, columns=lmcells.columns), lmcells], axis=0)
    labs = ([0] * Xi.shape[0]) + ([1] * lmcells.shape[0])
    PCs = runPCA(merged)
    df = pa.concat([pa.DataFrame(PCs, columns=["PC1", "PC2"]), pa.DataFrame(labs, columns=["label"])], axis=1)
    ax = df.plot.scatter(x="PC1", y="PC2", c="label", colormap='viridis')
    plt = ax.get_figure()
    plt.savefig(name)

def printReport(TP,orignal,sm,ct,allpred):
    TPrate = len(TP)/len(allpred)
    FPrate = abs(len(allpred)-len(TP))/ len(allpred)
    print ("in sample " + str(sm) + "; for cell type " + str(ct) + " TP: " + str(TPrate) + ' and FP: ' + str(FPrate) + ' with ' + str(len(allpred)) + " / " + str(len(orignal)) + ' cells ')

#def printACC2(sm,lables,predictedLabels,CTs,y_test_pred):
#    accu1 = accuracy_score(y_test_pred,lables)
#    accu11 = f1_score(lables, y_test_pred, average='weighted')
#    print('QDA Model: ' + ' // Accuracy : {:.2f}'.format(accu1) + ' // Accuracy : {:.2f}'.format(accu11))
#    for ct in CTs:
#        ind1  = list(np.where(lables == ct)[0])
#        ind2  = list(np.where(predictedLabels == ct)[0])
#        ind3 = list(np.where(np.array(y_test_pred) == ct)[0])
#        common = list(set(ind1) & set(ind2))
#        acc = (len(common)/len(ind1)) * 100
#        acc2 = (len(common)/len(ind2)) * 100
#        print ("in sample " + str(sm) + "; accuracy of cell type " + str(ct) + " is " + str(acc) + 'and' + str(acc2) + ' with ' + str(len(ind1)) + ' cells ')
#        common2 = list(set(ind1) & set(ind3))
#        acc = (len(common2)/len(ind1)) * 100
#        acc2 = (len(common2)/len(ind3)) * 100
#        print ("in sample " + str(sm) + "; QDA accuracy of cell type " + str(ct) + " is TP: " + str(acc) + ' and FP: 100 - ' + str(acc2) + ' with ' + str(len(ind1)) + ' cells ')

def getCenter2(data,name,nlandMarks,plot):
    #data = pa.DataFrame(data)
    # Computing the alpha shape
    edges = alpha_shape(data, alpha=0.80, only_outer=True) ## those cells that lie outside of the PC cluster
    edgeslist = [[i]+[j] for i, j in edges]
    desiredIndex  = list(chain(*edgeslist))
    warnings.filterwarnings("ignore")
    myPDF,axes,z = fast_kde(data[:,0],data[:,1],sample=True) ## the modified KDE downloaded from https://gist.github.com/joferkington/d95101a61a02e0ba63e5
    z = pa.DataFrame(stats.zscore(z))
    quants = [i for i in np.arange(0.98, 0.2, -0.12)] ###  [0.99,0.89,0.79,0.69,0.59,0.49,0.39,0.29,0.19] ## qualtile ranges
    threshs = [np.quantile(z, q) for q in quants] ## maximum value of Kernel density within each qantile range
    ## Idea was, we split the kernel densitites into 4 quantiles; each quantile will contribute to 'defined' number of cells
    ## each quantile has its own density threshold; withoin which randomly defined number of cells will be picked
    dens = z[:]
    for th in threshs:
        LandMarkcell = list(z[z[0] >= round(th, 3)].index)
        z.drop(LandMarkcell, inplace=True)
        if len(LandMarkcell) <= nlandMarks:
            nlandMarks = len(LandMarkcell)
        LandMarkcell = choices(LandMarkcell, k=nlandMarks)
        desiredIndex += LandMarkcell
    density = dens.iloc[desiredIndex, :]
    if plot == True:
        test0 = pa.DataFrame({"x":data[:,0],"y":data[:,1],"z":dens.loc[:,0]})
        ax=test0.plot.scatter(x='x',y='y',c='z',colormap='viridis') ## if you are running the above three lines; hash this line
        plt = ax.get_figure()
        plt.savefig(name)
    return Sample(X=desiredIndex,y=density) ## ******

def processArray(tmp,Sam2CT,Idx2sample,Idx2rowNum):
    for id1,data in tmp.items():
        sm = Idx2sample[id1]
        oid  = Idx2rowNum[id1]
        if oid in Sam2CT[sm]:
            Sam2CT[sm][oid] = np.append(Sam2CT[sm][oid], data, axis=0)
        else:
            Sam2CT[sm][oid] = data
    return Sam2CT

#def report(Sam2CT,indx2ct,test):
#    for sm in Sam2CT.keys(): ## reading samples one by one from template (Sam2CT)
#        print("Putting labels sample..." + sm)
#        out= []
#        for indx in range(test[sm].X.shape[0]):
#            if indx in Sam2CT[sm]: ## found in some cluster
#                data =Sam2CT[sm][indx] ## [1] score [2] CT/per epoch [3] all CTs predicted for this index in every epoch
#                CT = list(data[:,0].ravel()) ## list of CT predicted (in order) for this index
#                PP = list(data[:,1].ravel()) ## corresponding Post prob. for every prediction in CT
#                tmp={}
#                for ct1 in np.unique(CT):
#                    ind = np.where(ct1==CT)[0]
#                    out = [PP[f] for f in ind]
#                    score = sum(out) * len(out)
#                    tmp[score] = ct1
#                out += [indx2ct[tmp[max(tmp.keys())]]] ## final CT for this index
#            else:
#                out += ["Unknown"]
#        df0 = pa.DataFrame(out, columns=["PredictedLabels"])
#        df1 = pa.concat([test[sm].X,df0],axis=1 )
#        name = str(sm) + "labelledExpr.csv"
#        print("writing ..." + name )
#        df1.to_csv(name)
#        printACC(test[sm].X.iloc[:,"labels"], out)

def findLandmarks(X_train,y_train,X_test,y_test,CTs,nlandMarks,LandmarkPlots,relevantMarkers,indx2ct,UnkKey,ProjectName):
    ## Now also identify land marks for this cell type
    nlandMarks=50 #### from each quantile out of 10 quantiles of kernel densitites (PC1 vs PC2) pick this much number of cells
    d = len(relevantMarkers) # number of markers
    np.random.seed(12)
    index = faiss.IndexFlatL2(d) ## initiating faiss database
    xb = np.ascontiguousarray(np.float32(X_test.values)) ## making faiss database of epoch-specific marker expression matrix
    index.add(xb) ## indexing the database
    #########################
    landmarks = {}
    Ddensity = {}
    name = None
    f = open(ProjectName+ "/output.log", "a")
    for cellType in indx2ct.keys():
        if cellType != UnkKey:
            Xi = X_train.values[y_train==cellType]
            ct = indx2ct[cellType]
            if Xi.shape[0] > 100:
                if Xi.shape[0] > 80000: ## maximum 80K cells per cell type for LM identification
                    ri = np.random.choice(Xi.shape[0], size=80000, replace=False)
                    Xi = Xi[ri,:] ##random sub-sampling to 80k cells only 
                name =  "PCA_" + str(cellType) + ".pdf"
                print("optimizing landmarks for cell type " + str(ct))
                data = runPCA(Xi) ## applying PCA on cell matrix for this cell type
                LMind = getCenter2(data=data,name=name,nlandMarks=nlandMarks,plot=LandmarkPlots) ##LandmarkPlots: True or False if you want to have scatter plot of PCs with dense clsuetrs
                Testcellindex=np.where(y_test==cellType)[0] ## index of this cellype in this test dataset
                neighbours = len(Testcellindex)
                cells_DF=Xi[LMind.X,:] ## expression matrix of LM cells only ## ******
                ###################################################
                xq = np.ascontiguousarray(np.float32(cells_DF)) ## searching LM cells nearest neighbor in test set
                D, I = index.search(xq, neighbours) ##  ## get neighbours of all the landmarks cells for a given cell type ## I variable has rowindexes of cells that should belong to same distribution to the
                comparison={indx:None for indx in range(len(I))} ## an empty dict
                for indx in range(len(I)): ## read all the predicted neighbours of every LM one by one; reading the output of landmarks one by one
                    TruePos = set(I[indx]) & set(Testcellindex) ## common indxes between predicted indexes for this LM and orignal TP indexes
                    if (len(TruePos)/len(Testcellindex)) >= 0.09: ### atleast 0.9% of indexes are TP; otherwise I dont need that LM
                        comparison[indx] = TruePos
                ### now that we know which LM has predicted what TP indexes; we will now try to find minimum number of LM required to effectively predict more than 98% of TP indexes in test datasets
                found={}## for each of celltype index found ; how many LMs can find  this index
                rows = []
                densities = [] ## ******
                for indx in range(len(I)-1):
                    TP1 = comparison[indx] ## output of this landmark prediction; by deafult the very first index will always be used; no matter waht; its the most dense LM
                    if TP1 is not None:
                        #choice = [checkLM(TP1,comparison[indx2]) for indx2 in range(indx+1, len(I))] ##***** ## indexes that are TP predict by subsequent landmark
                        comparison = {indx2:checkLM(TP1,comparison[indx2]) for indx2 in range(indx+1, len(I))} ## if the first TP1 is better than anyone of the subsequenct LM than replace that subsequent LM value with None; else keep it as such
                        #print(sum([x is None for x in comparison.values()])) ## counting if None count is increaing over indx iterations
                        #if sum(choice) == 0: ## hurray this TP1 is better than all other TP2
                        rows = rows + [cells_DF[indx].tolist()] ## append the marker expression of this LM as list
                        densities = densities + [LMind.y.iloc[indx,0]] ## ****** adding the kernel densities of LM cells
                        found.update({tp:found.get(tp, 0) + 1 for tp in TP1})
                    if (len(found.keys())/neighbours) > 0.99: ## the current landmark has already predicted more than 98% of expected indexes
                        break ## if all the expected neigbours are already predicted; do not go ahead
                acc = (len(found.keys())/neighbours) * 100
                print('Achieved ' + str(acc) + '% of cells in Cell Type ' + str(ct) + ' with ' + str(len(rows)) + ' / ' + str(len(LMind.X)) +  ' landmark candidates')
                f.write('Achieved ' + str(acc) + '% of cells in Cell Type ' + str(ct) + ' with ' + str(len(rows)) + ' / ' + str(len(LMind.X)) +  ' landmark candidates\n')
                if acc <= 50:
                    rows = cells_DF[:,:]
                    f.write('[Note] Achieved less than 50% of cells in Cell Type ' + str(ct) + ', taking all landmark candidates\n')
                    print('[Note] Achieved less than 50% of cells in Cell Type ' + str(ct) + ', taking all landmark candidates')
                landmarks[cellType] = pa.DataFrame(rows, columns=X_train.columns)
                Ddensity[cellType] = densities ## list of LM densities (for each LM cell) in each cell type
            else:
                rows = Xi[:,:]
                f.write('[Note] Less than 100 cells in Cell Type ' + str(ct) + ', taking all cells as landmarks\n')
                print('[Note] Less than 100 cells in Cell Type ' + str(ct) + ', taking all cells as landmarks')
                landmarks[cellType] = pa.DataFrame(rows, columns=X_train.columns)
                Ddensity[cellType] = densities ## list of LM densities (for each LM cell) in each cell type
            indexes = []
            name = "PCA_" + str(cellType) + "LM.pdf"
            printLMcells = False
            if printLMcells:
                plot_LM_cells_in_PCA(Xi, rows, X_train.columns, name)
            for s in range(len(rows)):
                indexes.append("LM" + str(s)) ## adding rownames to landmark cells to probe them later ,e.g. LM0; LM1;LM2... dependng upon number of landmark cells
            landmarks[cellType].index = indexes
    f.close()
    return Sample(X=landmarks,y= Ddensity) ## a dict of cell type and best landmarks (panda df) with expression values

def runPCA(X):
    Xi = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2) ## since these are pre-sorted cells (hand-gated), we donot expect much variance
    X_pca = pca.fit_transform(Xi)
    return X_pca ## check for laoding


def svc_param_selection(X, y, nfolds,ct,threads,X_test,y_test):
    print("Training SVM for Cell Type: " + str(ct))
    param_grid = {'base_estimator__C': [0.1,0.01,0.001], 'base_estimator__gamma': [1,0.1,0.01], 'base_estimator__kernel': ['rbf']}
    if X.shape[0] > 15000: ## too many cells for svm to complete under practical time frame. 
        n_estimators = 35 ## number of times data will be randomly bagged 
        max_samples = 8000 ## to number of cells to be used for classification 
        n_iter_search = 10 ## number of random combinations of SVM hyper-parameters 
        bsp = True ## bootstrap ; false means no replcemenet while  bagging samples 
    else:
        n_estimators = 10
        max_samples = 1.0 ## take all 
        n_iter_search = 20
        bsp = True ## bootstrap 
    start = datetime.datetime.now()
    #cv=[(slice(None), slice(None))]
    cv = ShuffleSplit(test_size=0.20, n_splits=1, random_state=0)
    grid = RandomizedSearchCV(BaggingClassifier(base_estimator=SVC(probability=True),
                                                random_state = 123, max_samples=max_samples, 
                                                n_estimators=n_estimators, bootstrap=bsp, 
                                                n_jobs = threads), n_iter=n_iter_search, param_distributions=param_grid, cv=cv,verbose=1).fit(X, y)
    ### (VERY SLOW)) grid = RandomizedSearchCV(AdaBoostClassifier(base_estimator=SVC(probability=True),random_state = 123, learning_rate=0.3, n_estimators=100), n_iter=n_iter_search, param_distributions=param_grid, cv=2,verbose=1, n_jobs= threads).fit(X, y)
    finish = datetime.datetime.now()
    TT = finish-start
    print ("** SVM execution time : " + str(finish-start) + "  (Days:hr:mins:secs.ms)")
    svMod = grid.best_estimator_
    print('Best score for ', str(ct) , ' : ' , grid.best_score_)
    y_test_pred = svMod.predict(X_test)
    accu1 = precision_score(y_test, y_test_pred, average='micro')
    # accu1 = accuracy_score(y_test_pred, y_test)
    accu11 = f1_score(y_test, y_test_pred, average='weighted')
    print('SVM for cell type: ' + str(ct) + ' // Precision : {:.2f}'.format(accu1) + ' // F1-score : {:.2f}'.format(accu11))
    return Sample(X=svMod, y= TT,z=accu11)

def run_multi_layer_perceptron(X_train, y_train, X_test, y_test,threads,ct):
    print ("Training Multi-Layer Perceptron...." )
    param_grid = {'alpha': [1, 0.1,0.01], 'learning_rate_init':[0.01,0.1,1.0]}
    n_iter_search = 10
    if X_train.shape[0] > 100000:
        vf = 0.30  ## if the cell count in celltype is large then use only 70% of random cells for tree generation
    else:
        vf = 0.20  ## else use 80% of cells
    start = datetime.datetime.now()
    mlp = MLPClassifier(max_iter=2000, solver='sgd', learning_rate = 'adaptive', early_stopping=True, validation_fraction=vf, verbose=True, tol=0.001)
    cv = ShuffleSplit(test_size=0.20, n_splits=1, random_state=0)
    grid = RandomizedSearchCV(mlp, n_iter=n_iter_search, param_distributions=param_grid, cv=cv,verbose=1, n_jobs=threads).fit(X_train, y_train)
    finish = datetime.datetime.now()
    TT = finish - start
    print ("** MLP execution time : " + str(finish-start) + "  (Days:hr:mins:secs.ms)")
    model = grid.best_estimator_
    y_test_pred = model.predict(X_test)
    accu1 = precision_score(y_test, y_test_pred, average='micro')
    accu11 = f1_score(y_test, y_test_pred, average='weighted')
    print('Multi-Layer-Percepteron for cell type: ' + str(ct) + ' // Precision : {:.2f}'.format(accu1) + ' // F1-score : {:.2f}'.format(accu11))
    return Sample(X=model,y=accu11, z = TT)

def run_XGboost(X_train, y_train,X_test, y_test, ct,plotlogLoss,threads):
    eval_set = [(X_train, y_train), (X_test, y_test)]
    if X_train.shape[0] > 150000:
        subsampleThresh = 0.70  ## if the cell count in celltype is large then use only 70% of random cells for tree generation
    else:
        subsampleThresh = 0.80  ## else use 80% of cells
    posCount = len(np.where(y_train==1)[0])
    negCount = len(np.where(y_train==0)[0])
    spw = negCount / posCount
    start = datetime.datetime.now()
    xgbmdl = XGBClassifier(learning_rate=0.3, subsample=subsampleThresh, n_estimators=2000, max_depth=6,
                           objective='binary:logistic', gamma=2, reg_alpha=0.005, scale_pos_weight=spw,feature_selector="thrifty", top_k=5,
                           min_child_weight=3, nthread=threads).fit(X_train, y_train, early_stopping_rounds=10,
                                                                    eval_metric=["error@0.60", "logloss"], eval_set=eval_set, verbose=True)
    end = datetime.datetime.now()
    TT = end - start
    print ("XGboost execution time : " + str(end-start) + "  (Days:hr:mins:secs.ms)")                                                              
    y_test_pred = xgbmdl.predict(X_test)
    accu1 = precision_score(y_test, y_test_pred, average='micro')
    accu11 = f1_score(y_test, y_test_pred, average='weighted')
    print('XGBoosting for cell type: ' + str(ct) + ' // Precision : {:.2f}'.format(accu1) + ' // F1-score : {:.2f}'.format(accu11))
    if plotlogLoss == True:
        plot_XGboostLogLoss(model=xgbmdl)
        plt.bar(range(len(xgbmdl.feature_importances_)), xgbmdl.feature_importances_)
        plt.show()
        plt.clf()
    return Sample(X=xgbmdl,y=accu11,z = TT)

def QDAboost(X_train,y_train,X_test,y_test,my_test,ct,obj0,plotlogLoss,threads,method,ProjectName,CT):
    table = {}
    mlpMod_ = None
    ensemble = None
    svMod = None
    xgbmdl = None
    TimeTakenMLP = "N/A"
    TimeTakenSVM = "N/A"
    TimeTakenXGB = "N/A"
    F1stats = {}
    F1stats['Celltype'] = str(ct)
    #F1stats['LDA']=None
    #F1stats['MLP']=None
    #F1stats['SVM']=None
    #F1stats['Ensemble'] = None
    ### Multi-class LDA ###
    if method == "b" or method == "l":
        print("Running Multi-class LDA for cell type " + str(ct))
        y_test_pred = obj0.X.predict(X_test)
        accu2 = precision_score(my_test, y_test_pred, average='micro')
        accu22 = f1_score(my_test, y_test_pred, average='weighted')
        F1stats['LDA'] = accu22
        table[accu2] = obj0.X
    #### Binary Multi-Layer-Percepteron ####
    if method == "m" or method == "b" or method == "e":
        print("Running Multi-Layer perceptron for cell type " + str(ct))
        mlpout = run_multi_layer_perceptron(X_train, y_train, X_test, y_test,threads,ct)
        mlpMod_ =  mlpout.X
        table[mlpout.y] = mlpMod_
        TimeTakenMLP = mlpout.z.total_seconds()
        F1stats['MLP'] = mlpout.y
    #### Binary XGboost ###
    if method == "x" or method == "b" or method == "e":
        print("Running XGboost for cell type " + str(ct))
        samout = run_XGboost(X_train, y_train, X_test, y_test, ct, plotlogLoss, threads)
        xgbmdl  = samout.X
        table[samout.y] = xgbmdl
        TimeTakenXGB = samout.z.total_seconds()
        F1stats['XGBoost'] = samout.y
    ####### Training and Testing SVM #########
    if method=="e":
        print("Running SVM for cell type " + str(ct))
        svSam = svc_param_selection(X_train, y_train, 2, ct, threads,X_test,y_test)
        svMod = svSam.X
        TimeTakenSVM = svSam.y.total_seconds()
        F1stats['SVM'] = svSam.z
        ############################################
        ##### Ensemble model: Three algorithms #####
        ############################################
        print("Ensembling Models...")
        start = datetime.datetime.now()
        estimators = [('XGbooost', xgbmdl), ('SVM', svMod), ('MLP',mlpMod_)]
        # create our voting classifier, inputting our models
        ensemble = VotingClassifier(estimators, voting='soft', weights=[2,1,1])
        # fit model to training data
        ensemble.fit(X_train, y_train)  # test our model on the test data
        y_test_pred = ensemble.predict(X_test)
        ensemblePrecision = precision_score(y_test, y_test_pred, average='micro')
        ensembleF1 = f1_score(y_test, y_test_pred, average='weighted')
        finish = datetime.datetime.now()
        F1stats['Ensemble'] = ensembleF1
        print("**********************************")
        print("Ensemble execution time : " + str(finish-start) + "  (Days:hr:mins:secs.ms)")
        print("**********************************")
        print('Ensemble Model Result cell type: ' + str(ct) + ' // Precision : {:.2f}'.format(ensemblePrecision) + ' // F1-score : {:.2f}'.format(ensembleF1))
        print("**********************************")
    Time = pa.DataFrame({"Celltype":str(ct),"XGboost":[TimeTakenXGB],"MLP":[TimeTakenMLP],"SVM":[TimeTakenSVM], "CellCount": X_train.shape[0]})
    if method == "b" or method == "e":
        bestAcc = max(list(table.keys())) ## model with maximumm precision
        model = table[bestAcc]
    else:
        model = "N/A"
    ### Saving models on the go ###
    path = ProjectName + "/models_/C" + str(CT)
    os.mkdir(path)
    objbest = path + '/C1.PyData' ##.X
    joblib.dump(model, objbest)
    objMLP =  path + '/C2.PyData' ##.l
    joblib.dump(mlpMod_, objMLP)
    objensemble =  path + '/C3.PyData' ##.z
    joblib.dump(ensemble, objensemble)
    objXGB = path + '/C4.PyData' ##.y
    joblib.dump(xgbmdl, objXGB)
    objSVM = path + '/C5.PyData' ##.q
    joblib.dump(svMod, objSVM)
    #return Sample(X=model, y=xgbmdl, z = ensemble, l=mlpMod_, o=svMod,p=Time,q=F1stats)
    return Sample(X=None, y=None, z =None, l=None, o=None,p=Time,q=F1stats)


def countCTS(Y,ct2expr,indx2ct,CT):
    res = {}
    for CT2 in ct2expr.keys():
        count = len(np.where(Y==CT2)[0])
        res[indx2ct[CT2]] = (count/Y.shape[0])*100
    return pa.DataFrame(res, index = [indx2ct[CT]])
   
def trainHandgated(ExprSet,ct2expr, relevantMarkers,CTs,nlandMarks,LandmarkPlots,indx2ct,UnkKey,plotlogLoss,threads,method,ungated,postProbThresh,ProjectName):
    print("Splitting into train (60%) and test set (40%)")
    skf = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=randint(0,100000))
    models={}
    freqCT={}
    obj0 = None
    f = open(ProjectName + "/output.log", "a")
    #### Saving objects as soon as possible #####
    os.mkdir(ProjectName + "/models_")
    frq = ProjectName + "/others_/indx2ct.PyData"
    joblib.dump(indx2ct, frq)
    frq = ProjectName + "/others_/UnkKey.PyData"
    joblib.dump(UnkKey, frq)
    frq = ProjectName + "/others_/relevantMarkers.PyData"
    joblib.dump(relevantMarkers, frq)
    frq = ProjectName + "/others_/Method.PyData"
    joblib.dump(method, frq)
    ##############################################
    for train_index, test_index in skf.split(ExprSet.X,ExprSet.y):
        X_train = ExprSet.X.iloc[train_index,:]
        X_test = ExprSet.X.iloc[test_index,:]
        y_train = ExprSet.y.iloc[train_index,:].values.ravel()
        y_test = ExprSet.y.iloc[test_index,:].values.ravel()
        
    if (method == "x") or (method == "e") or (method == "m") or (method == "b"): ## if method is not multi-class LDA ; else take this above model
        obj = findLandmarks(X_train,y_train,X_test,y_test,CTs,nlandMarks,LandmarkPlots,relevantMarkers,indx2ct,UnkKey,ProjectName) ### identify the best landmarks with non-redundant outcome and good density
        lmarks = obj.X
        ## Saving Landmark cells object ####
        f.write("LandMark cells saved as obj object \n")
        frq = ProjectName + "/others_/obj.PyData"
        joblib.dump(obj, frq)
        ####################################
        f.write("Celltype, Number_of_landmark_cells\n")
        for keys,values in lmarks.items():
            f.write(str(indx2ct[keys]) + "," + str(values.shape[0]))
            f.write('\n')
        ## Next step is to create celltype specific model
        d = len(relevantMarkers) # number of markers
        np.random.seed(12)
        index = faiss.IndexFlatL2(d) ## initiating faiss database
        xb = np.ascontiguousarray(np.float32(ExprSet.X.values)) ## making faiss database of epoch-specific marker expression matrix
        index.add(xb) ## indexing the database
        Time = pa.DataFrame({"Celltype":[],"XGboost":[],"MLP":[],"SVM":[],"CellCount": []})
        #F1stats = pa.DataFrame({"Celltype":[],"XGboost":[],"MLP":[],"SVM":[],"Ensemble": []})
        F1stats = pa.DataFrame([])
        cDF = pa.DataFrame([])
        for CT in ct2expr.keys():
            if CT != UnkKey: ## Ignoring unknown cells for classification
                neighbours = ct2expr[CT].X.shape[0]
                if neighbours < 10:
                    sys.exit("Error: the number of cells for the cell type " + str(indx2ct[CT]) + " is less than 20. Please either remove the celltype from input training FCS files or add more training FCS datasets for this cell type")
                LMs = lmarks[CT]
                xq = np.ascontiguousarray(np.float32(LMs.values))
                D, I = index.search(xq, neighbours) ##  ## get neighbours of all the landmarks cells for a given cell type ## I variable has rowindexes of cells that should belong to same distribution to the
                indexes = I.ravel()
                Uindexes = np.unique(indexes)
                ## label the indexes ## this will include cell types from multiple CTs with non-linear boundaries. SVM can help
                X = ExprSet.X.iloc[Uindexes,:]
                Y = ExprSet.y.iloc[Uindexes,:]
                #### Check the difference between actual number of cells in this cell-type vs found as nearest neighbour using the given landmark cells
                f1 = len(np.where(ExprSet.y.values==CT)[0])
                f2 = len(np.where(Y.values == CT)[0])
                TrueNeaNeigh = (f2/f1) * 100
                tmp = countCTS(Y,ct2expr,indx2ct,CT)
                cDF = pa.concat([cDF,tmp], axis=0)
                f.write("The %age of True Nearest Neighbour for modelling for the cell type " + str(indx2ct[CT]) + " is " + str(TrueNeaNeigh) + "% with TP " + str(f2) + " cells\n")
                mY = Y[:]
                Y = Y.applymap(lambda x: 0 if x != CT else 1) # IN labels all celtypes other than this will be zero
                if len(np.unique(Y)) > 1:
                    for train_index, test_index in skf.split(X, Y):
                        X_train = X.iloc[train_index, :]
                        X_test = X.iloc[test_index, :]
                        y_train = Y.iloc[train_index, :].values.ravel()
                        y_test = Y.iloc[test_index, :].values.ravel()
                        my_test = mY.iloc[test_index, :].values.ravel() ## this has all the classes in test set instead of just binary classes
                        #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
                        models[CT] = QDAboost(X_train,y_train,X_test,y_test,my_test,indx2ct[CT],obj0,plotlogLoss,threads,method,ProjectName,CT) ## output is a sample obj
                        Time = pa.concat([Time,models[CT].p], axis=0)
                        F1stats = pa.concat([F1stats,pa.DataFrame([models[CT].q])], axis=0)
                        f.write("For Celltype: " + str(CT) + " Model saved\n")
                else: ## Faiss accurately predicted only one type of cell ; thats like impossible but lets just assume it happened
                    gmm = GMM(n_components=1).fit(X)
                    models[CT] = gmm
                freqCT[CT] = ct2expr[CT].X.shape[0]/ExprSet.X.shape[0]
        ## printing abundance table (cell type frequency observed in each cell type specific dataset ###
        f.close()
        cDF['CountCTS'] = ['CountCTS'] * cDF.shape[0]
        cDF.to_csv(ProjectName + "/output.log", mode='a',sep='\t')
        Time['Time'] = ['Time'] * Time.shape[0]
        Time.to_csv(ProjectName + "/output.log", mode='a',sep='\t', index=False)
        F1stats['F1stats'] = ['F1stats'] * F1stats.shape[0]
        F1stats.to_csv(ProjectName + "/output.log", mode='a',sep='\t', index=False)
        print("Saving all objects")
        saveObjects(ProjectName, obj, models, freqCT, indx2ct,UnkKey,relevantMarkers,method)
        f = open(ProjectName + "/output.log", "a")
        f.write("*** Session saved ****\n")
        f.close()
    elif method == "l":
        obj = AllvsAllQDA(X_train, y_train,X_test,y_test,indx2ct,relevantMarkers,UnkKey)
        models = None
        freqCT = None
        Time = None
        cDF = pa.DataFrame([])
        saveObjects(ProjectName, obj, models, freqCT, indx2ct,UnkKey,relevantMarkers,method)
    else:
        sys.exit("[Error]: Unknown method =>>>> " + str (method))
    #return Sample(X=obj, y=models, z=freqCT) ## obj.y is the densities of LM cells in obj.X ## obj0.X is single multi-class QDA model

def saveObjects(ProjectName,obj,models,freqCT,indx2ct,UnkKey,relevantMarkers,method):
    '''os.mkdir(ProjectName + "/models_")
    for CT,values in models.items():
        path = ProjectName + "/models_/C" + str(CT)
        os.mkdir(path)
        objbest = path + '/C1.PyData' ##.X
        joblib.dump(values.X, objbest)
        objMLP =  path + '/C2.PyData' ##.l
        joblib.dump(values.l, objMLP)
        objensemble =  path + '/C3.PyData' ##.z
        joblib.dump(values.z, objensemble)
        objXGB = path + '/C4.PyData' ##.y
        joblib.dump(values.y, objXGB)
        objSVM = path + '/C5.PyData' ##.q
        joblib.dump(values.o, objSVM)'''
    frq = ProjectName + "/others_/freqCT.PyData"
    joblib.dump(freqCT, frq)
    frq = ProjectName + "/others_/obj.PyData"
    joblib.dump(obj, frq)
    frq = ProjectName + "/others_/indx2ct.PyData"
    joblib.dump(indx2ct, frq)
    frq = ProjectName + "/others_/UnkKey.PyData"
    joblib.dump(UnkKey, frq)
    frq = ProjectName + "/others_/relevantMarkers.PyData"
    joblib.dump(relevantMarkers, frq)
    frq = ProjectName + "/others_/Method.PyData"
    joblib.dump(method, frq)


def printACC(indx2ct, origLabels, predictedLabels,sampleID, samPre, samF1, samrecall, type1):
    result = pa.DataFrame({"SampleID":[],"CellType":[],"precision":[],"recall":[],"F1":[], "TP":[], "FP":[], "FN":[] ,"CellCount_Exp":[], "CellCount_obs":[], "Sampleprecision":[], "SampleF1":[], "SampleRecall":[], "Type":[]})
    for ct in indx2ct.values():
        if ct != "Unknown22":
            print(str(ct) + "...")
            indx = np.where(ct==origLabels)[0]
            indx2 = np.where(ct==np.array(predictedLabels))[0]
            common = list(set(indx).intersection(indx2))
            TP = len(common)
            FN = len(indx) - len(common)
            FP = len(indx2) - len(common)
            if TP == 0 and FP == 0:
                precision = 0
            else:
                precision = TP / (TP + FP)
            if TP == 0 and FN == 0:
                recall= 0
            else:
                recall = TP / (TP + FN)
            if precision != 0 and recall != 0:
                F1 = 2 * ((precision * recall) / (precision + recall ))
            else:
                F1= 0
            if precision != 2:
                print ('Precision for celltype : ' + str(ct) + " " + str(type1) + " having " + str(len(indx2)) + "/" + str(len(indx)) + ' cells is : {:.2f}'.format(precision) + ' & recall is : {:.2f}'.format(recall) + ' & F1 score is :  {:.2f}'.format(F1))
                print ('###############################################################################')
                tmp = pa.DataFrame({"SampleID":[sampleID], "CellType":[ct],"precision":[precision],"recall":[recall],"F1":[F1], "TP":[TP], "FP":[FP], "FN":[FN],"CellCount_Exp":[len(indx)], "CellCount_obs":[len(indx2)],"Sampleprecision":[samPre], "SampleF1":[samF1], "SampleRecall":[samrecall], "Type":[type1]}, index=[ct])
                result = pa.concat([result,tmp], axis=0)
    return result

def CelltypePredPerSample(expr,SelfObject,relevantMarkers,indx2ct,origLabels,UnkKey,sampleID,postProbThresh,method,hideungated,ProjectName,calcPvalue):
    X = np.array(expr.loc[:,relevantMarkers])## a marker expression matrix for each sample; made by comboining one block from each cell
    f = open(ProjectName + "/output.log", "a")
    f.write("\n annotating sample: " + sampleID + "\n")
    if method != "l": ## method is not mLDA 
        ensemble_ = SelfObject.y ## Q/LDA/Xgboost/GMM model with highest F1 score
        obj = SelfObject.X ## a dict; chosen landmark cells (with expression values) for each cell type
        landmarks = obj.X
        freqCT = SelfObject.z ##  frequency/proportion of a given cell type in hand-gated cells
        d = len(relevantMarkers)  # number of markers
        np.random.seed(1234)
        index = faiss.IndexFlatL2(d) ## initiating faiss database
        xb = np.ascontiguousarray(np.float32(X)) ## making faiss database of epoch-specific marker expression matrix
        index.add(xb) ## indexing the database
        aa = np.zeros([X.shape[0], len(freqCT)], dtype=np.float64)
        CTindx=0
        for CT, EN in freqCT.items(): ## this dict has cell-type as key and expected prop. as value  ##EN : number of neighbours you need; this can be the number of cells expected for a given cell type ## count the number of cells supposed to be there in the dataset ##
            print(".", end="")
            ################## Nearest Neighbourhood approximation #############################
            neighbours = int(EN * len(X))
            xq = np.ascontiguousarray(np.float32(landmarks[CT].values)) ## getting from the dict ; should be panda data frame of landmark cells with column in same order as orignal expr dataframe ## for each celltype get the landmarks cells (as expression values of all markers from 'eset') ##
            D, I = index.search(xq, neighbours) ##  ## get neighbours of all the landmarks cells for a given cell type ## I variable has rowindexes of cells that should belong to same distribution to the
            indexes = list(np.unique(I.ravel())) ## remove the duplicate indexes predicted to be neighbours of all landmark cells
            tmpdf = np.concatenate((landmarks[CT].values,X[indexes]), axis=0) ## the first n cells in this case is landmark cell which is also used in modelling ## first cell is also the cell closest to the query cell in faiss
            ##################### scoring with model ##########
            tmpdf = pa.DataFrame(np.delete(tmpdf, slice(landmarks[CT].shape[0]), 0), columns=relevantMarkers) ## removing landmark cells from the cell-type specific exp matrix
            if method == "e": ## ensemble method 
                postProb  = ensemble_[CT].z.predict_proba(tmpdf)  ## prob. prob to cells using cell type specific SVM+QDA+XGboost ensemble model (soft maximum voting)
                column = np.argmax(ensemble_[CT].z.classes_) ## column number of cell type in the model; .y only for SVM ;; .z only for ensemble ## .l only for one-class QDA and .o only for mult-LDA
            elif method == "b": ## best model per cell type from XGboost/Multi-layer-percep/mLDA 
                postProb  = ensemble_[CT].X.predict_proba(tmpdf)  ## prob. prob to cells using cell type specific SVM+QDA+XGboost ensemble model (soft maximum voting)
                column = np.argmax(ensemble_[CT].X.classes_) ## column number of cell type in the model; .y only for SVM ;; .z only for ensemble ## .l only for one-class QDA and .o only for mult-LDA
            elif method == "m": ## multi-layer perceptron
                postProb  = ensemble_[CT].l.predict_proba(tmpdf)  ## prob. prob to cells using cell type specific SVM+QDA+XGboost ensemble model (soft maximum voting)
                column = np.argmax(ensemble_[CT].l.classes_) ## column number of cell type in the model; .y only for SVM ;; .z only for ensemble ## .l only for one-class QDA and .o only for mult-LDA]
            elif method == "x": ## XGboost
                postProb = ensemble_[CT].y.predict_proba(tmpdf)  ## prob. prob to cells using cell type specific model
                column = np.argmax(ensemble_[CT].y.classes_) ## column number of cell type in the model
            else:
                print ("Warning: Unknown method in argument [method]. Using XGboost")
                postProb = ensemble_[CT].y.predict_proba(tmpdf)  ## prob. prob to cells using cell type specific model
                column = np.argmax(ensemble_[CT].y.classes_) ## column number of cell type in the model
            tmp1 = list(postProb[:,column]) ## posterior prob. of cell to belong to this cell type using best classifier
            tmp1 = [r if r >postProbThresh else 0 for r in tmp1] ## if post. prob. of cell to be this cell type is <= 0.5 then lets not consider this cell as the CT cell type; make its post prob. 0
            c=Counter(tmp1)
            len(c.keys())
            for i in range(len(indexes)):
                aa[indexes[i], CTindx] = tmp1[i]
            CTindx += 1
        print("|")
        aa = pa.DataFrame(aa,columns=freqCT.keys())
        rowsum = aa.sum(axis=1)
        unknown = pa.DataFrame([1.0 if r == 0 else 0 for r in rowsum], columns=[0]) ## if the sum of the post. prob. across all cell type for a given cell is zero then make its post. prob. == 1 for "Unknown"cell type
        y_test_pred_prob = pa.concat([aa,unknown],axis=1)
        labels = y_test_pred_prob.idxmax(axis=1)
        indx2ct[0] = "Unknown"
        predictedLabels = [indx2ct[l] for l in labels]
    else:
        lda_ = SelfObject.X.X
        indx2ct[0] = "Unknown"
        y_test_pred = lda_.predict(X)
        y_test_pred_prob = lda_.predict_proba(X)
        y_test_pred[np.max(y_test_pred_prob, axis=1) < postProbThresh] = 0
        predictedLabels = [indx2ct[l] for l in y_test_pred]
    result= None
    df = None
    if origLabels is not None and len(origLabels) > 2:
        Pvalue = "NA"
        F1= f1_score(origLabels,predictedLabels,average="weighted")
        if calcPvalue:
            Pvalue = calculatePvalue(origLabels, predictedLabels, F1) ## calculating p-value of F1 SCORE for this sample 
        precision = precision_score(origLabels,predictedLabels, average='micro')
        recall = recall_score(origLabels,predictedLabels, average='micro')
        print("Prediction value: " + str(precision) + "& F1 score is " + str(F1) + " & P-value : " + str(Pvalue))
        f.write("Prediction value: " + str(precision) + "& F1 score is " + str(F1) + "& P-value : " + str(Pvalue))
        result = printACC(indx2ct, origLabels, predictedLabels, sampleID, F1,precision,recall,"With_UnGated")
        maxPostProb = np.max(y_test_pred_prob, axis=1)
        df = pa.DataFrame({'SampleID': sampleID, 'CellTypeOrignal': origLabels.values.tolist(), 'CellTypePredicted': predictedLabels,
                           'PosteriorProb': maxPostProb})
        #### Excluding Unknown cells from these dataset ###
        predictedLabels2 = list(compress(predictedLabels, origLabels != "Unknown"))
        origLabels2 = origLabels[origLabels != "Unknown"]
        F1= f1_score(origLabels2,predictedLabels2,average="weighted")
        precision = precision_score(origLabels2,predictedLabels2, average='micro')
        recall = recall_score(origLabels2,predictedLabels2, average='micro')
        result2 = printACC(indx2ct, origLabels2, predictedLabels2, sampleID, F1, precision, recall, "Without_UnGated")
        result = pa.concat([result,result2], axis=0)
        annotatedExpression = pa.concat([expr, pa.DataFrame(predictedLabels, columns=["labels"], index=expr.index), pa.DataFrame(origLabels.values, columns=["Truelabels"], index=expr.index)], axis=1)
        len(predictedLabels)
        name = str(ProjectName) + "/" + str(sampleID) + '_labled_expr.csv'
        print("Writing file.." + name)
        annotatedExpression.to_csv(name)  ## write best cell type predicted for this sample
    else:
        print ("Merging...")
        annotatedExpression = pa.concat([expr, pa.DataFrame(predictedLabels, columns=["labels"], index=expr.index)], axis=1)
        name = str(ProjectName) + "/" + str(sampleID) + '_labled_expr.csv'
        print("Writing file.." + name)
        annotatedExpression.to_csv(name)  ## write best cell type predicted for this sample 
    f.close()
    return result, df

def calculatePvalue(origLabels, predictedLabels, F1):
    count = 0
    interations = 100
    Y_fake = origLabels.copy()
    print("Iterating and calculating P-value...")
    for i in range(interations):
        print(i)
        np.random.shuffle(Y_fake)
        F1fake= f1_score(Y_fake,predictedLabels,average="weighted")
        if F1fake >= F1:
            count = count + 1
        else:
            print(str(F1) + ' > ' + str(F1fake))
    return count / interations

def getSet(ct2expr,indx2ct,relevantMarkers,normalize,ProjectName):
    mydf = pa.DataFrame()
    annota = pa.DataFrame()
    fl = open(ProjectName + "/output.log", "a")
    for key, value in ct2expr.items():
        #print(key)
        if value.X.shape[0] < 10:
            print ("Too Low number of cells (< 10) for training the cell type: " + str(indx2ct[key]) + " ... this Cell type ignored for analysis")
            fl.write("Too Low number of cells (< 10) for training the cell type: " + str(indx2ct[key]) + " ... this Cell type ignored for analysis\n")
        celltypes= pa.DataFrame([key] * value.X.shape[0])
        annota= pa.concat([annota,celltypes])
        mydf = pa.concat([mydf,value.X])
    #if normalize is True:
    #    exp = np.arcsinh((mydf - 1.0)/5.0)
    #    mydf = pa.DataFrame(exp, columns = relevantMarkers)
    out = Sample(X=mydf,y=annota)
    fl.close()
    return out

def loadObjects(ProjectName,method):
    if method =='l':
        obj = load_Object(ProjectName + "/others_/obj.PyData", "LDA model")
        indx2ct = load_Object(ProjectName + "/others_/indx2ct.PyData", "indexes to cell types")
        UnkKey = load_Object(ProjectName + "/others_/UnkKey.PyData", "Unknown Keys")
        relevantMarkers = load_Object(ProjectName + "/others_/relevantMarkers.PyData", "Relevant Markers")
        models = None
        freqCT = None
    else:
        freqCT = load_Object(ProjectName + "/others_/freqCT.PyData", "frequencies")
        obj = load_Object(ProjectName + "/others_/obj.PyData", "Landmarks")
        indx2ct = load_Object(ProjectName + "/others_/indx2ct.PyData", "indexes to cell types")
        UnkKey = load_Object(ProjectName + "/others_/UnkKey.PyData", "Unknown Keys")
        relevantMarkers = load_Object(ProjectName + "/others_/relevantMarkers.PyData", "Relevant Markers")
        models = {}
        for CT in freqCT.keys(): 
            X = load_Object(ProjectName + "/models_/C" + str(CT) + "/C1.PyData",str(CT) + "..celltype")
            y = load_Object(ProjectName + "/models_/C" + str(CT) + "/C4.PyData",str(CT) + "..celltype")
            z = load_Object(ProjectName + "/models_/C" + str(CT) + "/C3.PyData",str(CT) + "..celltype")
            l = load_Object(ProjectName + "/models_/C" + str(CT) + "/C2.PyData",str(CT) + "..celltype")
            o = load_Object(ProjectName + "/models_/C" + str(CT) + "/C5.PyData",str(CT) + "..celltype")
            models[CT] = Sample(X=X,y=y,z=z,l=l,o=o)
    return Sample(X=obj, y=models, z=freqCT,l=indx2ct, o=UnkKey,p=relevantMarkers, q=method)

def load_Object(file1,pr):
    print("loading...", pr)
    if not os.path.exists(file1):
        sys.exit("[Error while loading session]: File:  " + str (file1) + " doest not exist")
    else:
        obj = joblib.load(file1)
    return obj

def arcsinetransform(value,relevantMarkers,key):
    print("Normalizing..." + str(key))
    if len(value.columns) == len(relevantMarkers):
        value = np.clip(value.loc[:,relevantMarkers], a_min=0, a_max=None)  ## if expression is negative then make it zero as cell cannot be negatively expressed also it will create problem in normalizarion
        exp = np.arcsinh((value.X - 1.0)/5.0)
    elif len(value.columns) > len(relevantMarkers):
        expr1 = value.drop(columns=relevantMarkers, inplace=False) ## separating out other columns, if any 
        expr2 = value.loc[:,relevantMarkers] ## normalizing only the expression of relevant markers 
        expr2 = np.clip(expr2.loc[:,relevantMarkers], a_min=0, a_max=None)
        exp = np.arcsinh((expr2 - 1.0)/5.0)
        exp = pa.concat([exp,expr1],axis=1) ## again adding the columns excluded from normalization 
    else:
        print("[ERROR] Number of relevant markers are greater than markers in expression matrix of live cells")
        os.exit(0)
    return exp


def e2b(Fileinfo,relevantMarkers,nlandMarks,LandmarkPlots,plotlogLoss,threads,method,Findungated,postProbThresh,normalizeCell,ProjectName,loadModel,calcPvalue,header,ic):
    if not loadModel:
        train = load_Object(ProjectName + "/others_/train.PyData", "Training")
        indx2ct = train.y ## key is the number and value is the cell type name
        CTs  = list(indx2ct.values())
        UnkKey = 0 ## training set does not have Uknown cells
        for k,v in indx2ct.items():
            if (v == "Unknown") or (v=="Ungated") or (v=="Undefined") or (v=="Unclassified"):
                UnkKey = k
        ExprSet = getSet(train.X,indx2ct,relevantMarkers,normalizeCell,ProjectName) ## this training data is handgated cells which will also split futher into train and test set
        trainHandgated(ExprSet,train.X,relevantMarkers,CTs,nlandMarks,LandmarkPlots,indx2ct,UnkKey,plotlogLoss,threads,method,Findungated,postProbThresh,ProjectName)
        train = None ## cleaning the memory
        method = load_Object(ProjectName + "/others_/Method.PyData", "Method")
        SelfObject = loadObjects(ProjectName, method)
    else:
        method = load_Object(ProjectName + "/others_/Method.PyData", "Method")
        SelfObject = loadObjects(ProjectName, method)
        relevantMarkers = SelfObject.p
        indx2ct = SelfObject.l
        UnkKey =  SelfObject.o
        method = SelfObject.q
    print('annotating sample')
    md = pa.read_csv(Fileinfo,header=0)
    md.index = list(range(md.shape[0]))
    inputs = md.iloc[:,0]
    print ("Reading live cells for annotation...")
    output = pa.DataFrame({"SampleID":[],"CellType":[],"precision":[],"recall":[],"F1":[], "TP":[], "FP":[], "FN":[], "CellCount_Exp":[], "CellCount_obs":[], "Sampleprecision":[], "SampleF1":[], "SampleRecall":[], "Type":[]})
    PostProbCT = pa.DataFrame({'SampleID': [], 'CellTypeOrignal': [], 'CellTypePredicted': [], 'PosteriorProb': []})
    hideungated = False ## do not use ungated cells in F1/ACC analysis       
    for c,f in enumerate(inputs):
        if os.path.exists(f):
            match = re.search(r'fcs$', f)
            match2 = re.search(r'csv$', f)
            if match: ## if file is FCS
                panel, expr = fcsparser.parse(f, reformat_meta=True)
                desc = panel['_channels_']['$PnS'] ## marker name instead of meta name
                desc = desc.str.replace('.*_', '', regex=True)
                desc = desc.str.replace('-', '.', regex=True).tolist()
                expr.columns = desc
                print("FCS file: " + str(c) + "...loaded")
            elif match2: ## if file is CSV
                expr = pa.read_csv(f, delimiter=",", header=header, index_col=ic)
                print("CSV file: " + str(c) + "...loaded")
            #### lets annotaaaaaaaate this expr matrix ###
            origLabels=None
            if 'labels' in expr.columns:
                origLabels=expr.loc[:,'labels']
            sampleID =  md.iloc[c,1]
            print(str(sampleID) + '...')
            if normalizeCell:
                expr=arcsinetransform(expr,relevantMarkers,sampleID)
            result,pp = CelltypePredPerSample(expr,SelfObject,relevantMarkers,indx2ct,origLabels,UnkKey,sampleID,postProbThresh,method,hideungated,ProjectName,calcPvalue)
            if result is not None:
                output = pa.concat([output,result], axis=0)
                PostProbCT = pa.concat([PostProbCT, pp], axis=0)        
        else:
            print(str("*** Error ***: Line: " + str(c) + " or " + str(f) + "- Unknown File !!!"))
            exit
    if output.shape[0] > 0:
        name = str(ProjectName) + "/Method_" + str(method) + "_" + str("_Acc_stats.csv")
        output.to_csv(name)
        name = str(ProjectName) + "/Method_" + str(method) + "_" + str("_Posterior_Probability_Prediction_Result.csv")
        PostProbCT.to_csv(name)
        evalreport = output.loc[:,['SampleID','SampleF1','Sampleprecision','SampleRecall', 'Type']]
        evalreport = evalreport.drop_duplicates()
        sns.set(style="white")
        ax = sns.boxplot(x='Type', y='SampleF1', data=evalreport)
        ax = sns.swarmplot(x='Type', y='SampleF1', data=evalreport, color=".25", size=14, hue='SampleID', palette="muted")
        ax.set_title('Prediction Accuracy')
        plt.savefig(str(ProjectName)+'/accuracyReportF1.png')
        print('Accuracy plots saved in output directory')