from src.functions import *
'''
A tool to predict cell labels of closely related (but mutually exclusive) cell types in large scale CyTOF studies
@author: kaushik, abhinav
'''

##### Mandatory ######
handgatedFileinfo='example/Handgated.csv' 		## [Mandatory] hand-gated cells to be used for training
LiveFileinfo= 'example/LivecellsTraining.csv'    	## [Mandatory] All Live cells of the samples used for handgating (i.e. training); these samples will also be labelled
unlabelledDataset= 'example/Livecells.csv'    		## [Mandatory] All Live cells to be tested for annotation, i.e. for cell type identification
relevantMarkers = ["0","1","2","3","4","5","6","7"] 	## [Mandatory] lineage markers used for hand-gated; based on column names in FCS/CSV file  
outdir = 'MultCent' 					## Mandatory when loadModel=False] any name of choice. If directory will hold saved session and labelled CSV file.	
loadSession = '' 					## [optional; valid only when loadModel=True]  if user wants to reuse the previously built models/training then, put the name of directory having all the session files.


######### Optional #######
threads = 20 			## (Int) number of available threads to use; -1 if all available threads are to be used
method = 'x' 			## e= ensemble (XGboost+MLP+SVM) ; x = XGboost ; m=Multi-layer-perceptron ; l = multi-LDA; b = best model from x/m/l for each cell type (may use diffferent model for different cell type )
loadModel = False		## When you are using previously generated model you need to change this to 'True' and in the next parameter set the path of the CyAnno generated output folder.
postProbThresh=0.5 		## [0.0 to 1.0] Posterior Prob. threshold; if mLDA method is used then the recommended value is 0.4 else the default 0.5 should be good enough; for higher stringency increase this threshold with 1.0 is maximum
Findungated = True		## Logical; if True then 'ungated' cells will be predicted. However, orignal sample ID (live cells) must be present in LiveFileinfo. 'ungated' cells are defined as all the cells that not the part of any of 'gated' cell population.
						## if False then any of ungated cells will not be CALCULATED in training/testing (aka model generation);  though they will be identified/labelled in the validation/unlabelled dataset
						## [False] if you already have ungated popluation (as FCS/CSV file(s)) and you dont want the program to re-calculate the ungated cells then set it to False.
						## [True] if you have ungated population for some of the sample_ids but want to include ungated for other sample_ids set it to True
						## [True] if you have not included ungated population for any of the sample_id and want to include ungated population in the classification, set it to True
normalizeCell=True 		## Logical; if yes arcsine transformation with cofactor will we used to normalize both the handgated and unlabelled cell expression files.
cofactor=5.0 			## valid only when normalizecell == True; this is the cofactor for arcsin transformation of raw expression values 
header = 'infer' 		## 'infer' or 'None' ; does all the input csv files (LiveFileinfo, unlabelledDataset and handgatedFileinfo) contain header. infer means 'yes' otherwise its None
nlandMarks = 10 		## number of landmarks cells you need from each cell type. [feault 10] good enough for most studies; higher values means nore neighbouring cells; improves training but reduce execution speed 
cellCount = 20			## Minimum number of cells that should be present in the entire training dataset.
index_col = False 		## [For CSV only; False or 0] rownames in Marker expression csv file to be considered or not ; 0 means first column to use for rowname else use False; if you known that first column is marker expression value then set this to False




##############################################################################
########################### Do Not Edit ######################################
##############################################################################
plotlogLoss = False
calcPvalue = False
LandmarkPlots = False
train = None ## when loadmodel = True then obviously you dont need train object again ; all the models will be loaded as selfobject later on in the script 
if not loadModel: ## if you are not loading previously build model from ProjectName
    ProjectName = outdir
    DateTime= datetime.datetime.now()
    ProjectName = outdir + str("_") + str(DateTime.day) + str('_') + str(DateTime.month) + str('_') + str(DateTime.year) + str('_') + str(DateTime.hour) + str('_') + str(DateTime.second)
    print("Creating new directory name..." + ProjectName)
    os.makedirs(ProjectName)
    train = method0(handgatedFileinfo,filterCells,relevantMarkers,cellCount,header, index_col, Findungated,ProjectName,LiveFileinfo)
    method1(LiveFileinfo,normalizeCell,relevantMarkers,'infer',index_col, Findungated, train,ProjectName,cofactor) ##
else:
    ProjectName = loadSession

e2b(Fileinfo=unlabelledDataset,
            relevantMarkers=relevantMarkers,
            nlandMarks=nlandMarks,
            LandmarkPlots=LandmarkPlots,
            plotlogLoss=plotlogLoss,
            threads=threads,
            method=method,
            Findungated=Findungated,
            postProbThresh=postProbThresh,
            normalizeCell=normalizeCell,
            ProjectName=ProjectName,
            loadModel=loadModel,
            calcPvalue=calcPvalue,
            header=header,
            ic=index_col)


