# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:20:11 2020

@author: duygu86
"""

import pandas as pd
import numpy as np
import datetime
import re
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import Lasso,LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,confusion_matrix
from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rc('font', size=32)

def autolabel(rects):
    for rect in rects:
        bb = rect.get_bbox()
        if bb.y0!=bb.y1:
            h = (bb.y0+bb.y1)/2
            w = (bb.x0+bb.x1)/2
            ax.text(w,h, '%d'%int(bb.y1-bb.y0),
                    ha='center', va='center')


data = pd.read_excel('ODI-2020-2.xlsx');
keys = data.columns

# Delete empty spaces at the end
for i,k in enumerate(data[keys[0]]):
    k = re.sub(' *$','',k);
    data[keys[0]][i] = k

records, attribs = data.shape;
print('Number of Records --->',records)
print('Number of Attributes --->',attribs)
print('Attribute Types --->',keys)
for k in keys:
      p = pd.unique(data[k]);
      print('Attribute',k,'have',p.shape[0],'different values')
      if p.shape[0]<6:
          print('Possible values',p)

data[keys[10]][data[keys[10]]<0] = np.nan;
data[keys[10]][data[keys[10]]>100] = np.nan;
          
data['Age'] = 0;
today = datetime.datetime.now();

nat = pd.to_datetime('111',format='%d.%m.%Y',errors='coerce');
for i,k in enumerate(data[keys[7]]):
    k2 = pd.to_datetime(k,format='%d.%m.%Y',errors='coerce');
    if np.isnan(k2.year):
        k2 = pd.to_datetime(k,format='%m.%d.%Y',errors='coerce');
    if np.isnan(k2.year):
        k2 = pd.to_datetime(k,format='%Y.%m.%d',errors='ignore');

    if np.isnan(k2.year):
        data['Age'][i] = np.nan;
    else:
        year_difference = today.year - k2.year;
        check_this_year = (today. month, today.day) < (k2.month, k2.day);
        age = year_difference - check_this_year;
        data['Age'][i] = age;


descd = data.describe()

## Program Gender Distribution
programs = pd.unique(data[keys[0]])

prognames = np.array(['Computational Science', 
                     'QRM',
                     'BA', 
                     'CS',
                     'Dig. Bus. and Innov.', 
                     'Econometrics',
                     'AI', 
                     'Bioinf. and Sys. Bio.',
                     'Econ. and Op. Res.', 
                     'Big Data Eng.',
                     'Fin. and Tech.', 
                     'Business Adm.',
                     'Sci. Comp.', 
                     'Econ. and Data Sci.',
                     'Bioinf.', 
                     'Par. and Dist. Comp. Sys.',
                     'Inf. Sci.', 
                     'Finance', 
                     'Phys. and Astro.',
                     'Exchange', 
                     'Health Sci.', 
                     'Human Mov. Sci. Res.',
                     'Human Lang. Tech.', 
                     'Masters',
                     'Inf. Stud.-Data Sci.',
                     'Inf. Stud.-Inf. Sys.', 
                     'Med. Inf.',
                     'Data Sci.',
                     'CLS', 
                     'CPS', 
                     'Foren. Sci.', 
                     'Par. Comp. Sys.',
                     'Mech. Eng.'],
      dtype=object)
prog_dist = np.zeros((programs.shape[0],))
prog_male = np.zeros((programs.shape[0],))
prog_female = np.zeros((programs.shape[0],))
prog_unk = np.zeros((programs.shape[0],))

for i,p in enumerate(programs):
    prog_dist[i] = (data[keys[0]] == p).sum();
    temp = data[keys[5]][data[keys[0]] == p];
    prog_male[i] = (temp == 'male').sum();
    prog_female[i] = (temp=='female').sum();
    prog_unk[i] = (temp=='unknown').sum();
    
print(prog_dist)    


fig = plt.figure(figsize=(25,10))
ax = fig.add_axes([0,0,1,1])
rect1 = ax.bar(prognames,prog_male)
rect2 = ax.bar(prognames,prog_female,bottom=prog_male)
rect3 = ax.bar(prognames,prog_unk,bottom=prog_male+prog_female)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
   

ax.legend(labels=['Men', 'Women','Unknown'])
plt.title('Number of Students in Programs by Gender')
autolabel(rect1)
autolabel(rect2)
autolabel(rect3)

## Program ML Distribution
prog_ml = np.zeros((programs.shape[0],))
prog_noml = np.zeros((programs.shape[0],))
prog_unkml = np.zeros((programs.shape[0],))

for i,p in enumerate(programs):
    temp = data[keys[1]][data[keys[0]] == p];
    prog_ml[i] = (temp == 'yes').sum();
    prog_noml[i] = (temp=='no').sum();
    prog_unkml[i] = (temp=='unknown').sum();
    
# fig = plt.figure(figsize=(20,15))
(fig,(ax1,ax2,ax3,ax4)) = plt.subplots(1,4,figsize=(75,10))
rect1 = ax1.bar(prognames,prog_ml)
rect2 = ax1.bar(prognames,prog_noml,bottom=prog_ml)
rect3 = ax1.bar(prognames,prog_unkml,bottom=prog_ml+prog_noml)
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
    # tick.set_visible(False)

ax1.legend(labels=['ML Taken', 'ML Not Taken','Unknown'])
ax1.set_title('Number of Students in Programs by ML Course Taking')

## Program IR Distribution
prog_ml = np.zeros((programs.shape[0],))
prog_noml = np.zeros((programs.shape[0],))
prog_unkml = np.zeros((programs.shape[0],))

for i,p in enumerate(programs):
    temp = data[keys[2]][data[keys[0]] == p];
    prog_ml[i] = (temp == 1).sum();
    prog_noml[i] = (temp == 0).sum();
    prog_unkml[i] = (temp=='unknown').sum();
    
rect1 = ax2.bar(prognames,prog_ml)
rect2 = ax2.bar(prognames,prog_noml,bottom=prog_ml)
rect3 = ax2.bar(prognames,prog_unkml,bottom=prog_ml+prog_noml)
for tick in ax2.get_xticklabels():
    tick.set_rotation(90)

ax2.legend(labels=['IR Taken', 'IR Not Taken','Unknown'])
ax2.set_title('Number of Students in Programs by IR Course Taking')

## Program Statistics Distribution
prog_ml = np.zeros((programs.shape[0],))
prog_noml = np.zeros((programs.shape[0],))
prog_unkml = np.zeros((programs.shape[0],))

for i,p in enumerate(programs):
    temp = data[keys[3]][data[keys[0]] == p];
    prog_ml[i] = (temp == 'mu').sum();
    prog_noml[i] = (temp == 'sigma').sum();
    prog_unkml[i] = (temp=='unknown').sum();
    
rect1 = ax3.bar(prognames,prog_ml)
rect2 = ax3.bar(prognames,prog_noml,bottom=prog_ml)
rect3 = ax3.bar(prognames,prog_unkml,bottom=prog_ml+prog_noml)
for tick in ax3.get_xticklabels():
    tick.set_rotation(90)

ax3.legend(labels=['Stats Taken', 'Stats Not Taken','Unknown'])
ax3.set_title('Number of Students in Programs by Statistics Course Taking')

## Program Databases Distribution
prog_ml = np.zeros((programs.shape[0],))
prog_noml = np.zeros((programs.shape[0],))
prog_unkml = np.zeros((programs.shape[0],))

for i,p in enumerate(programs):
    temp = data[keys[4]][data[keys[0]] == p];
    prog_ml[i] = (temp == 'ja').sum();
    prog_noml[i] = (temp == 'nee').sum();
    prog_unkml[i] = (temp=='unknown').sum();
    
rect1 = ax4.bar(prognames,prog_ml)
rect2 = ax4.bar(prognames,prog_noml,bottom=prog_ml)
rect3 = ax4.bar(prognames,prog_unkml,bottom=prog_ml+prog_noml)
for tick in ax4.get_xticklabels():
    tick.set_rotation(90)

ax4.legend(labels=['DB Taken', 'DB Not Taken','Unknown'])
ax4.set_title('Number of Students in Programs by DB Course Taking')
plt.show()

## Stress by Program
stress = tuple()
for i,p in enumerate(programs):
    temp = data[keys[10]][data[keys[0]] == p];
    temp = temp.dropna();
    stress +=(temp,);

fig, ax = plt.subplots(figsize=(25,10))
ax.boxplot(stress)
ax.set_xticklabels(prognames);    
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.title('Boxplots of Stress Level by Program')
plt.savefig('Figs/bplot_stress_by_program.png',dpi=200)
plt.show()

## Stress by Gender boxplpts
stress = tuple()
gends = pd.unique(data[keys[5]]);
for i,p in enumerate(gends):
    temp = data[keys[10]][data[keys[5]] == p];
    temp = temp.dropna();
    stress +=(temp,);

fig, ax = plt.subplots(figsize=(25,10))
ax.boxplot(stress)
ax.set_xticklabels(gends);    
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.title('Boxplots of Stress Level by Gender')
plt.savefig('Figs/bplot_stress_by_gender.png',dpi=200)
plt.show()

# Good day 
goodday = pd.concat((data[keys[14]],data[keys[16]]))
genders = pd.concat((data[keys[5]],data[keys[5]]))
gds = pd.unique(goodday.dropna())
gds_male = np.zeros((gds.shape[0],))
gds_female = np.zeros((gds.shape[0],))
gds_unk = np.zeros((gds.shape[0],))
for i,p in enumerate(gds):
    temp = genders[goodday == p];
    gds_male[i] = (temp == 'male').sum();
    gds_female[i] = (temp=='female').sum();
    gds_unk[i] = (temp=='unknown').sum();
    
fig = plt.figure(figsize=(25,10))
ax = fig.add_axes([0,0,1,1])
rect1 = ax.bar(gds,gds_male)
rect2 = ax.bar(gds,gds_female,bottom=gds_male)
rect3 = ax.bar(gds,gds_unk,bottom=gds_male+gds_female)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
autolabel(rect1)
autolabel(rect2)
autolabel(rect3)
ax.legend(labels=['Men', 'Women','Unknown'])
plt.title('Number of Students by Good Day Definition and Gender')
plt.savefig('Figs/goodday_by_gender.png',dpi=200)
plt.show()


## Boxplots of Age by Gender
ages = tuple()
gends = pd.unique(data[keys[5]]);
for i,p in enumerate(gends):
    temp = data['Age'][data[keys[5]] == p];
    temp[temp==0] = np.nan
    temp = temp.dropna();
    ages +=(temp,);

fig, ax = plt.subplots(figsize=(25,10))
ax.boxplot(ages)
ax.set_xticklabels(gends);    
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.title('Boxplots of Age by Gender')
plt.show()


# Dealing with attributes
# FEAT PROGRAM
prog_feats = data[keys[0]].copy();
prog_class = dict()
for i,k in enumerate(programs):
    prog_class[str(i)] = k;
    prog_feats[prog_feats==k] = i;

# FEAT ML COURSE
ml_feats = data[keys[1]].copy();
ml_feats[ml_feats=='yes'] = 1;
ml_feats[ml_feats=='no'] = 0;
ml_feats[ml_feats=='unknown'] = 0;

# FEAT IR COURSE
ir_feats = data[keys[2]].copy();
ir_feats[ir_feats=='unknown'] = 0;

#FEAT STATS COURSE
stat_feats = data[keys[3]].copy();
stat_feats[stat_feats=='mu'] = 1;
stat_feats[stat_feats=='sigma'] = 0;
stat_feats[stat_feats=='unknown'] = 0;

#FEAT DB COURSE
db_feats = data[keys[4]].copy();
db_feats[db_feats=='ja'] = 1;
db_feats[db_feats=='nee'] = 0;
db_feats[db_feats=='unknown'] = 0;

# FEAT GENDERS
gend_feats = data[keys[5]].copy();
gend_feats[gend_feats=='female'] = 1;
gend_feats[gend_feats=='male'] = 0;
gend_feats[gend_feats=='unknown'] = 0;

# FEAT CHOCO
choco_feats = data[keys[6]].copy();
choco_class = dict()
for i,k in enumerate(pd.unique(choco_feats)):
    choco_class[str(i)] = k;
    choco_feats[choco_feats==k] = i;

# FEAT AGE
age_feats = data['Age'].copy();
age_feats = age_feats.fillna(np.round(np.median(age_feats.dropna())))

# FEAT NEIGHBORS
neigh_feats = data[keys[8]].copy();

# FEAT STAND UP
stand_feats = data[keys[9]].copy();
stand_feats[stand_feats=='yes'] = 1;
stand_feats[stand_feats=='no'] = 0;
stand_feats[stand_feats=='unknown'] = 0;

# FEAT MONEY
money_feats = data[keys[11]].copy();
money_feats[money_feats=='-'] = 0;
money_feats = money_feats.fillna(np.round(np.median(money_feats.dropna())))

# FEAT RANDOM
random_feats = data[keys[12]].copy();
random_feats = random_feats.map(float)
u = np.sort(random_feats.dropna())
thr = u[int(len(u)*0.9)]
random_feats[random_feats>thr] = np.median(random_feats.dropna())

# FEAT BED TIME
bedtime_feats = data[keys[13]].copy();
for i,k in enumerate(bedtime_feats):
    try:
        y = re.search(':', str(k));
        u = y.regs[0][0]
        v = max(0,u-2)
        bedtime_feats[i] = int(str(k)[v:u])
    except:
        bedtime_feats[i] = 0    

btf = bedtime_feats.copy()
bstress = tuple()
btfs = np.arange(0,24);

[a,b] = np.histogram(btf.dropna(),bins=np.arange(-0.5,24.,1))
fig, ax = plt.subplots(figsize=(25,10))
ax.bar(btfs,a/a.sum())
ax.set_title('Histogram of Bed Time')
plt.show()

# FEAT GOOD DAYS
gdays = pd.unique(pd.concat((data[keys[14]],data[keys[16]])))
gday_feats1 = data[keys[14]].copy();
gday_feats2 = data[keys[16]].copy();
gday_class = dict()
for i,k in enumerate(gdays):
    gday_class[str(i)] = k;
    gday_feats1[gday_feats1==k] = i;
    gday_feats2[gday_feats2==k] = i;


stss_feats = data[keys[10]].copy();
stss_feats[stss_feats<0] = np.nan;
stss_feats[stss_feats>100] = np.nan;
stss_feats = stss_feats.fillna(np.median(stss_feats.dropna()))

# feats = pd.concat((stss_feats,prog_feats,ml_feats,ir_feats,stat_feats,db_feats,
#                    gend_feats,choco_feats,age_feats,neigh_feats,stand_feats,
#                    money_feats,random_feats,bedtime_feats,gday_feats1,gday_feats2),axis=1)
feats = pd.concat((stss_feats,ml_feats,ir_feats,stat_feats,db_feats,
                   gend_feats,age_feats,neigh_feats,stand_feats,
                   money_feats,random_feats,bedtime_feats),axis=1)
feats = feats.dropna()
outcome = feats[feats.columns[0]]
feats = feats[feats.columns[1:]]

feats2 = pd.concat((stss_feats,prog_feats,ml_feats,ir_feats,stat_feats,db_feats,
                   gend_feats,choco_feats,age_feats,neigh_feats,stand_feats,
                   money_feats,random_feats,bedtime_feats,gday_feats1,gday_feats2),axis=1)
feats2 = feats2.dropna()
outcome2 = feats2[feats2.columns[0]]
feats2 = feats2[feats2.columns[1:]]
outcome22 = outcome2.copy()
outcome22[outcome2<=25] = 0
outcome22[outcome2>25] = 1
outcome22[outcome2>50] = 2
outcome2 = outcome22

for k in feats.columns:
    feats[k] = feats[k].map(float)

for k in feats2.columns:
    feats2[k] = feats2[k].map(float)
    

# Lasso with CV
X_train, X_test, y_train, y_test = train_test_split(feats, outcome, test_size=0.25, random_state=10)


las = Lasso(alpha=0.5)
scores = cross_validate(las, X_train,y_train, cv=5, 
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True,return_estimator=True)
scrs = np.zeros((5,))
for i in range(5):
    scrs[i] = scores['estimator'][i].score(X_train,y_train)
las_best = scores['estimator'][np.argmax(scrs)]
tr_score = las_best.score(X_train,y_train)
ts_score = las_best.score(X_test,y_test)
tr_mae = mean_absolute_error(y_train, las_best.predict(X_train))
ts_mae = mean_absolute_error(y_test, las_best.predict(X_test))
tr_mse = mean_squared_error(y_train, las_best.predict(X_train))
ts_mse = mean_squared_error(y_test, las_best.predict(X_test))

# KNN classifier with CV

X_train2, X_test2, y_train2, y_test2 = train_test_split(feats2, outcome2, test_size=0.25, random_state=10)

clf = KNeighborsClassifier(n_neighbors=10, weights='uniform')
scores = cross_validate(clf, X_train2,y_train2, cv=5, 
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True,return_estimator=True)

# Scores of CV and KNN classifier
scrs = np.zeros((5,))
for i in range(5):
    scrs[i] = scores['estimator'][i].score(X_train2,y_train2)
clf_best = scores['estimator'][np.argmax(scrs)]
tr_score = clf_best.score(X_train2,y_train2)
ts_score = clf_best.score(X_test2,y_test2)

tr_cfm = confusion_matrix(y_train2,clf_best.predict(X_train2),normalize='true')
ts_cfm = confusion_matrix(y_test2,clf_best.predict(X_test2),normalize='true')

# Random forest classifier with CV)
clf2 = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=5)
scores2 = cross_validate(clf2, X_train2,y_train2, cv=5, 
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True,return_estimator=True)
# Scores of CV and Random forest classifier
scrs2 = np.zeros((5,))
for i in range(5):
    scrs2[i] = scores2['estimator'][i].score(X_train2,y_train2)
clf_best2 = scores2['estimator'][np.argmax(scrs2)]
tr_score2 = clf_best2.score(X_train2,y_train2)
ts_score2 = clf_best2.score(X_test2,y_test2)

tr_cfm2 = confusion_matrix(y_train2,clf_best2.predict(X_train2),normalize='true')
ts_cfm2 = confusion_matrix(y_test2,clf_best2.predict(X_test2),normalize='true')
