# -*- coding: utf-8 -*-

"""
@author: Idriss
"""
import pandas as pd
import numpy as np
from math import sqrt,log
import random
import warnings
warnings.filterwarnings("ignore")

import os
os.chdir('C:/Users/imghabba/Desktop/Code')

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Enable
def enablePrint():
    sys.stdout = sys.__stdout__



class Distance : 


    def hellinger_dist(self,P,Q):
        """
        P : Dictionnary containing proportions for each category in one window
        Q : Dictionnary for the next window
        """
        diff = 0
        for key in P.keys():
            diff += (sqrt(P[key]) - sqrt(Q[key]))**2
        return 1/sqrt(2)*sqrt(diff)


    def KL_divergence(self,P,Q):
        """
        This method is used in Jensen_Shannon_divergence
        """
        div = 0
        for key in list(P.keys()):
            if P[key]!=0: #Otherwise P[key]*logP[key]=0
                div += P[key]*log(P[key]/Q[key])

        return div
  

    def Jensen_Shannon_divergence(self,P,Q):
        """
        P : Dictionnary containing proportions for each category in one window
        Q : Dictionnary for the next window
        """
        M = {}
        for key in list(P.keys()):
            M.update({key:(P[key]+Q[key])/2})
        
        return 1/2*(self.KL_divergence(P,M)+self.KL_divergence(Q,M))


class Drift_Detector :

    def __init__(self,data,n_batch,categorical_variables,gamma,threshold,distance):

        self.data = data
        self.n_batch = n_batch
        self.categorical_variables = categorical_variables
        self.gamma = gamma
        self.threshold = threshold
        self.distance = distance
        

    def generate_prop_dic(self,window,union_values):
        '''
        union_values : list containing union of unique values of the two windows
        '''
        dic = {}
        df = window.value_counts()
        n = window.shape[0]
        for key in union_values:
            if key in window.unique():
                dic.update({key:df.loc[key]/n})
            else:
                dic.update({key : 0})
        return dic 

    def windows_distance(self,ref_window,current_window):

        actual_dist = 0

        for feature in self.categorical_variables:
            ref_liste_values = ref_window[feature].unique()
            current_liste_values = current_window[feature].unique()
            union_values = list(set(ref_liste_values) | set(current_liste_values))
            ref_dic = self.generate_prop_dic(ref_window[feature],union_values)

            current_dic = self.generate_prop_dic(current_window[feature],union_values)
            
            actual_dist += self.distance(ref_dic,current_dic)

        actual_dist /= len(self.categorical_variables)

        return actual_dist

    def drift_detector(self):

        """
        data : The data on which we want to detect the drift
        n_batch : number of elements per batch.  
        gamma : 
        """

        # blockPrint()

        ref_window = self.data.iloc[0:self.n_batch]
        prev_dist = 0
        lambda_= 0
        sum_eps = 0
        sum_eps_sd = 0
        change = []
        i = 1

        while self.n_batch*(i+1) <= self.data.shape[0]:

            drift = False

            current_window = self.data.iloc[self.n_batch*i:self.n_batch*(i+1)]

            actual_dist = self.windows_distance(ref_window,current_window)

            dist_diff = actual_dist - prev_dist

            print('ref_window length is %f' %ref_window.shape[0])
            print('actual dist is {}'.format(actual_dist))
            print('previous dist is {}'.format(prev_dist))
            print('dist diff is {}'.format(dist_diff))

            #we update the adaptive threshold

            epsilon_hat = sum_eps/(i-lambda_)

            sigma_hat = sqrt(sum_eps_sd/(i - lambda_))

            beta_hat = epsilon_hat + self.gamma * sigma_hat   #One method to compute beta_hat

            print('epsilon hat is {}'.format(epsilon_hat))
            print('sigma_hat is {}'.format(sigma_hat))
            print('beta_hat is {}'.format(beta_hat))

            if abs(dist_diff) > beta_hat :

                # print('drift detected')

                lambda_ = i 
                # for feature in self.categorical_variables:
                #     print(ref_window[feature].value_counts(),current_window[feature].value_counts())
                change.append([range(self.n_batch*(i) - ref_window.shape[0],self.n_batch*(i)),range(self.n_batch*i,self.n_batch*(i+1))])
                ref_window = current_window
                drift = True

            else:
                ref_window = ref_window.append(current_window)

            i +=1
            prev_dist = actual_dist
            if drift == True:
                sum_eps = abs(dist_diff)
                sum_eps_sd = (abs(dist_diff)-epsilon_hat)**2
            else : 
                sum_eps += abs(dist_diff)
                sum_eps_sd += (abs(dist_diff)-epsilon_hat)**2

            print('/')
            print('/')
            print('/')

        return change

    def drift_detector2(self):

        """
        data : The data on which we want to detect the drift
        n_batch : number of elements per batch.  
         
        """

        # blockPrint()

        ref_window = self.data.iloc[0:self.n_batch]
        prev_dist = 0
        lambda_= 0
        sum_eps = 0
        sum_eps_sd = 0
        change = []
        i = 1

        while self.n_batch*(i+1) <= self.data.shape[0]:

            drift = False

            current_window = self.data.iloc[self.n_batch*i:self.n_batch*(i+1)]

            actual_dist = self.windows_distance(ref_window,current_window)

            print('ref_window length is %f' %ref_window.shape[0])
            print('actual dist is {}'.format(actual_dist))


            if abs(actual_dist) > self.threshold :
                print('drift detected')
                change.append([range(self.n_batch*(i) - ref_window.shape[0],self.n_batch*(i)),range(self.n_batch*i,self.n_batch*(i+1))])

                ref_window = current_window

                drift = True

            else:
                ref_window = ref_window.append(current_window)

            i +=1


            print('/')
            print('/')
            print('/')

        return change


class Discretizer :

    def __init__(self,method):
        #method is either equalsize bins or equalquantile bins
        self.method = method


    def fit(self,data,names,to_ignore):
        #to_ignore should contain the categorical variables and the IDS
        columns = list(set(list(data)) - set(to_ignore))

        tmp = []
        for col in columns : 
            tmp.append(len(data[col].unique()))
        tmp = np.array(tmp)
        threshold = np.mean(tmp)

        if names is None :
            self.numerical_cols = [col for col in columns if len(data[col].unique()) > threshold and data[col].isnull().any() == False and data[col].dtype != 'object']
        else:
            self.numerical_cols = list(set(data.columns).intersection(names))
        return self

    def equalsize(self,col):

        bin_col , self.bins_output[col.name] = pd.cut(col,self.n_bins,retbins = True,duplicates = 'drop')
        return bin_col 

    def equalquantile(self,col): 

        bin_col , self.bins_output[col.name] = pd.qcut(col,self.n_bins,retbins=True,duplicates = 'drop')
        return bin_col 

    def transform(self,data,n_bins):
        self.n_bins = n_bins
        self.bins_output = {}
        if self.method == "equalsize":
            data[self.numerical_cols] = data[self.numerical_cols].apply(self.equalsize, axis = 0) #apply function to each column
        if self.method == "equalquantile":
            data[self.numerical_cols] = data[self.numerical_cols].apply(self.equalquantile, axis = 0)
        
        return data , self.bins_output



def process_bin(data,numerical_cols):

    """
    Transforms values (intervals) of numerical_cols to integers because classifiers have problems to deal with names that contain [ , ] , < 
    """

    for feature in numerical_cols : 
        data[feature] = data[feature].apply(str)
        n = len(data[feature].unique())
        dic = {}
        for i in range(n):
            dic.update({data[feature].unique()[i]:i})
        data[feature] = data[feature].map(dic)
    return data

from random import choices

def cross_validation(data,target,n_batch,indexes,clf):
    """
    p_train: the percentage of the training set
    """
    n = len(indexes)

    precision_cv = []
    recall_cv = []
    f1_score_cv = []

    perc_train_cv = []
    perc_test_cv = []

    precision = []
    recall = []
    f1_score = []

    perc_train = []
    perc_test = []


    for i in range(int(n/n_batch)-1): 

        # train_indexes = indexes[:(i+1)*int(p_test*n)]
        # test_indexes = indexes[(i+1)*int(p_test*n):(i+2)*int(p_test*n)]

        train_indexes = indexes[:(i+1)*int(n_batch)]
        test_indexes = indexes[(i+1)*int(n_batch):(i+2)*int(n_batch)]

        #Now we will perform bootstraping
        for b in range(10):

            test_indexes_ = choices(test_indexes, k = len(test_indexes))

            X_train = data.iloc[train_indexes]
            y_train = target.iloc[train_indexes]

            try : 
                n_errors_train = y_train.value_counts().iloc[1]
            except :
                n_errors_train = 0

            if n_errors_train == 0 :
                continue

            X_test = data.iloc[test_indexes_]
            y_test = target.iloc[test_indexes_]

            clf.fit(X_train,y_train)

            output = clf.predict(X_test)

            perc_train_cv.append(n_errors_train/y_train.shape[0])
        
            try:
                perc_test_cv.append(y_test.value_counts().iloc[1]/y_test.shape[0])
            except: 
                perc_test_cv.append(0)

            precision_cv.append(precision_score(y_test,output))
            
            recall_cv.append(recall_score(y_test,output))

            f1_score_cv.append(metrics.f1_score(y_test,output))

        precision.append(np.mean(precision_cv))
        recall.append(np.mean(recall_cv))
        f1_score.append(np.mean(f1_score_cv))

        perc_train.append(np.mean(perc_train_cv))
        perc_test.append(np.mean(perc_test_cv))

    return precision,recall,f1_score,perc_train,perc_test


def predict_out(data,target,train_indexes,test_indexes,clf):

    precision = []
    recall = []
    f1_score = []

    percentage_out_train = []
    percentage_out_test = []

    for b in range(10) : 

        test_indexes_ = choices(test_indexes, k = len(test_indexes))

        clf.fit(data.iloc[train_indexes],target.iloc[train_indexes])

        output = clf.predict(data.iloc[test_indexes_])

        target_out_train = target.iloc[train_indexes]

        target_out_test = target.iloc[test_indexes_]

        try:
            percentage_out_train.append(target_out_train.value_counts().iloc[1]/target_out_train.shape[0])
        except:
            percentage_out_train.append(0)

        try:
            percentage_out_test.append(target_out_test.value_counts().iloc[1]/target_out_test.shape[0])
        except:
            percentage_out_test.append(0)

        precision.append(precision_score(target.iloc[test_indexes_],output))
        recall.append(recall_score(target.iloc[test_indexes_],output))
        f1_score.append(metrics.f1_score(target.iloc[test_indexes_],output))

    return precision,recall,f1_score,percentage_out_train,percentage_out_test


import itertools
import xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score,f1_score
from scipy import stats
from scipy.stats import normaltest
from scipy.stats import boxcox


def final_func(data,target,n_batch,n_bins,threshold,to_keep,dummy,categorical_variables,distance):
        
        data2 = data.copy() 

        Discretize = Discretizer("equalquantile")

        Discretize.fit(data, None , to_ignore = ['Trade date','Trade id'] + categorical_variables)

        numerical_cols = Discretize.numerical_cols

        binned_data , bins_output = Discretize.transform(data,n_bins)

        if to_keep == "all" : 
            Detector = Drift_Detector(binned_data,n_batch,categorical_variables + numerical_cols,None,threshold,distance)

        else:
            Detector = Drift_Detector(binned_data,n_batch,to_keep,None,threshold,distance)

        result = Detector.drift_detector2()

        print(result)

        # binned_data_1 = process_bin(binned_data,numerical_cols)

        # binned_data_2 = binned_data_1[categorical_variables + numerical_cols]

        # from sklearn import preprocessing

        # tmp = binned_data_1[numerical_cols]
        # tmp = tmp.apply(preprocessing.LabelEncoder().fit_transform)
        # binned_data_1[numerical_cols] = tmp

        if to_keep == "all": 
            to_keep = list(binned_data)

        else:  
            categorical_variables = list(set(categorical_variables).intersection(set(to_keep)))
            numerical_cols = list(set(numerical_cols).intersection(set(to_keep)))

        binned_cat = pd.get_dummies(data2[numerical_cols + categorical_variables] , columns = categorical_variables)

        # clf = XGBClassifier(n_estimators = 200 , learning_rate = 0.05, seed = 42)
        # clf = GradientBoostingClassifier(n_estimators = 200, learning_rate= 0.05, random_state = 42)
        clf = RandomForestClassifier(n_estimators = 50, random_state = 42)
        
        precision_within,recall_within,f1_score_within = [],[],[]

        precision_out,recall_out,f1_score_out = [],[],[]

        percentage_within_train = []
        percentage_within_test = []

        percentage_out_train = []
        percentage_out_test = []

        for i in range(len(result)):

            if len(result[i][0]) == n_batch :
                continue

            # 'Within' tests
            result_within = cross_validation(binned_cat,target,n_batch,result[i][0],clf)

            if result_within[0] != 0 and result_within[1] != 0:
                precision_within.extend(result_within[0])
                recall_within.extend(result_within[1])
                f1_score_within.extend(result_within[2])

            percentage_within_train.extend(result_within[3])
            percentage_within_test.extend(result_within[4])

            # 'Out' tests
            result_out = predict_out(binned_cat,target,result[i][0],result[i][1],clf)

            if result_out[0] != 0 and result_out[1] != 0:
                precision_out.extend(result_out[0])
                recall_out.extend(result_out[1])
                f1_score_out.extend(result_out[2])

            percentage_out_train.extend(result_out[3])
            percentage_out_test.extend(result_out[4])

        # blockPrint()
        print('/')
        print('/')
        print('/')

        print('the percentage_within_train is \n {}'.format(percentage_within_train))
        print('the percentage_within_test is \n {}'.format(percentage_within_test))

        # print('the precision within the domain is \n {}'.format(precision_within))
        # print('the recall within the domain is \n {}'.format(recall_within))
        print('the f1_score within the domain is \n {}'.format(f1_score_within))

        # print('the mean precision within the domain is {}'.format(np.mean(precision_within)))
        # print('the mean recall within the domain is {}'.format(np.mean(recall_within)))
        print('the mean f1_score within the domain is {}'.format(np.mean(f1_score_within)))

        print('/')
        print('/')
        print('/')

        print('the percentage_out_train is \n {}'.format(percentage_out_train))
        print('the percentage_out_test is \n {}'.format(percentage_out_test))

        # print('the precision out of the domain is \n {}'.format(precision_out))
        # print('the recall out of the domain is \n {}'.format(recall_out))
        print('the f1_score out of the domain is \n {}'.format(f1_score_out))

        # print('the mean precision out of the domain is {}'.format(np.mean(precision_out)))
        # print('the mean recall out of the domain is {}'.format(np.mean(recall_out)))
        print('the mean f1_score out of the domain is {}'.format(np.mean(f1_score_out)))

        print('/')
        print('/')
        print('/')

        
        t_test = stats.ttest_ind(f1_score_within,f1_score_out)
        enablePrint()

        #We will check for normality of data

        normal_test_in = stats.normaltest(f1_score_within,nan_policy = 'omit')
        print(n_batch, n_bins, normal_test_in)

        normal_test_out = stats.normaltest(f1_score_out,nan_policy = 'omit')
        print(n_batch, n_bins, normal_test_out)

        if normal_test_in[1] > 0.05 and normal_test_out[1] > 0.05:
            t_test = stats.ttest_ind(f1_score_within,f1_score_out)
            print(t_test)

        else:
            tmp_in = stats.boxcox(f1_score_within)[0]
            tmp_out = stats.boxcox(f1_score_out)[0]
            t_test = stats.ttest_ind(tmp_in,tmp_out)
            print(t_test)


        # res_box = stats.normaltest(tmp,nan_policy = 'omit')

        # print('normality test after boxcox for within f1_score for the combination {},{} is {}'.format(n_batch,n_bins,res_box))

        # tmp = stats.boxcox(f1_score_out)

        # res_box = stats.normaltest(tmp,nan_policy = 'omit')

        # print('normality test after boxcox for out f1_score for the combination {},{} is {}'.format(n_batch,n_bins,res_box))

        print('loop done')

        return to_keep,n_batch,threshold,n_bins,round(np.mean(f1_score_within),2),round(np.mean(f1_score_out),2),round(np.mean(percentage_within_train),2),round(np.mean(percentage_within_test),2),\
                round(np.mean(percentage_out_train),2),round(np.mean(percentage_out_test),2)

def process_features(data):

    data['Buy/Sell'] = data['Quantity'].apply(lambda x: 1 if x > 0 else 0)  
    for col in ['Price' , 'Amount' , 'Quantity']:
        data[col] = data[col].apply(lambda x: abs(x))
    return data


# print('the precision_out_total is \n {}'.format(precision_out_total))
# print('the recall_out_total is \n {}'.format(recall_out_total))

# print('the precision_within_total is \n {}'.format(precision_within_total))
# print('the recall_within_total is \n {}'.format(recall_within_total))

# print('the precision_within_out_total is \n {}'.format(precision_within_out_total))
# print('the recall_within_out_total is \n {}'.format(recall_within_out_total))

# print('the results list is \n {}'.format(results))

def write_lines(file,liste_elem):
    for elem in liste_elem:
        file.write('{}{}'.format(elem,';'))
    file.write("\n")


from functools import partial


def main():

    try:
        import multiprocessing 

        pool = multiprocessing.Pool(processes = multiprocessing.cpu_count() - 1)
        
        list_n_batch = [2000]

        list_threshold = [0.2]

        list_quantile = [10]    #100/q % of the data in each quantile

        # data = pd.read_csv('Datasets/options_natixis.csv',sep = ';',encoding = "cp1252",low_memory = False,na_values = "null")
        data = pd.read_csv('Datasets/options_ing.csv')

        data = data.fillna(-1,axis=1)

        data = data.sort_values('Trade date')

        target = data['ERROR']

        data = data.drop('ERROR',axis=1)

        data = process_features(data)

        categorical_variables = ['Payment currency','Counterparty','Entity','Buy/Sell','Quotite','Instrument classification','Calculation agent']

        to_keep = categorical_variables
        # to_keep = "all"

        output = []

        distance = Distance().hellinger_dist

        final_func_ = partial(final_func, data, target, to_keep = to_keep , dummy = True,categorical_variables = categorical_variables, distance = distance)

        output.append(pool.starmap(final_func_,itertools.product(list_n_batch,list_quantile,list_threshold)))


        with open('output.csv',"w") as output_file:
            write_lines(output_file,['to_keep','n_batch','threshold','n_bins','mean f1_score within','mean f1_score out','p_value','mean percentage within train','mean percetange within test','mean percentage out train',\
                'mean percentage within test'])
            
            for row in output:
                write_lines(output_file,row)
                write_lines(output_file,[])
    except:

        import pdb, traceback, sys
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

    

if __name__ == "__main__":
    main()






# dic={}
# for feature in categorical_variables+numerical_cols:
#     dic.update({feature:len(binned_data[feature].unique())})
# print(dic)

























