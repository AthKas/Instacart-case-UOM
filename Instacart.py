#!/usr/bin/env python
# coding: utf-8



# For data manipulation
import pandas as pd  
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
# Garbage Collector to free up memory
import gc                         
gc.enable()                       # Activate 



#install kaggleApi
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi({"username":"athinakasimidou","key":"77233655dde27b7c0e8c1e23587d96af"})
api.authenticate()
files = api.competition_download_files('instacart-market-basket-analysis')



#install Zipfile
import zipfile
with zipfile.ZipFile('instacart-market-basket-analysis.zip', 'r') as zip_ref:
    zip_ref.extractall('./input')


#install os
import os
working_directory = os.getcwd()+'/input'
os.chdir(working_directory)
for file in os.listdir(working_directory):   # get the list of files
    if zipfile.is_zipfile(file): # if it is a zipfile, extract it
        with zipfile.ZipFile(file) as item: # treat the file as a zip
           item.extractall()  # extract it in the working directory


# In[5]:


#read csv
orders = pd.read_csv('../input/orders.csv')
order_products_train = pd.read_csv('../input/order_products__train.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
products = pd.read_csv('../input/products.csv')
aisles = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')



# We convert character variables into category. 
# In Python, a categorical variable is called category and has a fixed number of different values
aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')



#Merge the orders DF with order_products_prior by their order_id, keep only these rows with order_id that they are appear on both DFs
op = orders.merge(order_products_prior, on='order_id', how='inner')
op.head()


#Create the arrays of last five orders for each user
op['order_number_back'] = op.groupby('user_id')['order_number'].transform(max) - op.order_number +1
op.head(15)



op5 = op[op.order_number_back <= 5]
op5.head()



#CREATE FEATURES
#Create distinct groups for each user, save the results to the user array
user = op.groupby('user_id')['order_number'].max().to_frame('user_t_orders') #
user = user.reset_index()
user.head()




#Create u_reorder_ratio
u_reorder = op.groupby('user_id')['reordered'].mean().to_frame('u_reordered_ratio') #
u_reorder = u_reorder.reset_index()
u_reorder.head()



#Create u_reordered_ratio_5
u_reorder5 = op5.groupby('user_id')['reordered'].mean().to_frame('u_reordered_ratio_5') #
u_reorder5 = u_reorder5.reset_index()
u_reorder5.head()



#Merge features to user array
user=user.merge(u_reorder,on='user_id',how='left')#
del u_reorder
gc.collect()

user.head()



#Merge features to user array
user=user.merge(u_reorder5,on='user_id',how='left')#
del u_reorder5
gc.collect()

user.head()





#Create mean_days
days = op.groupby('user_id')['days_since_prior_order'].mean().to_frame('mean_days')
days = days.reset_index()
days.head()



#Create mean_days_5
days5 = op5.groupby('user_id')['days_since_prior_order'].mean().to_frame('mean_days_5')
days5 = days5.reset_index()
days5.head()



#Merge features to user array
user = user.merge(days, on='user_id', how='left')
del days
gc.collect()
user.head()



#Merge features to user array
user = user.merge(days5, on='user_id', how='left')
del days5
gc.collect()
user.head()



#Create max_basket
user_max = op.groupby(['user_id','order_id'])['add_to_cart_order'].max().to_frame('max_basket')
user_max.head()


#Create mean_basket
user_max_ratio = user_max.groupby('user_id')['max_basket'].mean().to_frame('mean_basket')
user_max_ratio = user_max_ratio.reset_index()
user_max_ratio.head()



#Merge features to user array
user = user.merge(user_max_ratio, on='user_id', how='left')
del user_max_ratio
gc.collect()
user.head()



#Create max_basket5
user_max5 = op5.groupby(['user_id','order_id'])['add_to_cart_order'].max().to_frame('max_basket5')
user_max5.head()




#Create mean_basket5
user_max_ratio5 = user_max5.groupby('user_id')['max_basket5'].mean().to_frame('mean_basket5')
user_max_ratio5 = user_max_ratio5.reset_index()
user_max_ratio5.head()




#Merge features to user array
user = user.merge(user_max_ratio5, on='user_id', how='left')
del user_max_ratio5
gc.collect()
user.head()


#Detele non useful features from user array
del  user_max, user_max5
gc.collect()
user.head()



# Create distinct groups for each product and save the result for each product to the products array
prd = op.groupby('product_id')['order_id'].count().to_frame('prd_t_purchases') #
prd = prd.reset_index()
prd.head()



#Create prd_t_purchases5
prd5 = op5.groupby('product_id')['order_id'].count().to_frame('prd_t_purchases5') #
prd5 = prd5.reset_index()
prd5.head()


#Merge features to the product array
prd = prd.merge(prd5, on='product_id', how='left')
del prd5
gc.collect()
prd['prd_t_purchases5'] = prd['prd_t_purchases5'].fillna(0)
prd.head()



p_reorder = op.groupby('product_id').filter(lambda x: x.shape[0] >40)#####
p_reorder.head()



#Create p_reorder_ratio
p_reorder = op.groupby('product_id')['reordered'].mean().to_frame('p_reorder_ratio')
p_reorder = p_reorder.reset_index()
p_reorder.head()


#Merge features to the product array
prd = prd.merge(p_reorder, on='product_id', how='left')
del p_reorder
gc.collect()
prd.head()



#Fill NaN with 0
prd['p_reorder_ratio'] = prd['p_reorder_ratio'].fillna(0) #
prd.head()



#Create p_reorder_ratio5
p_reorder5 = op5.groupby('product_id')['reordered'].mean().to_frame('p_reorder_ratio5')
p_reorder5 = p_reorder5.reset_index()
p_reorder5.head()



#Merge features to the product array
prd = prd.merge(p_reorder5, on='product_id', how='left')
del p_reorder5
gc.collect()
prd['p_reorder_ratio5'] = prd['p_reorder_ratio5'].fillna(0)
prd.head()


#Create aop_mean
aop = op.groupby('product_id')['add_to_cart_order'].mean().to_frame("aop_mean")
aop = aop.reset_index()
aop.head()


#Merge features to the product array
prd = prd.merge(aop, on='product_id', how='left')
del aop
gc.collect()
prd.head()



#Create aop_mean5
aop5 = op5.groupby('product_id')['add_to_cart_order'].mean().to_frame("aop_mean5")
aop5 = aop5.reset_index()
aop5.head()



#Merge features to the product array
prd = prd.merge(aop5, on='product_id', how='left')
del aop5
gc.collect()
prd.head()



# Create distinct groups for each combination of user and product, count orders, save the result for each user X product to a new DataFrame 
uxp = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_t_bought') #
uxp = uxp.reset_index()
uxp.head()


#Create Times_Bought_N
times = op.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
times.head()


#Create total_orders
total_orders = op.groupby('user_id')['order_number'].max().to_frame('total_orders') #
total_orders.head()



#Create first_order_number
first_order_no = op.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')
first_order_no  = first_order_no.reset_index()
first_order_no.head()




#Merge features to the user x product array
span = pd.merge(total_orders, first_order_no, on='user_id', how='right')
span.head()


# The +1 includes in the difference the first order were the product has been purchased
span['Order_Range_D'] = span.total_orders - span.first_order_number + 1
span.head()



#Merge features to the user x product array
uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
uxp_ratio.head()



#Remove temporary DataFrames
del [times, first_order_no, span]



#Create uxp_reorder_ratio
uxp_ratio['uxp_reorder_ratio'] = uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D ##
uxp_ratio.head()



#Merge features to the user x product array
uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')
del uxp_ratio
uxp.head()


#Delete non useful features
uxp = uxp.drop(['total_orders', 'first_order_number', 'Order_Range_D', 'Times_Bought_N'], axis=1)
uxp.head()

#Create times_last5
last_five = op5.groupby(['user_id','product_id'])['order_id'].count().to_frame('times_last5')
last_five.head(10)

#Merge features to the user x product array
uxp = uxp.merge(last_five, on=['user_id', 'product_id'], how='left')
uxp['times_last5'] = uxp['times_last5'].fillna(0)
del last_five
uxp.head()

#Create uxp_aop
uxp_aop= op.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('uxp_aop')
uxp_aop.head()

#Create uxp_aop5
uxp_aop5 = op5.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('uxp_aop_5')
uxp_aop5.head()

#Merge features to the user x product array
uxp = uxp.merge(uxp_aop, on=['user_id', 'product_id'], how='left')
del uxp_aop
uxp['uxp_aop'] = uxp['uxp_aop'].fillna(0)
uxp.head()

#Merge features to the user x product array
uxp = uxp.merge(uxp_aop5, on=['user_id', 'product_id'], how='left')
del uxp_aop5
uxp['uxp_aop_5'] = uxp['uxp_aop_5'].fillna(0)
uxp.head()

#Create uxp_dow
uxp_dow= op.groupby(['user_id', 'product_id'])['order_dow'].mean().to_frame('uxp_dow')
uxp_dow.head()
#Create uxp_dow5
uxp_dow5= op5.groupby(['user_id', 'product_id'])['order_dow'].mean().to_frame('uxp_dow5')
uxp_dow5.head()

#Create uxp_hour
uxp_hour= op.groupby(['user_id', 'product_id'])['order_hour_of_day'].mean().to_frame('uxp_hour')
uxp_hour.head()
#Create uxp_hour5
uxp_hour5= op5.groupby(['user_id', 'product_id'])['order_hour_of_day'].mean().to_frame('uxp_hour5')
uxp_hour5.head()

#Merge features to the user x product array
uxp = uxp.merge(uxp_dow, on=['user_id', 'product_id'], how='left')
del uxp_dow
uxp['uxp_dow'] = uxp['uxp_dow'].fillna(0)
uxp.head()

#Merge features to the user x product array
uxp = uxp.merge(uxp_dow5, on=['user_id', 'product_id'], how='left')
del uxp_dow5
uxp['uxp_dow5'] = uxp['uxp_dow5'].fillna(0)
uxp.head()

#Merge features to the user x product array
uxp = uxp.merge(uxp_hour, on=['user_id', 'product_id'], how='left')
del uxp_hour
uxp['uxp_hour'] = uxp['uxp_hour'].fillna(0)
uxp.head()

#Merge features to the user x product array
uxp = uxp.merge(uxp_hour5, on=['user_id', 'product_id'], how='left')
del uxp_hour5
uxp['uxp_hour5'] = uxp['uxp_hour5'].fillna(0)
uxp.head()
#Remove temporary DataFrames
del op, op5
gc.collect()



#Merge uxp features with the user features
#Store the results on a new DataFrame
data = uxp.merge(user, on='user_id', how='left')
data.head()



#Remove temporary DataFrames
del uxp, user
gc.collect()



#Merge uxp & user features (the new DataFrame) with prd features
data = data.merge(prd, on='product_id', how='left') #
data.head()



#Remove temporary DataFrames
del prd
gc.collect()



## First approach:
# In two steps keep only the future orders from all customers: train & test 
orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]
orders_future.head(10)

## Second approach (if you want to test it you have to re-run the notebook):
# In one step keep only the future orders from all customers: train & test 
#orders_future = orders.loc[((orders.eval_set=='train') | (orders.eval_set=='test')), ['user_id', 'eval_set', 'order_id'] ]
#orders_future.head(10)

## Third approach (if you want to test it you have to re-run the notebook):
# In one step exclude all the prior orders so to deal with the future orders from all customers
#orders_future = orders.loc[orders.eval_set!='prior', ['user_id', 'eval_set', 'order_id'] ]
#orders_future.head(10)



# bring the info of the future orders to data DF
data = data.merge(orders_future, on='user_id', how='left')
data.head(10)


#Keep only the customers who we know what they bought in their future order
data_train = data[data.eval_set=='train'] #
data_train.head()



#Get from order_products_train all the products that the train users bought bought in their future order
data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )
data_train.head(15)



#Where the previous merge, left a NaN value on reordered column means that the customers they haven't bought the product. We change the value on them to 0.
data_train['reordered'] = data_train['reordered'].fillna(0)
data_train.head(15)




#We set user_id and product_id as the index of the DF
data_train = data_train.set_index(['user_id', 'product_id'])
data_train.head(15)



#We remove all non-predictor variables
data_train = data_train.drop(['eval_set', 'order_id'], axis=1)
data_train.head(15)



#Keep only the future orders from customers who are labelled as test
data_test = data[data.eval_set=='test'] #
data_test.head()


#We set user_id and product_id as the index of the DF
data_test = data_test.set_index(['user_id', 'product_id']) #
data_test.head()



#We remove all non-predictor variables
data_test = data_test.drop(['eval_set','order_id'], axis=1)
#Check if the data_test DF, has the same number of columns as the data_train DF, excluding the response variable
data_test.head()


## SPLIT DF TO: X_train, y_train (axis=1)
##########################################
X_train, y_train = data_train.drop('reordered', axis=1), data_train.reordered

# CREATE MODEL
###########################
## DISABLE WARNINGS
###########################
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
####################################
## SET BOOSTER'S RANGE OF PARAMETERS
# IMPORTANT NOTICE: Fine-tuning an XGBoost model may be a computational prohibitive process with a regular computer or a Kaggle kernel. 
# Be cautious what parameters you enter in paramiGrid section.
# More paremeters means that GridSearch will create and evaluate more models.
####################################    
paramGrid = {"max_depth":[9],
            "colsample_bytree":[0.6],
            "subsample":[0.7],
            "lambda": [0.95],
            "min_child_weight": [0.7],
            "eta": [0.074, 0.075, 0.076],
            "gamma": [6],
            }  

########################################
## INSTANTIATE XGBClassifier()
########################################
xgbc = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', num_boost_round=500, gpu_id=0, tree_method = 'gpu_hist')

##############################################
## DEFINE HOW TO TRAIN THE DIFFERENT MODELS
#############################################
gridsearch = GridSearchCV(xgbc, paramGrid, cv=5, verbose=2, n_jobs=1)

################################################################
## TRAIN THE MODELS
### - with the combinations of different parameters
### - here is where GridSearch will be exeucuted
#################################################################
model = gridsearch.fit(X_train, y_train)

##################################
## OUTPUT(S)
##################################
# Print the best parameters
print("The best parameters are: /n",  gridsearch.best_params_)

# Store the model for prediction
model = gridsearch.best_estimator_


##########################################
#TRAIN MODEL

D_train = xgb.DMatrix(data=X_train, label = y_train)
D_test = xgb.DMatrix(data=data_test)
########################################
## SET BOOSTER'S PARAMETERS
########################################
parameters = {"objective":'binary:logistic',
    'eval_metric':'logloss', 
              "max_depth":9,
            "colsample_bytree":0.6,
            "subsample":0.7,
            "lambda": 0.95,
            "min_child_weight": 0.7,
            "eta": 0.075,
            "gamma": 6,
               "gpu_id":0,
               "tree_method": 'gpu_hist'
             }
########################################
########################################
model = xgb.train(params = parameters, dtrain = D_train, num_boost_round = 500)
#xgb.plot_importance(model)


## OR set a custom threshold (in this problem, 0.21 yields the best prediction)
test_pred = (model.predict(D_test) >= 0.21)
test_pred[0:20] #display the first 20 predictions of the numpy array



#Save the prediction in a new column in the data_test DF
data_test['prediction'] = test_pred
data_test.head()



#Reset the index
final = data_test.reset_index()
#Keep only the required columns to create our submission file (Chapter 6)
final = final[['product_id', 'user_id', 'prediction']]

gc.collect()
final.head()
# Delete X_train , y_train
del [X_train, y_train]


#
orders_test = orders.loc[orders.eval_set=='test',("user_id", "order_id") ]
orders_test.head()


final = final.merge(orders_test, on='user_id', how='left')
final.head()


# In[86]:


#remove user_id column
final = final.drop('user_id', axis=1)
#convert product_id as integer
final['product_id'] = final.product_id.astype(int)

#Remove all unnecessary objects
del orders
del orders_test
gc.collect()

final.head()


d = dict()
for row in final.itertuples():
    if row.prediction== 1:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in final.order_id:
    if order not in d:
        d[order] = 'None'
        
gc.collect()

#We now check how the dictionary were populated (open hidden output)
d



#Convert the dictionary into a DataFrame
sub = pd.DataFrame.from_dict(d, orient='index')

#Reset index
sub.reset_index(inplace=True)
#Set column names
sub.columns = ['order_id', 'products']

sub.head()



#Check if sub file has 75000 predictions
sub.shape[0]
print(sub.shape[0]==75000)


#get csv
sub.to_csv('sub.csv', index=False)

