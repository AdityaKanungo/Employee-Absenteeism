
# coding: utf-8

# # Employee Absenteeism

# In[1]:


#import useful libraries 
import pandas as pd
import numpy as np
import os
from scipy.stats import chi2_contingency
#libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[182]:


#set current working directory
os.chdir('C:/Users/BATMAN/Desktop/project 2 edwisor')


# In[312]:


#import data for analysis
df = pd.read_excel('Absenteeism_at_work_Project.xls')
df1 = df


# ### Exlopratory Data Analysis (EDA)

# In[313]:


df.head()


# In[314]:


#Numeric and categorical variables
num_var = ['ID','Transportation expense','Service time','Age','Hit target','Absenteeism time in hours']
cat_names = ['Reason for absence','Day of the week','Seasons','Month of absence','Distance from Residence to Work','Work load Average/day ','Education','Son','Pet','Disciplinary failure','Social drinker','Social smoker','Height','Weight','Body mass index'] 


# In[315]:


var = ['ID','Reason for absence','Month of absence','Day of the week','Seasons','Transportation expense','Distance from Residence to Work','Service time','Age','Work load Average/day ','Hit target','Disciplinary failure','Education','Son','Social drinker','Social smoker','Pet','Weight','Height','Body mass index','Absenteeism time in hours']
for i in var:
    print(i)
    print(df[i].nunique())


# ### Exploratory Data Analysis

# ### - General Feature study & Visualizations

# In[7]:


#general study of total hours of absenteeism by features with binary category.
df1 =  (df[['Disciplinary failure', 'Absenteeism time in hours']].groupby(['Disciplinary failure'], as_index=False).mean().sort_values(by='Absenteeism time in hours', ascending=False))
df2 =  (df[['Social drinker', 'Absenteeism time in hours']].groupby(['Social drinker'], as_index=False).mean().sort_values(by='Absenteeism time in hours', ascending=False))
df3 =  (df[['Social smoker', 'Absenteeism time in hours']].groupby(['Social smoker'], as_index=False).mean().sort_values(by='Absenteeism time in hours', ascending=False))
print(df1)
print(df2)
print(df3)
pd.DataFrame({'yes': [120, 3108, 463], 'No': [4862, 1861, 4511]},index=['Disciplinary failure', 'Social drinker', 'Social smoker'])


# In[26]:


#This plot represents the total hours of absenteeism by features with binary category.
import plotly
plotly.tools.set_credentials_file(username='aditya.kanungo', api_key='AcSRoPu5SZ6lYsKGSLup')
import plotly.plotly as py
import plotly.graph_objs as go

trace1 = go.Bar(
    x=['Disciplinary failure', 'Social drinker', 'Social smoker'],
    y=[120, 3108, 463],
    name='Yes'
)
trace2 = go.Bar(
    x=['Disciplinary failure', 'Social drinker', 'Social smoker'],
    y=[4862, 1861, 4511],
    name='No'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# In[9]:


df_Son =pd.DataFrame(df[['Son', 'Absenteeism time in hours']].groupby(['Son'], as_index=False).mean().sort_values(by='Absenteeism time in hours', ascending=False))
objects = df_Son['Son']
y_pos = np.arange(len(objects))
performance = df_Son['Absenteeism time in hours']
plt.figure(figsize=(15,5))
plt.bar(y_pos, performance, align='center', color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Mean absenteeism with no. of son')
plt.show()


# In[10]:


#SOCIAL DRINKER
df_drink =pd.DataFrame(df[['Social drinker', 'Absenteeism time in hours']].groupby(['Social drinker'], as_index=False).count().sort_values(by='Absenteeism time in hours', ascending=False))
objects = df_drink['Social drinker']
y_pos = np.arange(len(objects))
performance = df_drink['Absenteeism time in hours']
plt.bar(y_pos, performance, align='center', color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('No of absenteeism in social drinkers')
plt.show()


# ###### From the above plot we can see that -
# - The number of absenteeism in social drinkers is more as compared to non-drinkers

# In[11]:


#SOCIAL SMOKER
df_smoke =pd.DataFrame(df[['Social smoker', 'Absenteeism time in hours']].groupby(['Social smoker'], as_index=False).mean().sort_values(by='Absenteeism time in hours', ascending=False))
df_smoke['Social smoker'] = df_smoke['Social smoker'].astype(str)
df_smoke['Social smoker'] = df_smoke['Social smoker'].replace('1','yes')
df_smoke['Social smoker'] = df_smoke['Social smoker'].replace('0','No')
objects = df_smoke['Social smoker']
y_pos = np.arange(len(objects))
performance = df_smoke['Absenteeism time in hours']
plt.bar(y_pos, performance, align='center', color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('No of absenteeism in social smoker')
plt.show()


# ###### From the above plot we can see that -
# - The avarage hours of absenteeism in social smokers is slightly more than non-smokers.

# In[12]:


#Medical Reason for absence
df_Reasonforabsence =pd.DataFrame(df[['Reason for absence', 'Absenteeism time in hours']].groupby(['Reason for absence'], as_index=False).count().sort_values(by='Absenteeism time in hours', ascending=False))
# 1 = Social smoker, 0 = Not a social smoker
objects = df_Reasonforabsence['Reason for absence']
y_pos = np.arange(len(objects))
performance = df_Reasonforabsence['Absenteeism time in hours']
plt.figure(figsize=(15,5))
plt.bar(y_pos, performance, align='center', color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Medical Reason wise absenteeism')
plt.show()


# ###### From the above plot we can see that ICD - 23,28,27 are the leading cause of absenteeism hours. 
# - Medical consultation 
# - Dental consultation  
# - Physiotherapy

# In[13]:


#Month for absence
df_Monthofabsence =pd.DataFrame(df[['Month of absence', 'Absenteeism time in hours']].groupby(['Month of absence'], as_index=False).count().sort_values(by='Absenteeism time in hours', ascending=False))
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].astype(str)
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('0.0','Unknown')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('1.0','Jan')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('2.0','Feb')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('3.0','Mar')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('4.0','Apr')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('5.0','May')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('6.0','Jun')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('7.0','Jul')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('8.0','Aug')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('9.0','Sep')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('10.0','Oct')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('11.0','Nov')
df_Monthofabsence['Month of absence'] = df_Monthofabsence['Month of absence'].replace('12.0','Dec')
objects = df_Monthofabsence['Month of absence']
y_pos = np.arange(len(objects))
performance = df_Monthofabsence['Absenteeism time in hours']
plt.figure(figsize=(15,5))
plt.bar(y_pos, performance, align='center', color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Month wise absenteeism')
plt.show()


# ###### From the above plot we can see that:
# - Abesnteeism was almost same in all the months, with slightly more in march.

# In[14]:


#Day for absence
df_Dayoftheweek =pd.DataFrame(df[['Day of the week', 'Absenteeism time in hours']].groupby(['Day of the week'], as_index=False).count().sort_values(by='Absenteeism time in hours', ascending=False))
df_Dayoftheweek['Day of the week'] = df_Dayoftheweek['Day of the week'].astype(str)
df_Dayoftheweek['Day of the week'] = df_Dayoftheweek['Day of the week'].replace('2','Monday')
df_Dayoftheweek['Day of the week'] = df_Dayoftheweek['Day of the week'].replace('3','Tuesday')
df_Dayoftheweek['Day of the week'] = df_Dayoftheweek['Day of the week'].replace('4','Wednesday')
df_Dayoftheweek['Day of the week'] = df_Dayoftheweek['Day of the week'].replace('5','Thursday')
df_Dayoftheweek['Day of the week'] = df_Dayoftheweek['Day of the week'].replace('6','Friday')
objects = df_Dayoftheweek['Day of the week']
y_pos = np.arange(len(objects))
performance = df_Dayoftheweek['Absenteeism time in hours']
plt.figure(figsize=(15,5))
plt.bar(y_pos, performance, align='center', color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Day wise absenteeism')
plt.show()


# ###### From the above plot we can see that :
# - Absenteeism was not significant on any perticular day, was almost same.

# In[15]:


#Reason for absence
df_Seasons =pd.DataFrame(df[['Seasons', 'Absenteeism time in hours']].groupby(['Seasons'], as_index=False).count().sort_values(by='Absenteeism time in hours', ascending=False))
df_Seasons['Seasons'] = df_Seasons['Seasons'].astype(str)
df_Seasons['Seasons'] = df_Seasons['Seasons'].replace('1','Summer')
df_Seasons['Seasons'] = df_Seasons['Seasons'].replace('2','Autmn')
df_Seasons['Seasons'] = df_Seasons['Seasons'].replace('3','Winter')
df_Seasons['Seasons'] = df_Seasons['Seasons'].replace('4','Spring')
objects = df_Seasons['Seasons']
y_pos = np.arange(len(objects))
performance = df_Seasons['Absenteeism time in hours']
plt.figure(figsize=(15,5))
plt.bar(y_pos, performance, align='center', color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Season wise absenteeism')
plt.show()


# ###### From the above plot we can see that :
# - Absenteeism was almost same in all seasons, was not significant in any perticular month

# In[16]:


#Deciplinary Failure
df_Disciplinaryfailure =pd.DataFrame(df[['Disciplinary failure', 'Absenteeism time in hours']].groupby(['Disciplinary failure'], as_index=False).mean().sort_values(by='Absenteeism time in hours', ascending=False))
df_Disciplinaryfailure['Disciplinary failure'] = df_Disciplinaryfailure['Disciplinary failure'].astype(str)
df_Disciplinaryfailure['Disciplinary failure'] = df_Disciplinaryfailure['Disciplinary failure'].replace('1.0','Yes')
df_Disciplinaryfailure['Disciplinary failure'] = df_Disciplinaryfailure['Disciplinary failure'].replace('0.0','No')
objects = df_Disciplinaryfailure['Disciplinary failure']
y_pos = np.arange(len(objects))
performance = df_Disciplinaryfailure['Absenteeism time in hours']
plt.figure(figsize=(5,5))
plt.bar(y_pos, performance, align='center', color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Averag absenteeism in Deciplinary failure')
plt.show()


# In[17]:


# Average Absenteeism in each category of Education
df_Education =pd.DataFrame(df[['Education', 'Absenteeism time in hours']].groupby(['Education'], as_index=False).mean().sort_values(by='Absenteeism time in hours', ascending=False))
df_Education['Education'] = df_Education['Education'].astype(str)
df_Education['Education'] = df_Education['Education'].replace('1.0','High school')
df_Education['Education'] = df_Education['Education'].replace('2.0','Graduate')
df_Education['Education'] = df_Education['Education'].replace('3.0','postgraduate')
df_Education['Education'] = df_Education['Education'].replace('4.0','masters/Doctor')
objects = df_Education['Education']
y_pos = np.arange(len(objects))
performance = df_Education['Absenteeism time in hours']
plt.figure(figsize=(15,5))
plt.bar(y_pos, performance, align='center', color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Average education wise Absenteeism')
plt.show()


# ###### From the above plot we can see that -
# - Rate of absenteeism is slightly more in highschool, as compared to other education category.

# In[18]:


trace1 = go.Bar(
    x=df['Transportation expense'],
    y=df['Absenteeism time in hours'],
    name='Transportation expense'
)

data = [trace1]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# In[19]:


trace1 = go.Bar(
    x=df['Distance from Residence to Work'],
    y=df['Absenteeism time in hours'],
    name='Transportation expense'
)

data = [trace1]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# In[20]:


trace1 = go.Bar(
    x=df['Service time'],
    y=df['Absenteeism time in hours'],
    name='Transportation expense'
)

data = [trace1]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# ###### From the above plot we can see that -
# - Rate of absenteeism is slightly more where the service time is more than 9

# In[21]:


trace1 = go.Bar(
    x=df['Age'],
    y=df['Absenteeism time in hours'],
    name='Transportation expense'
)

data = [trace1]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# In[22]:


trace1 = go.Bar(
    x=df['Work load Average/day '],
    y=df['Absenteeism time in hours'],
    name='Transportation expense'
)

data = [trace1]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# In[23]:


trace1 = go.Bar(
    x=df['Hit target'],
    y=df['Absenteeism time in hours'],
    name='Transportation expense'
)

data = [trace1]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# ###### From the above plot we can see that -
# - Rate of absenteeism is observed to be more when hit target is above 90.

# In[27]:


#Correlation alnalysis using heat-map

df_corr = df[num_var]
f, ax = plt.subplots(figsize=(10,10))
plt.title('Correlation between numerical predictors',size=14,y=1.05)
corr = df_corr.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap= sns.diverging_palette(220,10, as_cmap = True), square=True,
            annot = True,ax=ax)


# #### From the above heat-map we can infer the following:
# - None of the numeric variables are highly correlated.

# ### Pre-processing

# In[316]:


#Outliers and mising values
#Outlier analyis for numeric variables:
for i in num_var:
    q25 = df[i].quantile(0.25)
    q75 = df[i].quantile(0.75)
    iqr = q75 - q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    
    df.loc[df[i] < min, i ] = np.nan
    df.loc[df[i] > max, i ] = np.nan


# In[317]:


#Missing value analysis for numeric variables
#impute missing values(NAN) in contineos/numeric variables with mean
for i in num_var:
    df[i] = df[i].fillna(df[i].mean())


# In[318]:


#replace missing values with nan in categorical variables
for i in cat_names:
    df[i] = df[i].astype(str)
    df[i] = df[i].replace({'':np.nan},regex=True)
    df[i] = df[i].astype("category")
    for j in range(len(df)):
        if(df[i].iloc[j] == 'nan'):
            x = df['ID'].iloc[j]
            for k in range(len(df)):
                if(df['ID'].iloc[k] == x):
                    df[i].iloc[j] = df[i].iloc[k]


# In[319]:


var = ['ID','Reason for absence','Month of absence','Day of the week','Seasons','Transportation expense','Distance from Residence to Work','Service time','Age','Work load Average/day ','Hit target','Disciplinary failure','Education','Son','Social drinker','Social smoker','Pet','Weight','Height','Body mass index','Absenteeism time in hours']
for i in var :
    df[i] = df[i].astype(float)


# In[320]:


df = df.drop(["ID"], axis=1)


# ### Modeling

# In[321]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)


# In[322]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(train.iloc[:,0:19],train.iloc[:,19])

predictions_test = regr.predict(test.iloc[:,0:19])
predictions_train = regr.predict(train.iloc[:,0:19])

from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error for test data: %.2f" % mean_squared_error(test.iloc[:,19], predictions_test))
print("Mean squared error for train data: %.2f" % mean_squared_error(train.iloc[:,19], predictions_train))


# In[323]:


#linear regression using statsmodels
import statsmodels.api as sm
x = train.iloc[:,19]
y = train.iloc[:,0:19]
model = sm.OLS(x,y).fit()
model.summary()


# In[324]:


#Decision tree
from sklearn.tree import DecisionTreeRegressor
fit = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:19],train.iloc[:,19])

d_predictions_test = fit.predict(test.iloc[:,0:19])
d_predictions_train = fit.predict(train.iloc[:,0:19])

print("Mean squared error of test data : %.2f" % mean_squared_error(test.iloc[:,19], d_predictions_test))
print("Mean squared error of train data : %.2f" % mean_squared_error(train.iloc[:,19], d_predictions_train))


# ######  Now, after model development we will use this model on the whole data set to predict values and then will segregate it month wise to predict monthly loss.

# In[325]:


df1['Predicted Loss'] = fit.predict(df.iloc[:,0:19])
print("Mean squared error of train data : %.2f" % mean_squared_error(df1.iloc[:,20], df1['Predicted Loss']))


# In[326]:


df1 = df1.drop(['ID','Reason for absence','Day of the week','Seasons','Transportation expense','Distance from Residence to Work','Service time','Age','Work load Average/day ','Hit target','Disciplinary failure','Education','Son','Social drinker','Social smoker','Pet','Weight','Height','Body mass index','Absenteeism time in hours'], axis=1)


# In[327]:


df2 =pd.DataFrame(df1[['Month of absence', 'Predicted Loss']].groupby(['Month of absence'], as_index=False).sum().sort_values(by='Predicted Loss', ascending=False))


# In[328]:


for i in range(len(df2)):
    df2['Predicted Loss'].iloc[i] = int((df2['Predicted Loss'].iloc[i]))
    
test = df2.sort_values(by='Month of absence', ascending = True)


# In[331]:


test.to_csv('Monthly_loss.csv', index=False)

