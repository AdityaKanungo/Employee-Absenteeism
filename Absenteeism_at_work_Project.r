rm(list=ls())
setwd("C:/Users/BATMAN/Desktop/project 2 edwisor")
getwd()
install.packages("xlsx")
library("xlsx")
library(corrplot)
library(DMwR)
library(e1071)
library(caret)
library(class)
library(C50)
install.packages("readxl")
library(readxl)
# Load train data
df =  read_excel("Absenteeism_at_work_Project.xls")
df1 = df


#All features/variables
var = colnames(df)

#Numeric variables
numeric_data <- subset(df, select = c('ID','Transportation expense','Service time','Age','Hit target','Absenteeism time in hours'))
num_var = colnames(numeric_data)
#Categorical variables
category_data <- subset(df, select = c('Reason for absence','Day of the week','Seasons','Month of absence','Distance from Residence to Work','Work load Average/day','Education','Son','Pet','Disciplinary failure','Social drinker','Social smoker','Height','Weight','Body mass index'))
cat_var = colnames(category_data)

#Check missing vales 
sum(is.na(df$ID))
missing_val = data.frame(apply(df, 2, function(x){sum(is.na(x))}))

#use boxplot to remove replace outliers with NA
for(i in num_var){
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  df[,i][df[,i] %in% val] = NA
}

#imputing missing values with NA
df = knnImputation(df, k = 3)



