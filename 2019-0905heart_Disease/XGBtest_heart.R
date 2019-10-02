#
df<- read_csv("C:/Users/user/Documents/2019-0905heart_Disease/heart.csv",col_names = T,
              cols(
                age = col_double(),
                sex = col_factor(levels = c(0,1)),
                cp  = col_factor(levels = c(0,1,2,3)),
                trestbps = col_double(),
                chol = col_double(),
                fbs = col_factor(levels = c(0,1)),
                restecg = col_factor(levels = c(0,1,2)),
                thalach = col_double(),
                exang = col_factor(levels = c(0,1)),
                oldpeak = col_double(),
                slope = col_factor(levels = c(0,1,2)),
                ca = col_factor(levels = c(0,1,2,3,4)),
                thal = col_factor(levels = c(0,1,2,3)),
                target = col_factor(levels = c(0,1))
              )
              )
#主要
library(xgboost)
#讀入csv檔
library(readr)
#分割資料集
#library(scorecard)
library(rsample)   # for data splitting
#資料處理
library(dplyr)
library(vtreat)
#處理表格
library(data.table)
#矩陣
library(Matrix)
#畫圖
library(ggplot2)

df<- read_csv("C:/Users/user/Documents/2019-0905heart_Disease/heart.csv")

str(df)
head(df)
summary(df)

#分割訓練集和測試集
set.seed(123)
df.split <- initial_split(df, ratio = 0.7)
df.train <- training(df.split)
df.test <- testing(df.split)

# variable names
features <- setdiff(names(df.train), "traget")

# Create the treatment plan from the training data
treatplan <- vtreat::designTreatmentsZ(df.train, features, verbose = FALSE)

# Get the "clean" variable names from the scoreFrame
new_vars <- treatplan %>%
  magrittr::use_series(scoreFrame) %>%        
  dplyr::filter(code %in% c("clean", "lev")) %>% 
  magrittr::use_series(varName) 

# Prepare the training data
features_train <- vtreat::prepare(treatplan, df.train, varRestriction = new_vars) %>% as.matrix()
response_train <- df.train$`target`

# Prepare the test data
features_test <- vtreat::prepare(treatplan, df.test, varRestriction = new_vars) %>% as.matrix()
response_test <- df.test$`target`

#############
#training####
#############
# 将自变量转化为矩阵
df.train1 <- data.matrix(df.train[,c(1:13)]) 
# 利用Matrix函数，将sparse参数设置为TRUE，转化为稀疏矩阵
df.train2 <- Matrix(df.train1,sparse=T)
# 将因变量转化为numeric
df.train3 <- data.matrix(df.train[,14]) 
# 将自变量和因变量拼接为list
df.train4 <- list(data=df.train2,label=df.train3) 
# 构造模型需要的xgb.DMatrix对象，处理对象为稀疏矩阵
dtrain <- xgb.DMatrix(data = df.train4$data, label = df.train4$label)

#############
#testing#####
#############
# 将自变量转化为矩阵
df.test1 <- data.matrix(df.test[,c(1:13)]) 
# 利用Matrix函数，将sparse参数设置为TRUE，转化为稀疏矩阵
df.test2 <- Matrix(df.test1,sparse=T) 
# 将因变量转化为numeric
df.test3 <- data.matrix(df.test[,14]) 
# 将自变量和因变量拼接为list
df.test4 <- list(data=df.test2,label=df.test3) 
# 构造模型需要的xgb.DMatrix对象，处理对象为稀疏矩阵
dtest <- xgb.DMatrix(data = df.test1) 

# create hyperparameter grid創建超參數網格
hyper_grid <- expand.grid(
  eta = c(.01, .05, .1, .3),
  max_depth = c(1, 3, 5, 7),
  min_child_weight = c(1, 3, 5, 7),
  subsample = c(.65, .8, 1), 
  colsample_bytree = c(.8, .9, 1),
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

###################################
#使用迴圈一一去執行XGBoost模型套用不同參數組合的結果，並將結果指標儲存
##################################
# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # create parameter list
  # 目标函数：logistic的二分类模型，因为Y值是二元的
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  # reproducibility
  set.seed(123)
  
  # train model
  #"reg:linear"
  #"binary:logistic"
  xgb.tune <- xgb.cv(
    params = params,
    data = df.train2,
    label = df.train3,
    nrounds = 5000,
    nfold = 5,
    objective = "binary:logistic",  # for regression models
    verbose = 0,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_rmse_mean)
  hyper_grid$min_RMSE[i] <- min(xgb.tune$evaluation_log$test_rmse_mean)
}

hyper_grid %>%
  dplyr::arrange(min_RMSE) %>%
  head(10)
#####################################
#選擇模型
# parameter list
params <- list(
  objective = "binary:logistic",
  eta = 0.3,
  max_depth = 1,
  min_child_weight = 5,
  subsample = 0.8,
  colsample_bytree = 0.9
)

# train final model
#"binary:logistic"
#"reg:linear"
xgb.fit.final <- xgboost(
  params = params,
  data = df.train2,
  label = df.train3,
  nrounds = 50,
  objective = "binary:logistic",
  verbose = 0
)
##################################
##################################
##################################
##################################
##################################
##################################
##################################
# 将dataframe格式转换成xgb.DMatrix格式
# Y值的列名: 'target'
dtrain <- xgb.DMatrix(data=select(train,-target)%>%as.matrix,label= train$target%>%as.matrix)

#利用 xgb.cv 调参
best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

# 自定义调参组合
for (iter in 1:50) {
  param <- list(objective = "binary:logistic",     # 目标函数：logistic的二分类模型，因为Y值是二元的
                eval_metric = c("logloss"),                # 评估指标：logloss
                max_depth = sample(6:10, 1),               # 最大深度的调节范围：1个 6-10 区间的数
                eta = runif(1, .01, .3),                   # eta收缩步长调节范围：1个 0.01-0.3区间的数
                gamma = runif(1, 0.0, 0.2),                # gamma最小损失调节范围：1个 0-0.2区间的数
                subsample = runif(1, .6, .9),             
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 50                                   # 迭代次数：50
  cv.nfold = 5                                     # 5折交叉验证
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=dtrain, params = param, nthread=6, metrics=c("auc","rmse","error"),
                 nfold=cv.nfold, nrounds=cv.nround, watchlist = list(),
                 verbose = F, early_stop_round=8, maximize=FALSE)
  
  min_logloss = min(mdcv$evaluation_log[,test_logloss_mean])
  min_logloss_index = which.min(mdcv$evaluation_log[,test_logloss_mean])
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
}

(nround = best_logloss_index)
set.seed(best_seednumber)
best_seednumber
(best_param)                # 显示最佳参数组合，到后面真正的模型要用



#3) 绘制 auc | rmse | error 曲线
#mdcv$evaluation_log

xgb_plot=function(input,output){
  history=input
  train_history=history[,1:8]%>%mutate(id=row.names(history),class="train")
  test_history=history[,9:16]%>%mutate(id=row.names(history),class="test")
  colnames(train_history)=c("logloss.mean","logloss.std","auc.mean","auc.std","rmse.mean","rmse.std","error.mean","error.std","id","class")
  colnames(test_history)=c("logloss.mean","logloss.std","auc.mean","auc.std","rmse.mean","rmse.std","error.mean","error.std","id","class")
  
  his=rbind(train_history,test_history)
  his$id=his$id%>%as.numeric
  his$class=his$class%>%factor
  
  if(output=="auc"){ 
    auc=ggplot(data=his,aes(x=id, y=auc.mean,ymin=auc.mean-auc.std,ymax=auc.mean+auc.std,fill=class),linetype=class)+
      geom_line()+
      geom_ribbon(alpha=0.5)+
      labs(x="nround",y=NULL,title = "XGB Cross Validation AUC")+
      theme(title=element_text(size=15))+
      theme_bw()
    return(auc)
  }
  
  
  if(output=="rmse"){
    rmse=ggplot(data=his,aes(x=id, y=rmse.mean,ymin=rmse.mean-rmse.std,ymax=rmse.mean+rmse.std,fill=class),linetype=class)+
      geom_line()+
      geom_ribbon(alpha=0.5)+
      labs(x="nround",y=NULL,title = "XGB Cross Validation RMSE")+
      theme(title=element_text(size=15))+
      theme_bw()
    return(rmse)
  }
  
  if(output=="error"){
    error=ggplot(data=his,aes(x=id,y=error.mean,ymin=error.mean-error.std,ymax=error.mean+error.std,fill=class),linetype=class)+
      geom_line()+
      geom_ribbon(alpha=0.5)+
      labs(x="nround",y=NULL,title = "XGB Cross Validation ERROR")+
      theme(title=element_text(size=15))+
      theme_bw()
    return(error)
  }
  
}
#AUC曲線
xgb_plot(mdcv$evaluation_log[,-1]%>%data.frame,"auc")
