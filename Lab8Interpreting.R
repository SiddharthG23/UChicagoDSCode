#################################Lab 8#####################
# This script has two parts
#  - Using a predictive model to explain the results of a descriptive model
#  - using a clustering on the results of a predictive model to explain the results of the predictive model
#  (- optional: using the same weighted variables as in the second part above for identifying the top three reasons)

##########################Interpreting the results of a clustering using a decision tree #####

# prepare data
data(iris)    # load the data into memory

# summarize the dataset
summary(iris)

# before performing k-means clusters with iris data
# remove the Species variable by setting it to NULL
newiris=iris
newiris$Species = NULL

# apply kmeans and save the clustering result to kc.
# the parantheses tell R to print/evaluate the object, alternatively we could enter
# kc=kmeans(newiris,3); print(kc)  but this gives us a simpler way to do both in one line
# There are two inputs to kmeans the dataset of newiris and setting K to the value 3
# to understand the inputs and outputs you can ask for help from R using help(kmeans) or ?kmeans
# However, the help is really meant to be syntax help not help in understanding the algorithm
set.seed(1248765792)   # set the seed so the random initialization of the clusters is the same
( kc = kmeans(newiris, 3) )

# notice that kc is a list.  Quite frequently R returns lists as the result, so that it can
# organize many different variables into a single group.  In this case kmeans returns:
#   kc$cluster:   a vector of integers (from 1:K) indicating the cluster to which each point is allocated
#   kc$centers:   a matrix of cluster centres (the rows correspond to the clusters, columns are variables used for kmeans)
#   kc$totss:     total sum of squares, which says how much variation that originally was in the dataset
#   kc$withinss:  within-cluster sum of squares, vector with K dimensions, each element corresponds to a cluster
#   kc$betweenss: the between-cluster sum of squares which equals is the difference between totss and betweenss
#   kc$size:      the number of pionts in each cluster
#   kc$iter:      the number of iterations
#   kc$ifault:    integer that indicates a possible algorithm problem
str(kc)

#Make new categorical variable for Cluster3
Cluster3=ifelse(kc$cluster==3,"Yes","No")

#Attach the column to newiris
newiris =data.frame(newiris,Cluster3)

# Decision tree to "explain" features of Cluster1 as categorical value
if (!require(rpart)) {install.packages("rpart"); library(rpart)}
if (!require(rpart.plot)) {install.packages("rpart.plot"); library(rpart.plot)}
par(mfrow=c(1,1))
iris.tree=rpart(Cluster3~.,newiris)
prp(iris.tree,extra=101)

# compare the Species label with the cluster result
table(iris$Species, kc$cluster)
barplot(table(iris$Species,iris$Petal.Length),xlab ="Petal length")

# scatter plot of each cluster with a different color for each cluster
plot(newiris[c("Petal.Length", "Sepal.Width")], col=kc$cluster)
# plot the cluster centroids
points(kc$centers[,c("Petal.Length", "Sepal.Width")], col=1:3, pch=8, cex=2)
legend("topright",legend=as.character(1:3),col=1:3,pch=8,bty='n')

# plot iris data in boxplot
par(mfrow=c(4,1),mar=c(3,4,1,1))  # mfrow=c(4,1) tells R to plot 4 rows and 1 column, and mar is the margin
# boxplot (as well as many other functions in R) use what is known as formula syntax
# the basics of formula is left hand side is the dependent variance and the right side are
# the independent variables. So for instance Petal.Length~Species tells boxplot to 
# have Petal.Length in the y-axis against Species in the x-axis.  One nice thing is that
# when use formula syntax R knows that Species is a factor variable so it knows to 
# automatically create dummy variables for each of its levels "setosa", "versicolor" and
# "virginica".  Notice if you execute the command boxplot(Petal.Length) then we only
# get a single boxplot that has all species.
boxplot(Petal.Length~Species,data=iris,ylab="Petal Length")
boxplot(Petal.Width~Species,data=iris,ylab="Petal Width")
boxplot(Sepal.Length~Species,data=iris,ylab="Sepal Length")
boxplot(Sepal.Width~Species,data=iris,ylab="Sepal Width")
par(mfrow=c(1,1))  # reset to one graph in the panel


################## Using clustering to explain Logistic Regression coefficients######
# Make sure loans-default.csv is in the same folder as this file
loans = read.csv("loans-default.csv")

## Training and Validation using hold out method for logistic regression
set.seed(777)
N = nrow(loans)
randvalue <- runif(N)
trainsample <- randvalue < .7
testsample <- (randvalue >= .7)

### Remove correlated predictors using correlations
(correlations <- cor(loans[trainsample,c(1,3:14)]))
if (!require(corrplot)) {install.packages("corrplot"); library(corrplot)}
#Generate a heat map of correlated predictors
#  (the hclust parameter orders the rows and columns according to a hierarchical clustering method)
corrplot(correlations, order="hclust")
# int.rate is correlated with fico so remove it
# credit.policy is correlated with inq.last.6mths so remove it
# The . is shorthand for everything in the dataset, so all variables
#  The -int.rate along with the . says don't use the int.rate variable
# 
lrmdl = glm(default ~ . - int.rate -credit.policy, data=loans[trainsample,], family="binomial")
summary(lrmdl)
# Predict the probability of not fully paying back using lrmodel and test set
lr.default.prob = predict(lrmdl, newdata=loans[testsample,], type="response")
test = loans[testsample,]$default

# Create the confusion matrix of the test set
Conf.lr = table(loans[testsample,]$default, lr.default.prob > 0.3)
Conf.lr
# Accuracy of the logistic regression model
(Conf.lr[1,1]+Conf.lr[2,2])/sum(Conf.lr)


### cluster consumers based upon the data weighted by the logistic regression coefficients
### the purpose of this analysis is to identify a smaller set of 'prototypical' customers
 
# setup some values
ncluster=2    # number of clusters

# "weight" the data using the logistic regression coefficients.  this allows the cluster to look at
# the variables based upon their contribution to their log-odds ratio
coefdata=summary(lrmdl)$coefficients  # extract coefficients estimates and std errors and z values
parm=coefdata[,1]  # just extract the parameter estimates in the 1st column
modeldata=model.matrix(lrmdl,data=loans)
wuserdata=sweep(modeldata,MARGIN=2,parm,"*")  # multiply each row in userdata by parm vector


userpred = predict(lrmdl,newdata=loans,type='response')  # predict prob for all users
# let's only consider those users that have a high probability of default (say the top quartile)
(p75=quantile(userpred[testsample],.75))  # only consider those in the test data
useridx=which(userpred>=p75 & testsample)  # get the indexes of those loans with high default

# cluster the users into groups
set.seed(612490)   # make sure we get the same solution
grpA=kmeans(wuserdata[useridx,],ncluster,nstart=50)  # add nstart=50 to choose 50 different random seeds

# update the cluster solution with one new group that has everyone else
save.cluster=grpA$cluster   # save the original results
grpA$cluster=rep(ncluster+1,nrow(loans))  # by default place everyone into the last cluster
grpA$cluster[useridx]=save.cluster
grpA$centers=rbind(grpA$centers,colMeans(wuserdata[userpred<p75 & testsample,])) 

# create boxplot of userpred using the clusters
boxplot(userpred[testsample]~grpA$cluster[testsample],xlab="Cluster",ylab="Pr(Default)")
boxplot(loans$fico[testsample]~grpA$cluster[testsample],xlab="Cluster",ylab="fico")

# create a parallel plot to visualize the centroid values
parallelplot(grpA$centers,auto.key=list(text=as.character(1:nrow(grpA$centers)),space="top",columns=5,lines=T))
parallelplot(grpA$centers,auto.key=list(text=as.character(1:nrow(grpA$centers)),space="top",columns=5,lines=T),common.scale=TRUE)  # choose min and max across variables to give an absolute comparison

shortvarlist=c("fico","inq.last.6mths","revol.bal","installment")   # short list of variables
parallelplot(grpA$centers[,shortvarlist],varnames=shortvarlist,auto.key=list(text=as.character(1:nrow(grpA$centers)),space="top",columns=5,lines=T))


################## Finding the top 3 reasons for default from the LR model#####
# create vector of variables used in model called mvarlist, and add other variables that we want to write out
# these lines require the lrmdl and ctree to be created appropriately above
varlist=names(coefficients(lrmdl))[-1]   # get the variables used in your logistic regression moodel, except the intercept which is in first position
print(varlist)  # vector of variables to save

# create new dataset with just selected variables for all users
userdata=model.matrix(lrmdl,data=loans)  # construct the data used in the model

# "weight" the data using the logistic regression coefficients.  this allows the cluster to look at
# the variables based upon their contribution to their log-odds ratio
coefdata=summary(lrmdl)$coefficients  # extract coefficients estimates and std errors and z values
parm=coefdata[,1]  # just extract the parameter estimates in the 1st column
wuserdata=sweep(userdata,MARGIN=2,parm,"*")  # multiply each row in userdata by parm vector

# find the two most important reasons customer will default
vnames=colnames(wuserdata[,-1])  # create vector of variables used in model (except for intercept)
result=t(apply(wuserdata[,-1],1,function(x) {vnames[order(x)[1:3]]}))  # sort effects by row and return top 3
result=as.data.frame(result)  # turn the output into a data frame
userpred = predict(lrmdl,newdata=loans,type='response')  # predict prob for all users
result$prob=userpred  # include the probability
head(result[result$prob>.75,])  # list out few users, the 0.75 is an arbitrary threshold -- change as desired -- perhaps to percentile
table(result$V1[result$prob>.75])  # reason for default of most likely
table(result$V2[result$prob>.75])  # 2nd reason for default
table(result$V3[result$prob>.75])  # 3rd reason for defaulttab


