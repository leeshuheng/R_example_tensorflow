### credit: https://haosdent.gitbooks.io/tensorflow-document/content/tutorials/tflearn/
### credit: https://rstudio.github.io/tensorflow/tutorial_mnist_pros.html

library(tensorflow)

data(iris)
sam.idx <- sample(1:nrow(iris), 100, replace = F)

iris.train <- iris[sam.idx,]
iris.test <- iris[-sam.idx,]

#h <- tf$constant(matrix(c(10L, 20L, 10L), ncol = 3))

fc <- c(tf$contrib$layers$real_valued_column("", dimension = 4L))
classifier <- tf$contrib$learn$DNNClassifier(feature_columns = fc,
                                      n_classes = 3L, model_dir = "/tmp/iris_m",
                                      hidden_units = c(10L, 20L, 10L))
# Fit model.
classifier$fit(x = data.matrix(iris.train[,1:4]),
               y = as.integer(iris.train$Species) - 1L,
               steps = 2000L)

# Evaluate accuracy.
# base-0
classifier$evaluate(x = data.matrix(iris.test[,1:4]),
                    y = as.integer(iris.test$Species) - 1L)["accuracy"]

new.data <- matrix(c(6.4, 3.2, 4.5, 1.5, 5.8, 3.1, 5.0, 1.7), nrow = 2)

y <- classifier$predict(new.data)
reticulate::iterate(y, print)

y <- classifier$predict(data.matrix(iris.test[,1:4]))
reticulate::iterate(y, print)

datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

fc <- c(tf$contrib$layers$real_valued_column("", dimension = 784L))
classifier <- tf$contrib$learn$DNNClassifier(feature_columns = fc,
                                      n_classes = 10L, model_dir = "/tmp/mnist_m",
                                      hidden_units = c(392L, 196L, 196L)) 

train.labels <- as.integer(apply(mnist$train$labels, 1, function(x) which(x == 1)) - 1L)
classifier$fit(x = mnist$train$images,
               y = train.labels,
               steps = 100L)

test.labels <- as.integer(apply(mnist$test$labels, 1, function(x) which(x == 1)) - 1L)
classifier$evaluate(x = mnist$test$images,
                    y = test.labels)["accuracy"]
