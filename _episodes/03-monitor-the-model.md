---
title: "Monitor the training process"
teaching: 0
exercises: 0
questions:
- "How do I set the training goal?"
- "How do I monitor the training process?"
- "How do I detect (and avoid) overfitting?"
- "What are common options to improve the model performance?" 
objectives:
- "Explain the importance of splitting the training data"
- "Use the data splits to plot the training process"
- "Set the training goal for your deep neural network"
- "Measure the performance of your deep neural network"
- "Interpret the training plots to recognize overfitting"
- "Implement basic strategies to prevent overfitting"

keypoints:
- "Separate training, validation, and test sets allows monitoring and evaluating your model."
- "Batchnormalization scales the data as part of the model."
---

## Explore the data

### Import dataset
Here we want to work with the *weather prediction dataset* which can be [downloaded from Zenodo](https://doi.org/10.5281/zenodo.5071376).
It contains daily weather observations from 18 different European cities or places through the years 2000 to 2010. For all locations the data contains the variables ‘mean temperature’, ‘max temperature’, and ‘min temperature’. In addition, for multiple of the following variables are provided: 'cloud_cover', 'wind_speed', 'wind_gust', 'humidity', 'pressure', 'global_radiation', 'precipitation', 'sunshine', but not all of them are provided for all locations. A more extensive description of the dataset including the different physical units is given in accompanying metadata file.
![18 locations in the weather prediction dataset](../fig/03_weather_prediction_dataset_map.png)

~~~
import pandas as pd

filename_data = "weather_prediction_dataset.csv"
data = pd.read_csv(filename_data)
data.head()
~~~
{:.language-python}

| | DATE 	| MONTH | 	BASEL_cloud_cover 	| 	BASEL_humidity 	| 	BASEL_pressure	| ... |
|------:|------:|---------------:|--------------:|------------------:|------------:|------------:|
|0| 	20000101 	|1 	|8 	|0.89 	|1.0286|... |
|1| 	20000102 	|1 	|8 	|0.87 	|1.0318|... |
|2| 	20000103 	|1 	|5 	|0.81 	|1.0314|... |
|3| 	20000104 	|1 	|7 	|0.79 	|1.0262|... |
|4| 	20000105 	|1 	|5 	|0.90 	|1.0246|... |
{: .output}

### Brief exploration of the data
Let us start with a quick look at the type of features that we find in the data.
~~~
data.columns
~~~
{:.language-python}

~~~
Index(['DATE', 'MONTH', 'BASEL_cloud_cover', 'BASEL_humidity',
       'BASEL_pressure', 'BASEL_global_radiation', 'BASEL_precipitation',
       'BASEL_sunshine', 'BASEL_temp_mean', 'BASEL_temp_min',
       ...
       'STOCKHOLM_temp_min', 'STOCKHOLM_temp_max', 'TOURS_wind_speed',
       'TOURS_humidity', 'TOURS_pressure', 'TOURS_global_radiation',
       'TOURS_precipitation', 'TOURS_temp_mean', 'TOURS_temp_min',
       'TOURS_temp_max'],
      dtype='object', length=165)
~~~
{:.output}

> ## Exercise: Explore the dataset
>
> Let's get a quick idea of the dataset.
>
> * How many data points do we have?
> * How many features does the data have (don't count month and date as a feature)?
> * What are the different measured variabel types in the data and how many are there (humidity etc.) ? 
>
> > ## Solution
> > ~~~
> > data.shape
> > ~~~
> > {:.language-python}
> > This will give both the number of datapoints (3654) and the number of features (163 + month + date).
> > 
> > To see what type of features the data contains we could run something like:
> > ~~~
> > print({x.split("_")[-1] for x in data.columns if x not in ["MONTH", "DATE"]})
> > ~~~
> > {:.language-python}
> > ~~~
> > {'humidity', 'radiation', 'sunshine', 'gust', 'mean', 'max', 'precipitation', 'pressure', 'cover', 'min', 'speed'}
> > ~~~
> > {:.output}
> > An alternative way which is slightly more complicated but gives better results is using regex.
> > ~~~
> > import re
> > feature_names = set()
> > for col in data.columns:
> >     feature_names.update(re.findall('[^A-Z]{2,}', col))
> >     
> > feature_names
> > ~~~
> > In total there are 11 different measured variables.
> {:.solution}
{:.challenge}


## Prepare the data

### Select a subset and split into data (X) and labels (y)
The full dataset comprises 10 years (3654 days) from which we here will only select the first 3 years.
We will then define what exactly we want to predict from this data. A very common task with weather data is to make a predicion about the weather somewhere in the future, say the next day. The present dataset is sorted by "DATE", so for each row `i` in the table we can pick a corresponding feature and location from row `i+1` that we later want to predict with our model.
Here we will pick a rather difficult-to-predict feature, sunshine hours, which we want to predict for the location: BASEL.

~~~
nr_rows = 365*3
# data
X_data = data.loc[:nr_rows].drop(columns=['DATE', 'MONTH'])

# labels (sunshine hours the next day)
y_data = data.loc[1:(nr_rows + 1)]["BASEL_sunshine"]
~~~
{:.language-python}


### Split data and labels into training, validation, and test set
As with classical machine learning techniques, it is common in deep learning to split off a *test set* which remains untouched during model training and tuning. It is then later be used to evaluate the model performance. Here, we will also split off an additional *validation set*, the reason of which will hopefully become clearer later in this lesson.

> ## Exercise: Split data into training, validation, and test set
>
> Split the data into 3 completely separate set to be used for training, validation, and testing using the `train_test_split` function from `sklearn.model_selection`. This can be done in two steps.
> First, split the data into training set (70% of all datapoints) and validation+test set (30%). Then split the second one again into two sets (both roughly equal in size).
>
> * How many data points do you have in the training, validation, and test sets?
> 
>  **Hint:**
>  ~~~
>  from sklearn.model_selection import train_test_split
>  
>  X_train, X_not_train, y_train, y_not_train = train_test_split(X, y, test_size=0.3, random_state=0) 
>  ~~~
>  {:.language-python}
>
> > ## Solution
> > ~~~
> > from sklearn.model_selection import train_test_split
> > 
> > X_train, X_not_train, y_train, y_not_train = train_test_split(X_data, y_data, test_size=0.3, random_state=0) 
> > X_val, X_test, y_val, y_test = train_test_split(X_not_train, y_not_train, test_size=0.5, random_state=0)
> > 
> > print(f"Data was split into training ({X_train.shape[0]})," \
> >       f" validation ({X_val.shape[0]}) and test set ({X_test.shape[0]}).")
> > ~~~
> > {:.language-python}
> > 
> > ~~~
> > Data was split into training (767), validation (164) and test set (165).
> > ~~~
> > {:.output}
> {:.solution}
{:.challenge}
      
## Build a dense neural network

### Regression and classification - how to set a training goal
- Explain how to define the output part of a neural network
- What is the loss function (and which one to chose for a regression or classification task)?


In episode 2 we trained a dense neural network on a *classification task*. For this one hot encoding was used together with a Categorical Crossentropy loss function.
This measured how close the distribution of the neural network outputs corresponds to the distribution of the three values in the one hot encoding.
Now we want to work on a *regression task*, thus not prediciting the right class for a datapoint but a certain value (could in principle also be several values). In our example we want to predict the sunshine hours in Basel (or any other place in the dataset) for tomorrow based on the weather data of all 18 locations today. 

The network should hence output a single float value which is why the last layer of our network will only consist of a single node. 

> ## Create the neural network
>
> We have seen how to build a dense neural network in episode 2. 
> Try now to construct a dense neural network with 3 layers for a regression task.
> Start with a network of a dense layer with 100 nodes, followed by one with 50 nodes and finally an output layer.
> Hint: Layers in keras are stacked by passing a layer to the next one like this
> ~~~
> inputs = keras.layers.Input(shape=...)
> next_layer = keras.layers.Dense(..., activation='relu')(inputs)
> next_layer = keras.layers.Dense(..., activation='relu')(next_layer) 
> #here we used the same layer name twice, but that is up to you
> ...
> next_layer = ...(next_layer) 
> ...
> #stack as many layers as you like
> ~~~
> {:.language-python}
>
> * What must here be the dimension of our input layer?
> * How would our output layer look like? What about the activation function? Tip: Remember that the activation function in our previous classification network scaled the outputs between 0 and 1.
>
> > ## Solution
> > Here we wrote a function for generating a keras model, because we plan on using this again in the following.
> > ~~~
> > from tensorflow import keras
> > 
> > def create_nn():
> >     # Input layer
> >     inputs = keras.Input(shape=(X_data.shape[1],), name='input')
> > 
> >     # Dense layers
> >     layers_dense = keras.layers.Dense(100, 'relu')(inputs)
> >     layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)
> > 
> >     # Output layer
> >     outputs = keras.layers.Dense(1)(layers_dense)
> > 
> >     return keras.Model(inputs=inputs, outputs=outputs, name="weather_prediction_model")
> > 
> > model = create_nn()
> > model.summary()
> > ~~~
> > {:.language-python}
> >
> > ~~~
> > Model: "weather_prediction_model"
> > _________________________________________________________________
> > Layer (type)                 Output Shape              Param #   
> > =================================================================
> > input (InputLayer)           [(None, 163)]             0         
> > _________________________________________________________________
> > dense_0 (Dense)              (None, 100)               16400     
> > _________________________________________________________________
> > dense_1 (Dense)              (None, 50)                5050      
> > _________________________________________________________________
> > dense_2 (Dense)              (None, 1)                 51        
> > =================================================================
> > Total params: 21,501
> > Trainable params: 21,501
> > Non-trainable params: 0
> > _________________________________________________________________
> > ~~~
> > {:.output}
> >
> > The shape of the input layer has to correspond to the number of features in our data: 163
> > 
> > The output layer here is a dense layer with only 1 node. And we here have chosen to use *no activation function*.
> > While we might use *softmax* for a classification task, here we do not want to restrict the possible outcomes for a start.
> > 
> > In addition, we have here chosen to write the network creation as a function so that we can use it later again to initiate new models.
> {:.solution}
{:.challenge}

When compiling the model we can define a few very important aspects. We will discuss them now in more detail.

### Loss function:
The loss is what the neural network will be optimized on during training, so chosing a suitable loss function is crucial for training neural networks.
In the given case we want to stimulate that the prodicted values are as close as possible to the true values. This is commonly done by using the *mean squared error* (mse) or the *mean absolute error* (mae), both of which should work OK in this case. Often, mse is prefered over mae because it "punishes" large prediction errors more severely.
In keras this is implemented in the `keras.losses.MeanSquaredError` class (see keras documentation: https://keras.io/api/losses/).

### Optimizer:
Somewhat coupled to the loss function is the *optimizer* that we want to use. 
The *optimizer* here refers to the algorithm with which the model learns to optimize on the set loss function. A basic example for such an optimizer would be *stochastic gradient descent*. For now, we can largely skip this step and simply pick one of the most common optimizers that works well for most tasks: the *Adam optimizer*. 

### Metrics:
In our first example (episode 2) we plotted the progression of the loss during training. 
That is indeed a good first indicator if things are working alright, i.e. if the loss is indeed decreasing as it should. 
However, when models become more complicated then also the loss functions often become less intuitive. 
That is why it is good practice to monitor the training process with additional, more intuitive metrics. 
They are not used to optimize the model, but are simply recorded during training. 
With Keras they can simply be added via `metrics=[...]` and can contain one or multiple metrics of interest. 
Here we could for instance chose to use `'mae'` the mean absolute error, or the the *root mean squared error* (RMSE) which unlike the *mse* has the same units as the predicted values.

~~~
model.compile(optimizer='adam',
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])
~~~
{: .language-python}

## Train a dense neural network
Now that we created and compiled our dense neural network, we can start training it.
One additional concept we need to introduce though, is the `batch_size`.
This defines how many samples from the training data will be used to estimate the error gradient before the model weights are updated.
Larger batches will produce better, more accurate gradient estimates but also less frequent updates of the weights.
Here we are going to use a batch size of 32 which is a common starting point.
~~~
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    verbose=2)
~~~
{: .language-python}

We can plot the training process using the history:
~~~
import seaborn as sns
import matplotlib.pyplot as plt
history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df['root_mean_squared_error'])
plt.xlabel("epochs")
plt.ylabel("RMSE")
~~~
{: .language-python}
![Output of plotting sample](../fig/03_training_history_1_rmse.png)

This looks very promising! Our metric ("RMSE") is dropping nicely and while it maybe keeps fluctuating a bit it does end up at fairly low *RMSE* values.
But the *RMSE* is just the root *mean* squared error, so we might want to look a bit more in detail how well our just trained model does in predicting the sunshine hours.

## Evaluate our model
There is not a single way to evaluate how a model performs. But there is at least two very common approaches. For a *classification task* that is to compute a *confusion matrix* for the test set which shows how often particular classes were predicted correctly or incorrectly. For the present *regression task* however, it makes more sense to compare true and predicted values in simple scatter plot. Hint: use `plt.scatter()`.

First, we will do the actual prediction step. 
> ## Predict the labels for both training and test set and compare to the true values
> Even though we here use a different model architecture and a different task compared to episode 2, the prediction step is mostly identical.
> Use the model to predict the labels for the training set and the test set and then compare them in a scatter plot to the true labels. Hint: use `plt.scatter()`.
> 
> * Is the performance of the model as you expected (or better/worse)? 
> * Is there a noteable difference between training set and test set? And if so, any idea why?
> 
> > ## Solution
> > ~~~
> > y_train_predicted = model.predict(X_train)
> > y_test_predicted = model.predict(X_test)
> > ~~~
> > {: .language-python}
> > We can then compare those to the true labels, for instance by
> > ~~~
> > fig, axes = plt.subplots(1, 2, figsize=(12, 6))
> > plt.style.use('ggplot')  # optional, that's only to define a visual style
> > axes[0].scatter(y_train_predicted, y_train, s=10, alpha=0.5, color="teal")
> > axes[0].set_title("training set")
> > axes[0].set_xlabel("predicted sunshine hours")
> > axes[0].set_ylabel("true sunshine hours")
> > 
> > axes[1].scatter(y_test_predicted, y_test, s=10, alpha=0.5, color="teal")
> > axes[1].set_title("test set")
> > axes[1].set_xlabel("predicted sunshine hours")
> > axes[1].set_ylabel("true sunshine hours")
> > ~~~
> > {: .language-python}
> > ![Scatter plot to evaluate training and test set](../fig/03_regression_training_test_comparison.png)
> > 
> > The accuracy on the training set is fairly good. 
> > In fact, considering that the task of predicting the daily sunshine hours is really not easy it might even be surprising how well the model predicts that 
> > (at least on the training set). Maybe a little too good?
> > For those familiar with (classical) machine learning this might look familiar. 
> > It is a very clear signature of **overfitting** which means that the model has to some extend memorized aspects of the training data. 
> > As a result makes much more accurate predictions on the training data than on unseen data.
> {:.solution}
{:.challenge}

Overfitting also happens in classical machine learning, but there it is usually interpreted as the model having more parameters than the training data would justify (say, a decision tree with too many branches for the number of training instances). As a consequence one would reduce the number of parameters to avoid overfitting.
In deep learning the situation is slightly different. It can -same as for classical machine learning- also be a sign of having a *too big* model, meaning a model with too many parameters (layers and/or nodes). However, in deep learning higher number of model parameters are often still considered acceptable and models often perform best (in terms of prediction accuracy) when they are at the verge of overfitting. So, in a way, training deep learning models is always a bit like playing with fire...

## Watch your model training closely
As we just saw, deep learning models are prone to overfitting. Instead of iterating through countless cycles of model trainings and subsequent evaluations with a reserved test set, it is common practice to work with a second split off dataset to monitor the model during training. This is the *validation set* which can be regarded as a second test set. As with the test set the datapoints of the *validation set* are not used for the actual model training itself. Instead we evaluate the model with the *validation set* after every epoch during training, for instance to splot if we see signs of clear overfitting.

Let's give this a try!

We need to initiate a new model -- otherwise Keras will simply assume that we want to continue training the model we already trained above.
~~~
model = create_nn()
model.compile(optimizer='adam',
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])
~~~
{: .language-python}

But now we train it with the small addition of also passing it our validation set:
~~~
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    validation_data=(X_val, y_val),
                    verbose=2)
~~~
{: .language-python}

> ## Exercise: plot the training progress.
> 
> As before the history allows plotting the training progress. But now we can plot both the performance on the training data and on the validation data!
> * Is there a difference between the training and validation data? And if so, what would this imply?
> 
> > ## Solution
> > ~~~
> > history_df = pd.DataFrame.from_dict(history.history)
> > sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
> > plt.xlabel("epochs")
> > plt.ylabel("RMSE")
> > ~~~
> > {: .language-python}
> > ![Output of plotting sample](../fig/03_training_history_2_rmse.png)
> > 
> > This clearly shows that something is not completely right here. 
> > The model predictions on the validation set quickly seem to reach a plateau while the performance on the training set keeps improving.
> > That is a clear signature of overfitting.
> {:.solution}
{:.challenge}

## Counteract model overfitting
Overfitting is a very common issue and there are many strategies to handle it.
Most similar to classical machine learning might to **reduce the number of parameters**.

> ## Try to reduce the degree of overfitting by lowering the number of parameters
>
> We can keep the network architecture unchanged (2 dense layers + a one-node output layer) and only play with the number of nodes per layer.
> Try to lower the number of nodes in one or both of the two dense layers and observe the changes to the training and validation losses.
> If time is short: Suggestion is to run one network with only 10 and 5 nodes in the first and second layer.
>
> * Is it possible to get rid of overfitting this way?
> * Does the overall performance suffer or does it mostly stay the same?
> * How low can you go with the number of paramters without notable effect on the performance on the validation set?
>
> > ## Solution
> > ~~~
> > def create_nn(nodes1, nodes2):
> >     # Input layer
> >     inputs = keras.layers.Input(shape=(X_data.shape[1],), name='input')
> > 
> >     # Dense layers
> >     layers_dense = keras.layers.Dense(nodes1, 'relu')(inputs)
> >     layers_dense = keras.layers.Dense(nodes2, 'relu')(layers_dense)
> > 
> >     # Output layer
> >     outputs = keras.layers.Dense(1)(layers_dense)
> > 
> >     return keras.Model(inputs=inputs, outputs=outputs, name="model_small")
> > 
> > model = create_nn(10, 5)
> > model.summary()
> > ~~~
> > {:.language-python}
> >
> > ~~~
> > Model: "model_small"
> > _________________________________________________________________
> > Layer (type)                 Output Shape              Param #   
> > =================================================================
> > input (InputLayer)           [(None, 163)]             0         
> > _________________________________________________________________
> > dense_21 (Dense)             (None, 10)                1640      
> > _________________________________________________________________
> > dense_22 (Dense)             (None, 5)                 55        
> > _________________________________________________________________
> > dense_23 (Dense)             (None, 1)                 6         
> > =================================================================
> > Total params: 1,701
> > Trainable params: 1,701
> > Non-trainable params: 0
> > _________________________________________________________________
> > ~~~
> > {:.output}
> >
> > ~~~
> > model.compile(optimizer='adam',
> >               loss='mse',
> >               metrics=[keras.metrics.RootMeanSquaredError()])
> > history = model.fit(X_train, y_train,
> >                     batch_size = 32,
> >                     epochs = 200,
> >                     validation_data=(X_val, y_val), verbose = 2)
> >                     
> > history_df = pd.DataFrame.from_dict(history.history)
> > sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
> > plt.xlabel("epochs")
> > plt.ylabel("RMSE")
> > ~~~
> > {:.language-python}
> > 
> > ![Output of plotting sample](../fig/03_training_history_3_rmse_smaller_model.png)
> > 
> > There is obviously no single correct solution here. But you will have noticed that the number of nodes can be reduced quiet a bit!
> > 
> > In general, it quickly becomes a very complicated search for the right "sweet spot", i.e. the settings for which overfitting will be (nearly) avoided but which still performes equally well.
> > 
> {:.solution}
{:.challenge}

We saw that reducing the number of parameters can be a strategy to avoid overfitting.
In practice, however, this is usually not the (main) way to go when it comes to deep learning.
One reason is, that finding the sweet spot can be really hard and time consuming. And it has to be repeated every time the model is adapted, e.g. when more training data becomes available.

## Early stopping: stop when things are looking best
Arguable **the** most common technique to avoid (severe) overfitting in deep learning is called **early stopping**.
As the name suggests, this technique just means that you stop the model training if things do not seem to improve anymore.
More specifically, this usually means that the training is stopped if the validation loss does not (notably) improve anymore.
Early stopping is both intuitive and effective to use, so it has become a standard addition for model training.

To better study the effect, we can now savely go back to models with many (too many?) parameters:
~~~
model = create_nn(100, 50)
model.compile(optimizer='adam',
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])
~~~
{: .language-python}

To apply early stopping during training it is easiest to use keras `EarlyStopping` class.
This allows to define the condition of when to stop training. In our case we will say when the validation loss is lowest.
However, since we have seen quiet some fluctuation of the losses during training above we will also set `patience=10` which means that the model will stop training of the validation loss has not gone down for 10 epochs.
~~~
from tensorflow.keras.callbacks import EarlyStopping

earlystopper = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
    )

history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 200,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper],
                    verbose = 2)
~~~
{: .language-python}

As before, we can plot the losses during training:
~~~
history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
plt.xlabel("epochs")
plt.ylabel("RMSE")
~~~
{: .language-python}

![Output of plotting sample](../fig/03_training_history_3_rmse_early_stopping.png)

This still seems to reveal the onset of overfitting, but the training stops before the discrepancy between training and validation loss can grow further.
Despite avoiding severe cases of overfitting, early stopping has the additional advantage that the number of training epochs will be regulated automatically.
Instead of comparing training runs for different number of epochs, early stopping allows to simply set the number of epochs to a desired maximum value.

What might be a bit unintuitive is that the training runs might now end very rapidly
This might spark the question: have we really reached an optimum yet?
And often the answer to this is "no", which is why early stopping frequently is combined with other approaches to hinder overfitting from happening.
Overfitting means that a model (seemingly) performs better on seen data compared to unseen data. One then often also says that it does not "generalize" well.
Techniques to avoid overfitting, or to improve model generalization, are termed **regularization techniques** and we will come back to this in **episode 4**.


## BatchNorm: the "standard scaler" for deep learning
A very common step in classical machine learning pipelines is to scale the features, for instance by using sckit-learn's `StandardScaler`.
This can in principle also be done for deep learning.
An alternative, more common approach, is to add **BatchNormalization** layers which will learn how to scale the input values.
Similar to dropout, batch normalization is available as a network layer in keras and can be added to the network in a similar way.
It does not require any additional parameter setting. 

~~~
from tensorflow.keras.layers import BatchNormalization
~~~
{: .language-python} 

> ## Excercise: Add a BatchNormalization layer as the first layer to your neural network.
> Look at the [documentation of the batch normalization layer](https://keras.io/api/layers/normalization_layers/batch_normalization/). Add this as a first layer to the model we defined above. Then, train the model and compare the performance to the model without batch normalization.
>
> > ## Solution
> > ~~~
> > def create_nn():
> >     # Input layer
> >     inputs = keras.layers.Input(shape=(X_data.shape[1],), name='input')
> > 
> >     # Dense layers
> >     layers_dense = keras.layers.BatchNormalization()(inputs)
> >     layers_dense = keras.layers.Dense(100, 'relu')(layers_dense)
> >     layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)
> > 
> >     # Output layer
> >     outputs = keras.layers.Dense(1)(layers_dense)
> > 
> >     # Defining the model and compiling it
> >     return keras.Model(inputs=inputs, outputs=outputs, name="model_batchnorm")
> > 
> > model = create_nn()
> > model.compile(loss='mse', optimizer='adam', metrics=[keras.metrics.RootMeanSquaredError()])
> > model.summary()
> > ~~~
> > {: .language-python}
> > 
> > Which is then trained as above:
> > ~~~
> > history = model.fit(X_train, y_train,
> >                     batch_size = 32,
> >                     epochs = 1000,
> >                     validation_data=(X_val, y_val),
> >                     callbacks=[earlystopper],
> >                     verbose = 2)
> > 
> > history_df = pd.DataFrame.from_dict(history.history)
> > sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
> > plt.xlabel("epochs")
> > plt.ylabel("RMSE")
> > ~~~
> > {: .language-python}      
> > 
> > ![Output of plotting sample](../fig/03_training_history_5_rmse_batchnorm.png)
> {:.solution}
{:.challenge}



## Run on test set and compare to naive baseline
It seems that no matter what we add, the overall loss does not decrease much further (we at least avoided overfitting though!).
Let's again plot the results on the test set:
~~~
y_test_predicted = model.predict(X_test)

plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(y_test_predicted, y_test, s=10, alpha=0.5)
plt.xlabel("predicted sunshine hours")
plt.ylabel("true sunshine hours")
~~~
{: .language-python} 

![Output of plotting sample](../fig/03_regression_test_5_dropout_batchnorm.png)

Well... certainly not perfect. But how good or bad is this? Maybe not good enough to plan your picnic for tomorrow.
But let's better compare it to a naive baseline.

> ## Exercise: Create a similar scatter plot as above for a reasonable baseline
>
> What can we take as a baseline? 
> Maybe the simplest prediction to make would be to say: Tomorrow we will have the same number of sunshine hours as today.
> Let's compare to this.
> 
> > ## Solution
> > We can here just take the `BASEL_sunhine` column of our data, because this contains the sunshine hours from one day before what we have as a label.
> > ~~~
> > plt.figure(figsize=(5, 5), dpi=100)
> > plt.scatter(X_test["BASEL_sunshine"], y_test, s=10, alpha=0.5)
> > plt.xlabel("sunshine hours yesterday")
> > plt.ylabel("true sunshine hours")
> > ~~~
> > {: .language-python} 
> > 
> > ![Output of plotting sample](../fig/03_regression_test_5_naive_baseline.png)
> {:.solution}
{:.challenge}


# Outlook
Correctly predicting tomorrow's sunshine hours is apparently not that simple. 
Our models get the general trends right, but still predictions vary quiet a bit and can even be far off.

> ## Open question: What could be next steps to further improve the model?
> 
> With unlimited options to modify the model architecture or to play with the training parameters, deep learning can trigger very extensive hunting for better and better results.
> Usually models are "well behaving" in the sense that small chances to the architectures also only result in small changes of the performance (if any).
> It is often tempting to hunt for some magical settings that will lead to much better results. But do those settings exist?
> Applying common sense is often a good first step to make a guess of how much better *could* results be. 
> In the present case we might certainly not expect to be able to reliably predict sunshine hours for the next day with 5-10 minute precision.
> But how much better our model could be exactly, often remains difficult to answer.
> 
> * What changes to the model architecture might make sense to explore?
> * Ignoring changes to the model architecture, what might notably improve the prediction quality?
> 
> > ## Solution
> > This is on open question. And we don't actually know how far one could push this sunshine hour prediction (try it out yourself if you like! We're curious!).
> > But there is a few things that might be worth exploring.
> > 
> > Regarding the model architecture:
> > * In the present case we do not see a magical silver bullet to suddenly boost the performance. But it might be worth testing if *deeper* networks do better (more layers).
> > 
> > Other changes that might impact the quality notably:
> > * The most obvious answer here would be: more data! Even this will not always work (e.g. if data is very noisy and uncorrelated, more data might not add much).
> > * Related to more data: use data augmentation. By creating realistic variations of the available data, the model might improve as well.
> > * More data can mean more data points (you can test it yourself by taking more than the 3 years we used here!)
> > * More data can also mean more features! What about adding the month? 
> > * The labels we used here (sunshine hours) are highly biased, many days with no or nearly no sunshine but few with >10 hours. Techniques such as oversampling or undersampling might handle such biased labels better.
> > Another alternative would be to not only look at data from one day, but use the data of a longer period such as a full week. 
> > This will turn the data into time series data which in turn might also make it worth to apply different model architectures...
> > 
> {:.solution}
{:.challenge}
