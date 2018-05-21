import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(0.0,10,1000000);
noise = np.random.randn(len(x_data));

y_true = (0.5*x_data) + 5 + noise;

feature_cols = [tf.feature_column.numeric_column('x',shape=[1])];
estimator = tf.estimator.LinearRegressor(feature_columns = feature_cols)
x_train, x_test, y_train,y_test = train_test_split(x_data,y_true,test_size=0.3);

input_function = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,shuffle=True,
                                                    num_epochs=100)

test_input_function = tf.estimator.inputs.numpy_input_fn({'x':x_test},y_test,batch_size=8,shuffle=True,
                                                         num_epochs=100)

estimator.train(input_fn = input_function,steps=1000); 
print("train metrics")
train_metrics = estimator.evaluate(input_fn = input_function,steps=100);
print("test metrics")
eval_metrics = estimator.evaluate(input_fn = test_input_function,steps=100);

print("Training Data Metrics",train_metrics)

print("Testing Data Metrics",eval_metrics)

#to get the pridicted values
print("predictions------------------------")
Prediction=[];
in_test = tf.estimator.inputs.numpy_input_fn({'x':x_data},shuffle=False);
for pred in estimator.predict(input_fn=in_test):
    Prediction.append(pred['predictions'])# this pred has a key value 'predictions' which will give us output of the data
print("Predictions ---------------------------------")
print("okaykjjntjrtnhjrny")
plt.scatter(x_data,y_true);
plt.plot(x_data,Prediction,'r*');
plt.show();