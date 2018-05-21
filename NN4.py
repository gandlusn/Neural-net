import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
dib = pd.read_csv('pima-indians-diabetes.csv')
print(dib.head());
print(dib.columns);
cols_to_norm = ["Number_pregnant","Glucose_concentration","Blood_pressure","Triceps","Insulin","BMI","Pedigree"]

dib[cols_to_norm] = dib[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))
print(dib.head());

# now convert every Column in to tensor flow Input Feature

num_preg = tf.feature_column.numeric_column('Number_pregnant');
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration');
dias_press = tf.feature_column.numeric_column('Blood_pressure');
tricep = tf.feature_column.numeric_column('Triceps');
insulin = tf.feature_column.numeric_column('Insulin');
bmi = tf.feature_column.numeric_column('BMI');
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree');
age= tf.feature_column.numeric_column('Age');

assign_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
# we can use the below one if their are undeeads of categories inside the category column
#assign_group = tf.feature_column.categorical_column_with_hash_bucket('Group',hash_bucket_size=4);
init = tf.global_variables_initializer(); # we intailze all variables before we run
# now to convert the continuous column in to categorical type column we can use bucketrized column to covert 
#whole column in to classes all the values between those values.
age_bucket = tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])
features = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,assign_group,age_bucket]

#TRAIN Test Split

x_data = dib.drop('Class',axis=1)
labels = dib['Class'];

x_train, x_test, y_train,y_test = train_test_split(x_data,labels,test_size=0.3);


# create a input function

#input_function = tf.estimator.inputs.numpy_input_fn({'x':x_data},labels,batch_size=10,num_epochs = 1000,shuffle=True);
#we used Pandas data frames to store data so we will use pandas input function
input_function = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs = 10000,shuffle=True);

# create the estimator
# so here we use the featire column a input which will apply the normalization on data after we sumbit the data to the model
estimator = tf.estimator.LinearClassifier(feature_columns=features,n_classes=2);
 
estimator.train(input_fn=input_function,steps=10000);


# testing 
eval_input = tf.estimator.inputs.pandas_input_fn(x=x_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False);
results = estimator.evaluate(eval_input);
print(results);

# Prediction
pred_input = tf.estimator.inputs.pandas_input_fn(x=x_test,batch_size=10,num_epochs=1,shuffle=False);
Predictions = estimator.predict(pred_input);
my_pred = list(Predictions);

print("Final Predcitions : --------------------------",my_pred);