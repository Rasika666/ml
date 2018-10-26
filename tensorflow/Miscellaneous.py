#deep net with Abstraction
# TF learn
# keras
# TF-Slim
# Layers
# Estomator API
# layers

################ grap the data ##################
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as  tf 
from tensorflow import estimator
from sklearn.metrics import confusion_matrix, classification_report

wine_data = load_wine()
#print(wine_data['DESCR'])

feat_data = wine_data['data']
label = wine_data['target']

# split data 
x_train, x_test, y_train, y_test = train_test_split(feat_data, label, test_size = 0.3,random_state=101)

# preprocessing
scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)
print(feat_data)


feat_cols = [tf.feature_column.numeric_column('x',shape=[13])]
deep_model = estimator.DNNClassifier(hidden_units=[13,13,13], feature_columns=feat_cols,n_classes=3,optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01))


input_fn = estimator.inputs.numpy_input_fn(x={'x':scaled_x_train},y=y_train, batch_size=10,num_epochs=5, shuffle=True)
deep_model.train(input_fn, steps=500)

input_fn_eval = estimator.inputs.numpy_input_fn(x={'x':scaled_x_test}, batch_size=10,num_epochs=5, shuffle=False)

preds = list(deep_model.predict(input_fn=input_fn_eval))

predictions = [p['class_ids'][0] for p in preds]

#print(predictions)

#evaluate the predected ones
#print(classification_report(y_test, predictions))