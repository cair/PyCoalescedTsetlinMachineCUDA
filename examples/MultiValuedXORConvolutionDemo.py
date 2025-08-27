import numpy as np
import keras
from keras.datasets import imdb
from time import time
from sklearn.metrics import f1_score

from PyCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D, MultiClassTsetlinMachine

clauses = 2
T = 20
s = 1.0

sequence_length = 10
number_of_examples = 10000
max_int_value = 5
max_neutral_value = 5

X_train = np.zeros((number_of_examples, 1, sequence_length, max_int_value + max_neutral_value), dtype=np.uint32)
Y_train = np.zeros(number_of_examples, dtype=np.uint32)
for i in range(number_of_examples):
	x_1 = np.random.randint(max_int_value, dtype=np.uint32)
	x_2 = np.random.randint(max_int_value, dtype=np.uint32)

	pattern_position = np.random.randint(sequence_length-1, dtype=np.uint32)

	for j in range(sequence_length):
		if j == pattern_position:
			X_train[i, 0, j, x_1] = 1
		elif j == pattern_position + 1:
			X_train[i, 0, j, x_2] = 1
		else:
			X_train[i, 0, j, max_int_value + np.random.randint(max_neutral_value, dtype=np.uint32)] = 1

	if ((x_1 % 2 == 0) and (x_2 % 2 == 0)) or ((x_1 % 2 == 1) and (x_2 % 2 == 1)):
		Y_train[i] = 0
	else:
		Y_train[i] = 1

	print(Y_train[i], X_train[i])

gregre

tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (2, 1))
for i in range(epochs):
	start_training = time()
	for batch in range(batches):
		tm.fit(X_train[batch*batch_size_train:(batch+1)*batch_size_train], Y_train[batch*batch_size_train:(batch+1)*batch_size_train], epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	Y_test_predicted = np.zeros(0, dtype=np.uint32)
	for batch in range(batches):
		Y_test_predicted = np.concatenate((Y_test_predicted, tm.predict(X_test[batch*batch_size_test:(batch+1)*batch_size_test])))
	result_test = 100*(Y_test_predicted == Y_test[:batch_size_test*batches]).mean()
	f1_test = f1_score(Y_test[:batch_size_test*batches], Y_test_predicted, average='macro')*100
	stop_testing = time()

	Y_train_predicted = np.zeros(0, dtype=np.uint32)
	for batch in range(batches):
		Y_train_predicted = np.concatenate((Y_train_predicted, tm.predict(X_train[batch*batch_size_train:(batch+1)*batch_size_train])))
	result_train = 100*(Y_train_predicted == Y_train[:batch_size_train*batches]).mean()

	f1_train = f1_score(Y_train[:batch_size_train*batches], Y_train_predicted, average='macro')*100

	print("#%d F1 Test: %.2f%% F1 Train: %.2f%% Accuracy Test: %.2f%% Accuracy Train: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, f1_test, f1_train, result_test, result_train, stop_training-start_training, stop_testing-start_testing))