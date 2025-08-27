import numpy as np
import keras
from keras.datasets import imdb
from time import time
from sklearn.metrics import f1_score

from PyCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D, MultiClassTsetlinMachine

clauses = 4
T = 20
s = 1.0

sequence_length = 2
number_of_examples = 10000
max_int_value = 2
max_neutral_value = 0
epochs = 10
noise = 0.1

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

	if np.random.rand() <= noise:
        Y_train[i] = 1 - Y_train[i]

print(Y_train[0:10])
print(X_train[0:10])

X_test = np.zeros((number_of_examples, 1, sequence_length, max_int_value + max_neutral_value), dtype=np.uint32)
Y_test = np.zeros(number_of_examples, dtype=np.uint32)
for i in range(number_of_examples):
	x_1 = np.random.randint(max_int_value, dtype=np.uint32)
	x_2 = np.random.randint(max_int_value, dtype=np.uint32)

	pattern_position = np.random.randint(sequence_length-1, dtype=np.uint32)

	for j in range(sequence_length):
		if j == pattern_position:
			X_test[i, 0, j, x_1] = 1
		elif j == pattern_position + 1:
			X_test[i, 0, j, x_2] = 1
		else:
			X_test[i, 0, j, max_int_value + np.random.randint(max_neutral_value, dtype=np.uint32)] = 1

	if ((x_1 % 2 == 0) and (x_2 % 2 == 0)) or ((x_1 % 2 == 1) and (x_2 % 2 == 1)):
		Y_test[i] = 0
	else:
		Y_test[i] = 1

tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (1, 2))
for i in range(epochs):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	Y_test_predicted =tm.predict(X_test)
	stop_testing = time()
	result_test = 100*(Y_test_predicted == Y_test).mean()
	f1_test = f1_score(Y_test, Y_test_predicted, average='macro')*100

	Y_train_predicted = tm.predict(X_train)
	result_train = 100*(Y_train_predicted == Y_train).mean()
	f1_train = f1_score(Y_train, Y_train_predicted, average='macro')*100

	print("#%d F1 Test: %.2f%% F1 Train: %.2f%% Accuracy Test: %.2f%% Accuracy Train: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, f1_test, f1_train, result_test, result_train, stop_training-start_training, stop_testing-start_testing))