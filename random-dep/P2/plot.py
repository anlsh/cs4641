import matplotlib.pyplot as plt
from temp import * # where the variables are defined

datasetPercentage = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]

plt.figure()
plt.title('Randomized Hill Climbing: Performance x Training Set Size\n(5500 training iterations)')
plt.plot(datasetPercentage, test_error, '-', label='test error')
plt.plot(datasetPercentage, train_error, '-', label='train error')
plt.legend()
plt.xlabel('Percentage of Dataset')
plt.ylabel('Sum of Squares Error')
plt.show()


plt.figure()
plt.title('Randomized Hill Climbing: Train/Test time x Training Set Size\n(5500 training iterations)')
plt.plot(datasetPercentage, test_time, '-', label='test time')
plt.plot(datasetPercentage, train_time, '-', label='train time')
plt.legend()
plt.xlabel('Percentage of Dataset')
plt.ylabel('Train/Test time')
plt.show()
