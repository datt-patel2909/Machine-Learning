import statistics
import numpy as np


data = [12, 15, 20, 20, 25, 30, 35, 40, 40, 40, 45]

print("Dataset:", data)


mean = statistics.mean(data)
print("Mean:", mean)


mode = statistics.mode(data)
print("Mode:", mode)


median = statistics.median(data)
print("Median:", median)


variance = statistics.variance(data)
print("Variance:", variance)


std_dev = statistics.stdev(data)
print("Standard Deviation:", std_dev)

Q1 = np.percentile(data, 25)
Q2 = np.percentile(data, 50)
Q3 = np.percentile(data, 75)

print("Q1 (First Quartile):", Q1)
print("Q2 (Second Quartile / Median):", Q2)
print("Q3 (Third Quartile):", Q3)


IQR = Q3 - Q1
print("Interquartile Range (IQR):", IQR)
