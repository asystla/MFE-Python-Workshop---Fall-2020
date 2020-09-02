#!/usr/bin/env python
# coding: utf-8

# # Python
# 
# Python is a high-level, interpreted, object-oriented, and multi-purpose programming language. 

# # Understanding The Environment

# In[ ]:


100*12.5/5


# In[ ]:


100*12.5/5


# In[ ]:


# Comment type 1
'Comment type 2'
"Comment type 3"

""" Commments (doc string)

dadasdas
asdada
"""

print('Hello World 123!')


# In[ ]:


print("MFE Class of {}".format(2021))


# # Numerical Operations

# In[ ]:


# Assigning values to variables 'x' and 'y'
x = 7
y = 10
x,y


# In[ ]:


# Python is a case sensitive language (X != x)
A = 17
a = 15
A,a


# In[ ]:


# Addition
x+y


# In[ ]:


# Division
y/x


# In[ ]:


# Quotient
y//x


# In[ ]:


# Remainder
y%x


# In[ ]:


# Power
x**y


# In[ ]:


# Variables can be anything (like below) 
mfe_class_of_2021 = 80
print(mfe_class_of_2021)


# # Data Types

# ### Arithmetic Data Types

# In[ ]:


# Type tells us the variable type inside the brackets
x = 10
type(x)


# In[ ]:


# variable x can take any value - integer, boolean, string, complex numbers
x = 2
y = 2
print(x == y)


# In[ ]:


print(x>y)


# In[ ]:


print(x!=y)
# Talk about case sensitivity of True/False and != is the not equal in py


# ### String Data Type

# In[ ]:


x = "MFE Class of 2021"
type(x)


# In[ ]:


x = "A B C D E"
dir(x)


# # Data Structures
# 
# Data structures are a collection of related data. There are 4 built in data structures in Python - List, Tuple, Dictionary, and Set. 
# 
# ### List

# In[ ]:


# A list always starts and ends with the square brackets []
int_list = [1,2,3]
print(type(int_list))
int_list


# In[ ]:


# A list can store other lists and any data type (no restrictions)
mix_list = [[1,2,3],"MFE Class of 2020",3.14,5+4j,True]
mix_list


# In[ ]:


# Method 1 of adding a variable to the end of a list
mix_list.append('Hello World')
mix_list


# In[ ]:


# Something very interesting happens when you try to print(list.append())
print(mix_list.append('Hello World'))


# In[ ]:


# Method 2 of adding a variable to a list (the more preferred way)
mix_list = [[1,2,3],"MFE Class of 2020",3.14,5+4j,True]
mix_list.insert(5,'Hello World') # synatax is insert(index,variable)
mix_list


# In[ ]:


# Printing the length of a list
print(mix_list.__len__())
print(len(mix_list))


# In[ ]:


# Removing an item from list
int_list = [1,2,2,3,4,5]
int_list.remove(2)


# In[ ]:


# Another way to remove an item from list
del int_list[1]
int_list


# In[ ]:


# Referencing items in a list
int_list = [10,11,12,13,14]
int_list[0]


# In[ ]:


int_list[2:]


# In[ ]:


int_list[-1]


# In[ ]:


# Renaming item in a list
int_list[0] = 100
int_list


# In[ ]:


# Memory allocation (very important concept)
list1 = [1,2,3,4,5]
list2 = list1
list2[0] = 100
list1


# In[ ]:


# Understanding the object and it's attributes
dir(int_list)


# ### Tuple

# In[ ]:


# A tuple always starts and ends with the round brackets ()
int_tuple = (1,2,3)
print(type(int_tuple))
int_tuple


# In[ ]:


int_tuple[0] = 100


# ### Dictionary

# In[ ]:


# Dictionary is like an address book, in Py it is specified by a key:value
dictionary = {"student":['student1','student2','student3'],"id":('id1','id2','id3'),"workshop_grade":['A','A-','A+']}
dictionary


# In[ ]:


dictionary.keys()


# In[ ]:


dictionary.values()


# In[ ]:


dictionary['workshop_grade']


# In[ ]:


dictionary.items()


# # NumPy

# In[ ]:


import numpy
import numpy as np


# ### Creating a NumPy array

# In[ ]:


# Creating a numpy array from a list
my_array = np.array([1.,2,3])
my_array


# In[ ]:


# Creating an array using linspace (returns evenly spaced numbers between two intervals)
my_array = np.linspace(start = 0,stop = 10,num = 101)
my_array


# In[ ]:


# Creating an array using arange (returns array with start, stop-step and length of array ~ (stop-start)//step)
my_array = np.arange(start = 0,stop = 10,step = 3)
my_array


# In[ ]:


# Combining multiple data types into Numpy array - observe what happens in the output
my_array = np.array(['ab',2,True])
my_array


# ### NumPy Matrix

# In[ ]:


# Reshaping a regular array into a matrix
my_array = np.linspace(0,10,12)
my_matrix = my_array.reshape(4,3)
my_matrix


# In[ ]:


# A numpy matrix is essentially a stack of numpy arrays
my_matrix[0]


# In[ ]:


# Creating a numpy matrix of zeros
my_matrix = np.zeros((5,5))
my_matrix


# ### Selection and Broadcasting

# In[ ]:


# Selecting range of items
my_array = np.linspace(0,10,101)
my_array[90:]


# In[ ]:


# Selecting specific items
my_array[[10,15,20]]


# In[ ]:


# Broadcasting
my_array[90:] = 100
my_array


# In[ ]:


# Selecting items in a matrix
my_matrix = np.linspace(0,10,9).reshape(3,3)
my_matrix


# In[ ]:


# Selecting middle column of all rows
my_matrix[0:,1:]


# In[ ]:


# A new way of selecting items
my_array = np.linspace(-1,1,3)
print(my_array)
my_array>0
bool_array = [True,False,False]
my_array[bool_array]


# In[ ]:


# A new way of selecting items
my_matrix = np.linspace(0,10,9).reshape(3,3)
print(my_matrix)
my_matrix>5


# ### NumPy Operations

# In[ ]:


dir(np)


# In[ ]:


np.array([1,2,3])


# In[ ]:


np.asarray([1,2,3])


# In[ ]:


# Max value of numpy array
my_array = np.linspace(-10,10,21)
print(my_array)
my_array.max()


# In[ ]:


# Argmax value of numpy array
my_array = np.linspace(-10,10,21)
my_array.argmax()


# In[ ]:


# Alternate ways to compute max, argmax, sum, cumsum
my_array.sum() # We can do this because the numpy array has this attribute


# In[ ]:


# Exponentials
np.exp(my_array)


# In[ ]:


# Warning of log
np.mean(my_array)


# In[ ]:


# Stacking numpy arrays
my_array1 = np.array([1,2,3])
my_array2 = np.array([4,5,6])
print(my_array1,my_array2)
print(np.hstack((my_array1,my_array2)))


# In[ ]:


# Vertically stacking arrays
print(np.vstack((my_array1,my_array2)))


# In[ ]:


# Sorting your array
my_array = np.random.rand(10)
my_array.sort()


# ### Matrix Operations

# In[ ]:


# Matrix addition
my_matrix1 = np.linspace(0,10,20).reshape(4,5)
my_matrix2 = np.linspace(30,40,20).reshape(4,5)
my_matrix1


# In[ ]:


my_matrix2


# In[ ]:


# Matrix addition
my_matrix1+my_matrix2


# In[ ]:


# Adding constant
my_matrix1+100


# In[ ]:


# Matrix multiplication two 4x5 matrices 
np.matmul(my_matrix1,my_matrix2)


# In[ ]:


# What happens when we compare a matrix to a constant
mat = np.linspace(0,10,16).reshape(4,4)
print(mat)
mat>=5


# In[ ]:


# Identity matrix
identity_matrix = np.identity(10)
identity_matrix


# In[ ]:


# Diagonal of matrix
np.diag(identity_matrix)


# In[ ]:


# The Linear Algebra library of NumPy (advanced operations)
get_ipython().run_line_magic('pinfo', 'np.linalg')


# ### Random Numbers

# In[ ]:


# Generating random numbers using Numpy's random function
get_ipython().run_line_magic('pinfo', 'np.random')


# In[ ]:



# Creates an array of random numbers from [0,1) - uniform distribution
random_array = np.random.rand(10)
random_array


# In[ ]:


# Generating a random matrix (Uniform distribution)
random_matrix = np.random.rand(5,5)
random_matrix


# In[ ]:


# Generating random integers
np.random.randint(low=0,high=10,size=100)


# In[ ]:


# Setting seed (very important in future)
np.random.seed(123)
np.random.rand(5)


# In[ ]:


np.random.seed(123)
np.random.rand(5)


# # Take Home Questions
# 
# To keep you thinking about Python

# - Create a dictionary with five stocks (any five stocks from S&P 500). The keys of the dictionary that you will have to populate are - Name, Company Sector, Price, Market Cap (USD Mn), and Price-Earnings Ratio. You should find such data on Yahoo Finance. For example (https://finance.yahoo.com/quote/TSLA/).
# - Using NumPy's arange and linspace, generate the same arrays where start = 0, stop = 10 and the array has size of 101 elements.
# - Generate 50 random numbers from [0,1) using np.random.rand function and calculate the mean, median, and variance.
# 
# - Now reshape the above array into a 10x5 matrix (10 rows and 5 columns). calculate the same statistics for each **column**.
# - Generate 50 random numbers from [0,1). Split this array into two equal arrays of size 25 each. Now reshape the two new arrays into 5x5 matrices and multiply both of them. Calculate the determinant of the new matrix (check wiki if you don't know what it means). *Hint - you will find the determinant function part of the numpy linalg function (the linear algebra library)*
