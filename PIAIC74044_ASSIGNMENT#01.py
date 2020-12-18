#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the numpy package under the name np
import numpy as np


# In[2]:


#Create a null vector of size 10
null_vector = np.zeros(10)
null_vector


# In[3]:


#Create a vector with values ranging from 10 to 49
a = np.arange(10, 50)
a


# In[4]:


#Find the shape of previous array in question 3
np.shape(a)


# In[5]:


#Print the type of the previous array in question 3
a.dtype


# In[6]:


#Print the numpy version and the configuration
np.__version__


# In[7]:


np.show_config()


# In[8]:


#Print the dimension of the array in question 3
a.ndim


# In[9]:


#Create a boolean array with all the True values
A = np.array([4, 5, 6, 7, 8, 9])

A >= 4


# In[10]:


#Create a two dimensional array
B = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
B


# In[11]:


#Create a three dimensional array
C = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
C


# In[12]:


##Difficulty Level Easy
#Reverse a vector (first element becomes last)
D = np.arange(10)
D


# In[13]:


D[::-1]


# In[14]:


#Create a null vector of size 10 but the fifth value which is 1
E = np.zeros(10)
E[4] = 1
E


# In[15]:


#Create a 3x3 identity matrix
F = np.identity(3)
F


# In[16]:


#Convert the data type of the given array from int to float
arr = np.array([1, 2, 3, 4, 5])
arr.astype('float')


# In[17]:


#Multiply arr1 with arr2
arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
arr3 = arr1 * arr2
arr3


# In[18]:


#Make an array by comparing both the arrays provided above
arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
comparison = arr1 == arr2
equal_arr = comparison.all()
equal_arr


# In[83]:


#Extract all odd numbers from arr with values(0-9)
arr = np.arange(10)
Z = arr[arr % 2 == 1]
Z


# In[85]:


#Replace all odd numbers to -1 from previous array
Z[0:5] = -1
Z


# In[22]:


#Replace the values of indexes 5,6,7 and 8 to 12
arr = np.arange(10)
arr[5:9]=12
arr


# In[23]:


#Create a 2d array with 1 on the border and 0 inside
X = np.ones((5,5))
X[1:-1,1:-1] = 0
X


# In[24]:


###Difficulty Level Medium
#Replace the value 5 to 12
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[1][1] = 12
arr2d


# In[25]:


#Convert all the values of 1st array to 64
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0] = 64
arr3d


# In[26]:


#Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it
A = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
A[0][0]


# In[27]:


#Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it
A[1][1]


# In[28]:


#Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows
A[:, :4]


# In[29]:


#Create a 10x10 array with random values and find the minimum and maximum values
Y = np.random.randn(10, 10)
Y


# In[30]:


Y.max()


# In[31]:


Y.min()


# In[32]:


#Find the common items between a and b
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
c = np.intersect1d(a, b)
c


# In[33]:


#Find the positions where elements of a and b match
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a==b)


# In[71]:


#Find all the values from array data where the values from array names are not equal to Will
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
names != 'Will'


# In[36]:


#Find all the values from array data where the values from array names are not equal to Will and Joe
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
names != 'Will' 


# In[54]:


###Difficulty Level Hard
#Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.
Z = np.arange(1, 16).reshape(5, 3)
Z


# In[55]:


#Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.
Y = np.arange(1, 17).reshape(2, 2, 4)
Y


# In[64]:


#Swap axes of the array you created in Question 32
Y.swapaxes(1, 2)


# In[40]:


#Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0
Y = np.arange(10)
np.sqrt(Y)


# In[ ]:


#Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays


# In[44]:


#Find the unique names and sort them out!
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# In[45]:


np.sort(names)


# In[53]:


#From array a remove all items present in array b
a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
np.delete(a, 4)


# In[47]:


#Following is the input NumPy array delete column two and insert following new column in its place
sampleArray = np.array([[34,43,73], [82,22,12], [53,94,66]])
newColumn = np.array([[10,10,10]])
sampleArray[:, 2:] = 10
sampleArray


# In[48]:


#Find the dot product of the above two matrix
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
z = np.dot(x, y)
z


# In[49]:


#Generate a matrix of 20 random values and find its cumulative sum
z = np.random.randn(4, 5)
z


# In[50]:


z.cumsum()


# In[ ]:




