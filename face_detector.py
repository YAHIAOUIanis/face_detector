#!/usr/bin/env python
# coding: utf-8

# In[43]:


import cv2


# In[44]:


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[45]:


img = cv2.imread('image.jpg')


# In[46]:


faces = face_cascade.detectMultiScale(img, 1.1, 4)


# In[47]:


for (x, y, w, h) in faces: 
  cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imwrite("face_detected.png", img) 
print('Successfully saved')


# In[ ]:





# In[ ]:




