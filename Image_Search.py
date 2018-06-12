# Databricks notebook source
import cv2
import matplotlib.pyplot as plt
import re
from skimage import img_as_ubyte
from skimage.color import rgba2rgb

from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from azure.cognitiveservices.search.imagesearch.models import ImageType, ImageAspect, ImageInsightModule
from msrest.authentication import CognitiveServicesCredentials


# COMMAND ----------

# MAGIC %md
# MAGIC Images and search sites can be dynamic, here I am using a list and dict; many other options could be used to instead.

# COMMAND ----------

products = [{'Name': 'PAM Original 6 OZ', 'File': 'PAM_Original_6_OZ.jpg' }]

sites = ['walmart.com','target.com']

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Variable used in code: Image Folder and Image Subscription Key

# COMMAND ----------

IMAGES_FOLDER = "/dbfs/mnt/images/"

IMG_SEARCH_SUBSCRIPTION_KEY = "<BingSearchKey>"


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Create Image Directory - only needs to run once

# COMMAND ----------

# MAGIC %fs
# MAGIC mkdirs "/mnt/images/"

# COMMAND ----------

# MAGIC %md
# MAGIC Mount Blob Storage
# MAGIC 
# MAGIC https://docs.azuredatabricks.net/spark/latest/data-sources/azure/azure-storage.html

# COMMAND ----------

dbutils.fs.mount(source = "wasbs://<container>@<storage-account>.blob.core.windows.net/",mount_point = "/mnt/images/",extra_configs = {"fs.azure.account.key.<storage-account>.blob.core.windows.net": "<storage-account-key>"})

# COMMAND ----------

# MAGIC %md
# MAGIC ###Functions

# COMMAND ----------

# MAGIC %md
# MAGIC Retrieve a list of image search results using Cognitive Services Image Search API. Arguments: search query string and subscription key.

# COMMAND ----------

def retrieve_images(search,key):
  client = ImageSearchAPI(CognitiveServicesCredentials(key))
 
  try:
    image_results = client.images.search(query=search,freshness='Month')
    
    print("Search images for query " + search)
    return image_results
  except Exception as err:
    print("Encountered exception. {}".format(err))
    return null


# COMMAND ----------

# MAGIC %md
# MAGIC Return single image from search results - for simplicity bring back first item.

# COMMAND ----------

def retrieve_first_img_url(image_results):
  if image_results.value:
    first_image_result = image_results.value[0] #grab first image from search results
    print("Image result count: {}".format(len(image_results.value)))
    print(first_image_result.content_url)
    
    url = first_image_result.content_url 
    #remove extra args from url, just grab upto image
    url_clean = re.match(r'.*(png|jpg|jpeg)',url,re.M|re.I).group(0)
    print(url_clean)
    
    return url_clean

# COMMAND ----------

# MAGIC %md
# MAGIC Convert url into OpenCV image - this is important for image comparison using OpenCV

# COMMAND ----------

def url_to_image(url): 
    img = io.imread(url) #read url
    img = rgba2rgb(img) #remove alpha
    cv_image =cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_RGB2BGR) #convert from skimage to opencv

	# return the image
    return cv_image

# COMMAND ----------

# MAGIC %md
# MAGIC plot function to simplify code

# COMMAND ----------

def plot_img(figtitle,subtitle,img1,img2,site):
  fig = plt.figure(figtitle, figsize=(10, 5))
  plt.suptitle(subtitle,fontsize=24)
  ax = fig.add_subplot(1, 2, 1)
  ax.set_title("Base",fontsize=12)
  plt.imshow(img1)
  plt.axis("off")
  ax = fig.add_subplot(1, 2, 2)
  ax.set_title(site,fontsize=12)
  plt.imshow(img2)
  plt.axis("off")

  display(plt.show())

# COMMAND ----------

#Use the first product and first site

product = products[0]
site = sites[0]

print("Product: " + product['Name'] + "\nSite: " + site)

# COMMAND ----------

# MAGIC %md
# MAGIC Retrieve img1 from blob storage (mounted dir) and img2 from Image Search API

# COMMAND ----------

print(product['Name'] + ":" + site)

img1 = IMAGES_FOLDER  + product['File']
orig_img =  cv2.imread(img1)

# query = "site: website.com search product string

image_results = retrieve_images("site: " + site + " " +  product['Name'],IMG_SEARCH_SUBSCRIPTION_KEY)
img2 = retrieve_first_img_url(image_results)
    
comp_img = url_to_image(img2)


# COMMAND ----------

plot_img("Image Compare" + site,"Original Images",cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB),cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB),site)


# COMMAND ----------

dbutils.fs.unmount("dbfs:/mnt/images/")
