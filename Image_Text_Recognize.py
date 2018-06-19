# Databricks notebook source
from skimage import io
import simplejson as json
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from pyspark.sql import *
import re
from skimage import img_as_float
from skimage import img_as_ubyte
from skimage.color import rgba2rgb
import http.client, urllib.request, urllib.parse, urllib.error, base64


from azure.cognitiveservices.vision.computervision import ComputerVisionAPI
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from azure.cognitiveservices.search.imagesearch.models import ImageType, ImageAspect, ImageInsightModule
from msrest.authentication import CognitiveServicesCredentials



# COMMAND ----------

# MAGIC %md
# MAGIC Images and search sites can be dynamic, here I am using a list and dict; many other options could be used to instead.

# COMMAND ----------

products = [{'Name': 'PAM Original Cooking Spray, 6 Ounce', 'File': 'PAM_Original_6_OZ.jpg' }]

sites = ['walmart.com','target.com']

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Variable used in code: Image Folder and Subscription Keys, and Computer Visions Location 'region'
# MAGIC 
# MAGIC 
# MAGIC You can see additional Computer Vision features by visiting:
# MAGIC 
# MAGIC https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/

# COMMAND ----------


IMAGES_FOLDER = "/dbfs/mnt/images/"

IMG_SEARCH_SUBSCRIPTION_KEY = "<BingSearchKey>"

COMP_VIS_SUBSCRIPTION_KEY = "<ComputerVisionKey>"
COMPUTERVISION_LOCATION = os.environ.get("COMPUTERVISION_LOCATION", "<ComputerVisionRegion>")


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Create Image Directory - only needs to run once

# COMMAND ----------

#%fs
#mkdirs "/mnt/images/"

# COMMAND ----------

# MAGIC %md
# MAGIC Mount Blob Stroage
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
    image_results = client.images.search(query=search)
    
    print("Search images for query " + search)
    return image_results
  except Exception as err:
    print("Encountered exception. {}".format(err))
    return null


# COMMAND ----------

# MAGIC %md
# MAGIC Return simgle image from search results - for simplicity bring back first item.

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

# MAGIC %md
# MAGIC #### Text Retrieve Functions

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC retriev_text_from_img - Uses SDK function recognize_text_in_stream()
# MAGIC 
# MAGIC 
# MAGIC _Recognize Text operation. When you use the Recognize Text interface, the response contains a field called “Operation-Location”. The “Operation-Location” field contains the URL that you must use for your Get Handwritten Text Operation Result operation_
# MAGIC 
# MAGIC [Computer Vision SDK Python Docs](https://docs.microsoft.com/en-us/python/api/azure.cognitiveservices.vision.computervision?view=azure-python)
# MAGIC 
# MAGIC 
# MAGIC The code then captures the JSON results using the 'Operation-Location' URL.

# COMMAND ----------

def retrieve_text_from_img(img):
    client = ComputerVisionAPI(COMPUTERVISION_LOCATION, CognitiveServicesCredentials(COMP_VIS_SUBSCRIPTION_KEY))
    
    #raw - returns the direct response alongside the deserialized response
    with open(os.path.join(IMAGES_FOLDER, img), "rb") as image_stream:
        txt_analysis2=client.recognize_text_in_stream(image_stream,raw=True)
    
    #give Computer Vision some time to process image, could also be a while loop checking status (20s is arbitrary) 
    time.sleep(20)
    
    #Operation-Location contains url to results, use it to get the processed JSON results
    headers = {'Ocp-Apim-Subscription-Key':COMP_VIS_SUBSCRIPTION_KEY}

    url = txt_analysis2.response.headers['Operation-Location']

    return json.loads(requests.get(url, headers=headers).text)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC retrieve_text_from_url - Uses SDK function recognize_text()
# MAGIC 
# MAGIC 
# MAGIC _Recognize Text operation. When you use the Recognize Text interface, the response contains a field called “Operation-Location”. The “Operation-Location” field contains the URL that you must use for your Get Handwritten Text Operation Result operation_
# MAGIC 
# MAGIC [Computer Vision SDK Python Docs](https://docs.microsoft.com/en-us/python/api/azure.cognitiveservices.vision.computervision?view=azure-python)
# MAGIC 
# MAGIC 
# MAGIC The code then captures the JSON results using the 'Operation-Location' URL.

# COMMAND ----------

def retrieve_text_from_url(imgurl):
    client = ComputerVisionAPI(COMPUTERVISION_LOCATION, CognitiveServicesCredentials(COMP_VIS_SUBSCRIPTION_KEY))
    txt_analysis2=client.recognize_text(imgurl,raw=True, mode='Printed')
    
    #give Computer Vision some time to process image, could also be a while loop checking status (20s is arbitrary)  
    time.sleep(20)
    
    #Operation-Location contains url to results, use it to get the processed JSON results
    headers = {'Ocp-Apim-Subscription-Key':COMP_VIS_SUBSCRIPTION_KEY}

    url = txt_analysis2.response.headers['Operation-Location']

    return json.loads(requests.get(url, headers=headers).text)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC retrieve_text_from_url_v2 -  The new preview OCR engine (through "Recognize Text" API operation) has even better text recognition results for English. The SDK used here is for V1 (which could change soon), we can specify V2 in the API call and compare the results.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC [Computer Vision V2 API](https://westus.dev.cognitive.microsoft.com/docs/services/5adf991815e1060e6355ad44/operations/587f2c6a154055056008f200)

# COMMAND ----------

def retrieve_text_from_url_v2(imgurl):
  
  
  headers = {
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': COMP_VIS_SUBSCRIPTION_KEY
  }

  #pass in mode and set raw equal to 'true'
  params = urllib.parse.urlencode({
    'mode': 'Printed',
    'raw':'True'
  })

  try:
    conn = http.client.HTTPSConnection('<ComputerVisionRegion>.api.cognitive.microsoft.com')
    conn.request("POST", "/vision/v2.0/recognizeText?%s" % params, "{'url':'" + imgurl + "'}" , headers)
    response = conn.getresponse()
    ol = response.headers.get('Operation-Location')
    conn.close()
  except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
  
  #give Computer Vision some time to process image, could also be a while loop checking status (30s is arbitrary)
  time.sleep(30)
    
  #clear parms
  params = urllib.parse.urlencode({})

  try:
    conn = http.client.HTTPSConnection('<ComputerVisionRegion>.api.cognitive.microsoft.com')
    conn.request("GET", "/vision/v2.0/textOperations/" + ol.split('/')[-1] + "/?%s" % params, "" , headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
  except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
    
  return json.loads(data)

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

image_results = retrieve_images("site:" + site + " \"" +  product['Name'] + "\"",IMG_SEARCH_SUBSCRIPTION_KEY)
img2 = retrieve_first_img_url(image_results)
    
comp_img = url_to_image(img2)


# COMMAND ----------

plot_img("Image Compare" + site,"Original Images",cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB),cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB),site)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve Text From Images

# COMMAND ----------

# MAGIC %md
# MAGIC Retrieve text from first image, display words from JSON

# COMMAND ----------

words1 = retrieve_text_from_img(img1)
for b in words1['recognitionResult']['lines']:
    print(b['text'])

# COMMAND ----------

# MAGIC %md
# MAGIC Retrieve text from second image (url), display words from JSON

# COMMAND ----------

words2 = retrieve_text_from_url(img2)
for b in words2['recognitionResult']['lines']:
    print(b['text'])

# COMMAND ----------

# MAGIC %md
# MAGIC Retrieve text from second image (url) using the V2 API, display words from JSON. 

# COMMAND ----------

words3 = retrieve_text_from_url_v2(img2)
for b in words3['recognitionResult']['lines']:
    print(b['text'])

# COMMAND ----------

dbutils.fs.unmount("dbfs:/mnt/images/")
