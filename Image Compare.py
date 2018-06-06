# Databricks notebook source
from skimage.measure import compare_ssim
import cv2
import matplotlib.pyplot as plt




# COMMAND ----------

# MAGIC %md
# MAGIC ### Mount Azure Storage
# MAGIC 
# MAGIC https://docs.azuredatabricks.net/spark/latest/data-sources/azure/azure-storage.html

# COMMAND ----------


dbutils.fs.mount(source = "wasbs://<your-container-name>@<your-storage-account-name>.blob.core.windows.net",mount_point = "/mnt/images/",extra_configs = {"fs.azure.account.key.<your-storage-account-name>.blob.core.windows.net": "<your-storage-account-access-key>})

# COMMAND ----------

IMAGES_FOLDER = "/dbfs/mnt/images/"


# COMMAND ----------

def crop_image(img):
    #prevent changes to original image
    img_o = img.copy()
    
    gray = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY) # convert to grayscale

    # threshold 
    retval, thresh_gray = cv2.threshold(gray, thresh=200, maxval=256, type=cv2.THRESH_BINARY)

    #find black pixels
    points = np.argwhere(thresh_gray==0) 
    #store in x, y coords
    points = np.fliplr(points)     
    
    x, y, w, h = cv2.boundingRect(points) 
    
    #expand box and do not allow negative (image may be x,y 0,0)
    x, y, w, h = x-10 if x-10>0 else 0, y-10 if y-10>0 else 0,w+20, h+20 
    print(x,y,w,h)
    
    # create a cropped region of the gray image
    crop = img[y:y+h, x:x+w] 

    return crop


# COMMAND ----------

def plot_img(figtitle,subtitle,img1,img2,site):
  
  #create figure with std size
  fig = plt.figure(figtitle, figsize=(10, 5))
  
  plt.suptitle(subtitle,fontsize=24)
  
  ax = fig.add_subplot(1, 2, 1)  
  # base is hardcoded for img1
  ax.set_title("Base",fontsize=12)
  plt.imshow(img1)
  plt.axis("off")
  
  ax = fig.add_subplot(1, 2, 2)
  # site is used in site iteration
  ax.set_title(site,fontsize=12)
  plt.imshow(img2)
  plt.axis("off")

  display(plt.show())

# COMMAND ----------

img1 = IMAGES_FOLDER  + "PAM_Original_6_OZ_ST.jpg"
orig_img =  cv2.imread(img1)

img2 = IMAGES_FOLDER  + "PAM_Original_6_OZ.jpg"
comp_img = cv2.imread(img2)

# COMMAND ----------

plot_img("Image Compare Orig" ,"Original Images",cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB),cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB),"Compare")


# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Crop to Images - Remove whitespace
# MAGIC SSIM Scores are biased by whitespace 

# COMMAND ----------

print(orig_img.shape)
print(comp_img.shape)

orig_img = crop_image(orig_img)
comp_img = crop_image(comp_img)

print(orig_img.shape)
print(comp_img.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resize Images for scoring
# MAGIC Size to smallest image

# COMMAND ----------

#identify smallest image size
small = min(orig_img.shape,comp_img.shape)[:2][::-1]
print(small)

# COMMAND ----------

#resize to smallest image
orig_img = cv2.resize(orig_img,dsize =small)
comp_img = cv2.resize(comp_img,small)

print(orig_img.shape)
print(comp_img.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Generate Multichannel (full color) SSIM  - Image Compare
# MAGIC SSIM Scores range from -1 to 1, with 1 indicating a "perfect match"

# COMMAND ----------

(score, diff) = compare_ssim(orig_img, comp_img, full=True,multichannel=True)

plot_img("Image Compare Multi","SSIM: %.2f" % (score),cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB),cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB),"Compare")



# COMMAND ----------

# MAGIC %md
# MAGIC ### Greyscale Image Compare - Normalize color difference

# COMMAND ----------

#create images for gray compare
gray1 = cv2.cvtColor(orig_img.copy(), cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(comp_img.copy(), cv2.COLOR_BGR2GRAY)


(score, diff) = compare_ssim(gray1, gray2, full=True,multichannel=False,gaussian_weights=True)

plot_img("Image Compare Gray" ,"Gray SSIM: %.2f" % (score),cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR),cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR),"Compare")


# COMMAND ----------

dbutils.fs.unmount("dbfs:/mnt/images/")
