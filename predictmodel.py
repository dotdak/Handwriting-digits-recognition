#import modules
import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from scipy.misc.pilutil import imresize

IMG_HEIGHT = 28
IMG_WIDTH = 28
CLASS_N = 10 # 0-9

USER_IMG = 'test_image.png'#'./test_data/im8.png'

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def load_digits(fn):
    print('loading "%s for training" ...' % fn)
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (DIGIT_WIDTH, DIGIT_HEIGHT))
    resized_digits = []
    for digit in digits:
        resized_digits.append(imresize(digit,(IMG_WIDTH, IMG_HEIGHT)))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return np.array(resized_digits), labels

def get_digits(contours, hierarchy):
    hierarchy = hierarchy[0]
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]   
    final_bounding_rectangles = []
    #find the most common heirarchy level - that is where our digits's bounding boxes are
    u, indices = np.unique(hierarchy[:,-1], return_inverse=True)
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]
    
    for r,hr in zip(bounding_rectangles, hierarchy):
        x,y,w,h = r
        #this could vary depending on the image you are trying to predict
        #we are trying to extract ONLY the rectangles with images in it (this is a very simple way to do it)
        #we use heirarchy to extract only the boxes that are in the same global level - to avoid digits inside other digits
        #ex: there could be a bounding box inside every 6,9,8 because of the loops in the number's appearence - we don't want that.
        #read more about it here: https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        if ((w*h)>250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy: 
            final_bounding_rectangles.append(r)    

    return final_bounding_rectangles

def proc_user_img(img_file):
    print('loading "%s for digit recognition" ...' % img_file)
    im = cv2.imread(img_file)
    
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    digits_rectangles = get_digits(contours,hierarchy)  #rectangles of bounding the digits in user image
    test_data = []
    position = []
    i = -1
    
    for rect in digits_rectangles:
        x,y,w,h = rect
        #cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        #im_digit = imgray[y:y+h,x:x+w]
        im_digit = thresh[y-4:y+h+4,x-6:x+w+6]
        
        a = im_digit.shape[0] if im_digit.shape[0]>im_digit.shape[1] else im_digit.shape[1]
        blank_image = np.zeros((a,a), np.uint8)
        blank_image.fill(255)
        b = (int)(a/2 - im_digit.shape[1]/2)
        c = (int)(a/2 - im_digit.shape[0]/2)
        #blank_image = im_digit.copyTo(blank_image(0,a,im_digit.cols,im_digit.rows))
        blank_image[c:c+im_digit.shape[0],b:b+im_digit.shape[1]] = im_digit
        #im_digit = (255-im_digit)
        #cv2.imwrite("abc.png",im_digit)
        im_digit = imresize(blank_image,(IMG_WIDTH ,IMG_HEIGHT))
        #im_digit = im_digit.reshape(1,IMG_WIDTH*IMG_HEIGHT)
        #normalized_digit = [k/255 if k>90 else 0 for k in im_digit[0]]
        test_data.append(im_digit)
        i += 1
        #cv2.imwrite(string,im_digit)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        position.append((x,y))
        #cv2.putText(im, str(i), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    #cv2.imwrite("overlay.png",im)
    return test_data, im, i, position

def write_result(im, result, j, position, name):
    for i in range(j):
        cv2.putText(im, str(result[i]), position[i],cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    cv2.imwrite(name,im)
    
def predictint(imvalue):       
    # Define the model (same as when creating the model file)
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
       
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()    
    
    with tf.Session() as sess:
        sess.run(init_op)
        new_saver = tf.train.import_meta_graph('model2.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))          
        prediction=tf.argmax(y_conv,1)
        return prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)

def imageprepare(argv):    
    #im = Image.open(argv).convert('L')
    im = Image.fromarray(argv)
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels 
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1  
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    #newImage.save("sample.png")
    tv = list(newImage.getdata()) #get pixel values    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255-x)/255 for x in tv] 
    return tva
    #print(tva)

def main(argv):
    test, im, i, position = proc_user_img(argv)
    result = []
    print("Number of digits: ", i+1)
    for j in range(i+1):
        imvalue = imageprepare(test[j])
        predint = predictint(imvalue)
        result.append(predint[0])
    blank_image = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
    blank_image.fill(255)
    write_result(im, result, i+1, position, "numbers_covered.png")
    write_result(blank_image, result, i+1, position, "result.png")
if __name__ == "__main__":
    main(sys.argv[1])
