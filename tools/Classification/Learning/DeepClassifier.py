from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import numpy as np
import cv2
import boto3
import io
import tempfile
from rex.settings import AWS_STORAGE_BUCKET_NAME,AWS_SECRET_ACCESS_KEY,AWS_ACCESS_KEY_ID

Labels = {'Aafush':0,'Dasheri':0,'Jamadar':0,'Kesar':0,'Langdo':0,'Rajapuri':0,'Totapuri':0}

def PredictImage(Image,ClassifierName='Mango_Mix_Classifier_model'):

    s3 = boto3.client('s3',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    key = 'static/'+ ClassifierName + '.h5'
    obj = s3.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=key)
    data = obj['Body'].read()
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(data)
    temp.close()

    model = models.load_model(str(temp.name))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    model.summary()
    img = cv2.resize(Image,dsize=(512,512))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img = image.load_img('Test.jpg', target_size=(512, 512))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    y = model.predict(img_tensor)
    y = np.round(np.float64(y),4) * 100
    cnt = 0
    for key in Labels.keys():
        print(str(key) + " :: " +str(y[0][cnt]))
        Labels[key] = round(y[0][cnt],4)
        cnt += 1
    return Labels
