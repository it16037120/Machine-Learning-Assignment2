# set the matplotlib backend
import imutils
import matplotlib

from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from mimarilerim.alexnet_modified import networkArchFonc

title = 'Covid ECG features maps CN_kFold'
hangi_calisma = "covid_ECG_training"  
ag_modeli = "Modified-Alexnet" + " + Adam Optimizier"
dataset = 'preprocessed_dataset'  
etiket = ["NO-FINDINGS", "COVID-19"]
sira = 34
EPOCHS = 50 
INIT_LR = 1e-3
BS = 128  
boyut = 256  
sinif = 2
derinlik = 3
fboyut = 20
cm_renkler = "Blues"
fold_sayisi = 5

ana_dizin = "../" #YOU CAN CHANGE


trainingCSV_kfold_train_and_validate = ana_dizin + hangi_calisma + "/sonuclar/" + "kfold_train_and_validateCSV.log"
sira = str(sira) + "-" + title
dataset = ana_dizin + hangi_calisma + "/" + dataset
fig_boyut = (fboyut, fboyut)
resim_boyut = (boyut, boyut)
sonuc_kayit = ana_dizin + hangi_calisma + "/sonuclar/"
model_save = ana_dizin + hangi_calisma + '/modeller/' + sira + "kFold" + "-M"
acc_kayit = ana_dizin + hangi_calisma + "/sonuclar/" + sira + "kFold" + "-ACC"
loss_kayit = ana_dizin + hangi_calisma + "/sonuclar/" + sira + "kFold" + "-LOSS"
val_acc_kayit = ana_dizin + hangi_calisma + "/sonuclar/" + sira + "kFold" + "-VALACC"
val_loss_kayit = ana_dizin + hangi_calisma + "/sonuclar/" + sira + "kFold" + "-VALLOSS"
cm_kayit = ana_dizin + hangi_calisma + "/sonuclar/" + sira + "kFold" + "-CM"
val_cm_kayit = ana_dizin + hangi_calisma + "/sonuclar/" + sira + "kFold" + "-CMVAL"
val_PR_kayit = ana_dizin + hangi_calisma + "/sonuclar/" + sira + "kFold" + "-PRVAL"

PR_kayit = ana_dizin + hangi_calisma + "/sonuclar/" + sira + "kFold" + "-PR"
ROC_kayit = ana_dizin + hangi_calisma + "/sonuclar/" + sira + "kFold" + "-ROC"
val_ROC_kayit = ana_dizin + hangi_calisma + "/sonuclar/" + sira + "kFold" + "-ROCVAL"


def load_data():
    # initialize the data and labels
    print("[INFO] loading images...")
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(dataset)))
   

    
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)  
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
       
        image = cv2.resize(image, resim_boyut, interpolation=cv2.INTER_AREA)
        
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        if label == "N":
            label = 0
        elif label == "C":
            label = 1
        
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels


def create_model():
    # initialize the model
    print("[INFO] compiling model...")
    model = networkArchFonc.build(width=boyut, height=boyut, depth=derinlik, classes=sinif)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    return model


def train_and_validate_model(model, data, labels):


    # partition the data into training and testing splits using 75% of
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.25, random_state=42)
    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=sinif)
    testY = to_categorical(testY, num_classes=sinif)

    csv_logger = CSVLogger(trainingCSV_kfold_train_and_validate, append=True)
    # train the network
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, batch_size=BS,
                  validation_data=(testX, testY),  
                  epochs=EPOCHS, verbose=0, callbacks=[csv_logger])
    return H,testX,testY


def tic():
    
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print( "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        return str(time.time() - startTime_for_tictoc)
    else:
        print ("Toc: start time not set")

print("[INFO] Cross-Validation basliyor...")

# fix random seed 
seed = 14
np.random.seed(seed)
data, labels = load_data()
# define 5-fold cross validation
kfold = StratifiedKFold(n_splits=fold_sayisi, shuffle=True, random_state=seed)
testing_acc_values = []
fold = 1
#for curve for testing
tprs = []
aucs = []
fprs_kf = []
tprs_kf = []
mean_fpr = np.linspace(0, 1, 100)

#validation curves
val_tprs = []
val_aucs = []
val_fprs_kf = []
val_tprs_kf = []



#for training
training_ACCs=[]
training_loses=[]
val_ACCs=[]
val_loses=[]

for train, test in kfold.split(data, labels):
    # create model
    print("Running Fold {0} / {1}".format(fold, fold_sayisi))
   
    model = create_model()
    # Fit the model
    tic()
    H,validationX,validationY=train_and_validate_model(model, data[train], labels[train])
    egitim_suresi=toc()
    training_ACCs.append(H.history["accuracy"])
    training_loses.append(H.history["loss"])
    val_ACCs.append(H.history["val_accuracy"])
    val_loses.append(H.history["val_loss"])

    # evaluate the model
    labels_test_cat = to_categorical(labels[test], num_classes=sinif)
 
    scores = model.evaluate(data[test], labels_test_cat, verbose=0)

    print("For Fold {0} / {1}  {2}: {3:.2f}".format(fold, fold_sayisi, model.metrics_names[1], scores[1] * 100))
    testing_acc_values.append(scores[1] * 100)

    # predict the model
    print("[INFO] Cizim islemleri basliyor... for testing")
    print("[INFO] Cizim islemleri basliyor for validation...")

   

    fold = fold + 1
print("AVG CV Results: %.2f%% (+/- %.2f%%)" % (np.mean(testing_acc_values), np.std(testing_acc_values)))
print("[INFO] Cross-Validation tamamlandi!...")
