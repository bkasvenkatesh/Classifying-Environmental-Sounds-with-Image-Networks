from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot


# Load HDF5 dataset
import h5py
train = h5py.File('/usr/share/digits/digits/jobs/20161117-040356-6e54/train_db/0.h5', 'r')
val = h5py.File('/usr/share/digits/digits/jobs/20161117-040356-6e54/val_db/0.h5', 'r')
test = h5py.File('/usr/share/digits/digits/jobs/20161117-040336-1289/test_db/0.h5', 'r')
#print(train.keys())
X_train = train['data']
X_test = test['data']
X_val = val['data']
y_train = train['label']
y_train = to_categorical(y_train, nb_classes=50)
y_test = test['label']
y_test = to_categorical(y_test, nb_classes=50)
y_val = val['label']
y_val = to_categorical(y_val, nb_classes=50)

# BUILD CRNN
inputs = Input(shape=(3, 256, 256))
x = Convolution2D(30, 3, 3, border_mode='valid', activation='relu', name='conv1')(inputs)
x = BatchNormalization(axis=1, mode=0)(x)
x = MaxPooling2D(pool_size=(3, 2), strides=(3, 2), name='pool1')(x)
x = Dropout(0.1, name='dropout1')(x)

x = Convolution2D(60, 3, 3, border_mode='valid', activation='relu', name='conv2')(x)
x = BatchNormalization(axis=1, mode=0)(x)
x = MaxPooling2D(pool_size=(3, 2), strides=(3, 2), name='pool2')(x)
x = Dropout(0.1, name='dropout2')(x)

x = Convolution2D(60, 3, 3, border_mode='valid', activation='relu', name='conv3')(x)
x = BatchNormalization(axis=1, mode=0)(x)
x = MaxPooling2D(pool_size=(4, 2), strides=(4, 2), name='pool3')(x)
x = Dropout(0.1, name='dropout3')(x)

x = Convolution2D(60, 3, 3, border_mode='valid', activation='relu', name='conv4')(x)
x = BatchNormalization(axis=1, mode=0)(x)
x = MaxPooling2D(pool_size=(4, 2), strides=(4, 2), name='pool4')(x)
x = Dropout(0.1, name='dropout4')(x)

print(x.get_shape())

x = Permute((3, 1, 2))(x)
x = Reshape((14, 60))(x)

x = GRU(30, return_sequences=True, name='gru1')(x)
x = GRU(30, return_sequences=False, name='gru2')(x)
x = Dropout(0.3, name='dropout5')(x)

output = Dense(50, activation='sigmoid', name='output')(x)

model = Model(inputs, output)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, decay=1e-8, momentum=0.9, nesterov=True), 
              metrics=['accuracy'])

#TRAINING

batch_size = 40
nb_epoch = 100
 
model_filename = 'crnnmodel.pkl' 
 
callbacks = [
    EarlyStopping(monitor='val_acc',
                  patience=10,
                  verbose=1,
                  mode='auto'),
    
    ModelCheckpoint(model_filename, monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    mode='auto'),
]
#model.load_weights('crnnmodel(61-100epoch).pkl')
history = model.fit(X_train, y_train,
                    batch_size=batch_size, 
                    nb_epoch=nb_epoch,
                    callbacks=callbacks,
                    verbose=1, validation_data=(X_test, y_test), shuffle="batch")

model.load_weights(model_filename) 
 
score = model.evaluate(X_val, y_val, verbose=0) 
print('Test score:', score[0])
print('Test accuracy:', score[1])
 
# plot learning curves
 
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

