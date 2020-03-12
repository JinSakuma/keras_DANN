from __future__ import print_function, division
from keras import layers
from keras.layers import Input
from keras.models import Model
from keras.engine.network import Network
from keras.preprocessing import image
from keras.optimizers import Adam, SGD
import tqdm
import tensorflow as tf
from flipGradientTF import GradientReversal
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.backend import tensorflow_backend as K
import pandas as pd
import matplotlib.pyplot as plt

from data_generator import DataGenerator

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)


class DANN():
    def __init__(self, n_classes, config, output_name="output"):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_res = (self.img_rows, self.img_cols)
        self.n_classes = n_classes
        # data_path
        self.st_path = config.SOURCE_TRAIN_PATH
        self.sv_path = config.SOURCE_VALID_PATH
        self.tt_path = config.TARGET_TRAIN_PATH
        self.tv_path = config.TARGET_VALID_PATH

        self.output_name = output_name
        if not os.path.exists(self.output_name):
            os.makedirs(self.output_name, exist_ok=True)

        self.output_file = open(os.path.join(self.output_name, "progress.csv"), "w")
        names = ["epoch", "lr_lambda", "hp_lambda", "loss_d_t", "acc_d_t",
                 "loss_c_t", "acc_c_t", "loss_d_v", "acc_d_v", "loss_c_v_s",
                 "acc_c_v_s", "loss_c_v_t", "acc_c_v_t"]
        self.output_file.write(",".join(names)+"\n")

        self._build_models()

    def _build_models(self, hp_lambda=1., lr=0.001):

        # Input images from both domains
        img = Input(shape=self.img_shape)

        self.e = self._build_extracter()
        self.c = self._build_classifier()
        self.d = self._build_discriminator()

        f = self.e(img)
        gradInv = self.gradInv = GradientReversal(hp_lambda=hp_lambda)
        K.set_value(gradInv.hp_lambda, hp_lambda)
        fInv = gradInv(f)

        cls = self.c(f)
        dom = self.d(fInv)

        self.model = Model(inputs=img, outputs=cls, name="model")
        self.compile(self.model, lr, name='classifier')

        self.classifier = Model(inputs=img, outputs=cls, name="classifier")
        self.compile(self.classifier, lr, name='classifier')

        self.discriminator = Model(inputs=img, outputs=dom, name="discriminator")
        self.compile(self.discriminator, lr * 0.1, name='discrimimator')

    def _build_extracter(self):
        inp = Input(shape=self.img_shape)
        h = inp
        h = layers.Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu')(h)
        h = layers.MaxPooling2D(2, strides=2)(h)
        h = layers.Conv2D(48, kernel_size=5, strides=1, padding='same', activation='relu')(h)
        h = layers.MaxPooling2D(2, strides=2)(h)
        h = layers.Flatten()(h)
        self.n_features = K.int_shape(h)[-1]
        # h = layers.Dense(50, activation='relu')(h)
        outY = h

        return Model(inp, outY, name="extracter")
#         return Network(inp, outY, name="extracter")

    def _build_classifier(self):
        inp = Input(shape=(self.n_features,))
        h = inp
        h = layers.Dense(100, activation='relu')(h)
        h = layers.Dense(100, activation='relu')(h)
        h = layers.Dense(10, activation='softmax')(h)
        outY = h

#         return Network(inp, outY, name="classifier")
        return Model(inp, outY, name="classifier")

    def _build_discriminator(self):
        inp = Input(shape=(self.n_features,))
        h = inp
        h = layers.Dense(100, activation='relu')(h)
        h = layers.Dense(1, activation='sigmoid')(h)
        # h = layers.Dense(   2, activation='softmax')(h)
        outY = h

        return Model(inp, outY, name="discriminator")
#         return Network(inp, outY, name="discriminator")

    def _lr_scheduler(self, epoch, max_epoch):
        mu0 = 0.01
        alpha = 10.
        beta = 0.75
        p = float(epoch) / max_epoch

        return mu0 / ((1. + alpha * p) ** beta)

    def _hp_scheduler(self, epoch, max_epoch):
        gamma = 10.
        p = float(epoch) / max_epoch
        return (2. / (1. + np.exp(-1 * gamma * p)) - 1.)

    def compile(self, model, lr, name='classifier'):
        optimizer = SGD(lr=lr, momentum=0.9)
        # optimizer = Adam(lr=lr)
        if name == 'classifier':
            model.compile(loss="categorical_crossentropy",
                          metrics=["accuracy"],
                          optimizer=optimizer)

        elif name == 'discrimimator':
            model.compile(loss=["binary_crossentropy"],
                          metrics=["accuracy"],
                          optimizer=optimizer)

    def train(self, batch_size=16, train_Steps=100, val_Steps=5, nEpochs=100):
        gen_S_train = DataGenerator(self.n_classes, self.img_res, self.st_path, batch_size)
        gen_S_val = DataGenerator(self.n_classes, self.img_res, self.sv_path, batch_size)
        gen_T_train = DataGenerator(self.n_classes, self.img_res, self.tt_path, batch_size)
        gen_T_val = DataGenerator(self.n_classes, self.img_res, self.tv_path, batch_size)

        gen_S_train.on_epoch_end()
        gen_T_train.on_epoch_end()

        prev = 0
        for epoch in range(nEpochs):
            # setting parameters
            K.set_value(self.gradInv.hp_lambda, self._hp_scheduler(epoch, nEpochs))
            lr = self._lr_scheduler(epoch, nEpochs)
            self.compile(self.classifier, lr, name='classifier')
            self.compile(self.discriminator, lr*0.1, name='discrimimator')

            # train
            res_S_train = np.array([0., 0.])
            res_d_train = np.asarray([0., 0.])
            for batch in tqdm.tqdm(range(train_Steps)):
                xSt, ySt = gen_S_train.next()
                lSt = self.classifier.train_on_batch(xSt, ySt)
                res_S_train += np.asarray(lSt)

                xTt, _ = gen_T_train.next()
                dSt = np.zeros((batch_size))
                dTt = np.ones((batch_size))

                ldt = self.discriminator.train_on_batch(np.concatenate((xSt, xTt)), np.concatenate((dSt, dTt)))
                res_d_train += np.asarray(ldt)

            los_S_train, acc_S_train = res_S_train / train_Steps
            los_d_train, acc_d_train = res_d_train / train_Steps

            # valid
            res_S_val = np.array([0., 0.])
            res_T_val = np.asarray([0., 0.])
            res_d_val = np.asarray([0., 0.])
            for batch in range(val_Steps):
                xSv, ySv = gen_S_val.next()
                lSv = self.classifier.test_on_batch(xSv, ySv)
                res_S_val += np.asarray(lSv)

                xTv, yTv = gen_T_val.next()
                lTv = self.classifier.test_on_batch(xTv, yTv)
                res_T_val += np.asarray(lTv)

                dSv = np.zeros((batch_size))
                dTv = np.ones((batch_size))

                ld = self.discriminator.test_on_batch(np.concatenate((xSv, xTv)), np.concatenate((dSv, dTv)))
                res_d_val += np.asarray(ld)

            los_S_val, acc_S_val = res_S_val / val_Steps
            los_T_val, acc_T_val = res_T_val / val_Steps
            los_d_val, acc_d_val = res_d_val / val_Steps

            gen_S_val.on_epoch_end()
            gen_T_val.on_epoch_end()

            val_lr_lambda = K.get_value(self.classifier.optimizer.lr)
            val_hp_lambda = K.get_value(self.gradInv.hp_lambda)

            print()
            print("[Epoch %d, lr=%.1e, lambda=%.1e]" % (epoch, val_lr_lambda, val_hp_lambda))
            print("<Train> [Dom loss: %.2f, acc: %3d%%] [Cls(S) loss: %.2f, acc: %3d%%]"
                  % (los_d_train, acc_d_train*100.,
                     los_S_train, acc_S_train*100.,))
            print("<Val>   [Dom loss: %.2f, acc: %3d%%] [Cls(S) loss: %.2f, acc: %3d%%] [Cls(T) loss: %.2f, acc: %3d%%]"
                  % (los_d_val, acc_d_val*100.,
                     los_S_val, acc_S_val*100.,
                     los_T_val, acc_T_val*100.,))

            # save weights when Target loss is the minimum
            if acc_T_val >= prev:
                self.classifier.save_weights(os.path.join(self.output_name, 'dann_acc-{:.2f}_loss-{:.2f}.hdf5'.format(acc_T_val, los_T_val)))
                prev = acc_T_val

            # draw graph
            outlines = []
            outlines.append(epoch)
            outlines.append(val_lr_lambda)
            outlines.append(val_hp_lambda)

            outlines.append(los_d_train)
            outlines.append(acc_d_train*100.)
            outlines.append(los_S_train)
            outlines.append(acc_S_train*100.)
            outlines.append(los_d_val)
            outlines.append(acc_d_val*100.)
            outlines.append(los_S_val)
            outlines.append(acc_S_val*100.)
            outlines.append(los_T_val)
            outlines.append(acc_T_val*100.)

            self.output_file.write(",".join([str(x) for x in outlines])+"\n")
            self.output_file.flush()

            d = pd.read_csv(self.output_name+"/progress.csv")
            if d.shape[0] == 1:
                continue
            d = d.interpolate()
            p = d.plot(x="epoch", y=["acc_d_v", "acc_c_v_s", "acc_c_v_t"])
            fig = p.get_figure()
            fig.savefig(self.output_name+"/graph.png")
            plt.close()

    def train_source_only(self, batch_size=16, nSteps=1000, nEpochs=10):
        gen_S_train = DataGenerator(self.n_classes, self.img_res, self.st_path, batch_size)
        gen_S_val = DataGenerator(self.n_classes, self.img_res, self.sv_path, batch_size)
        gen_T_val = DataGenerator(self.n_classes, self.img_res, self.tv_path, batch_size)

        self.classifier.fit_generator(gen_S_train, validation_data=gen_S_val, steps_per_epoch=nSteps, epochs=nEpochs, verbose=1)
        print("source only:")
        source_res = self.classifier.evaluate_generator(gen_S_val, steps=len(gen_S_val))
        print("Source Acc: %.2f" % source_res[1])
        target_res = self.classifier.evaluate_generator(gen_T_val, steps=len(gen_T_val))
        print("Target Acc: %.2f" % target_res[1])
        # save model
        self.classifier.save_weights(os.path.join(self.output_name, '/source_only_{:.2f}.hdf5'.format(target_res[1]*100)))
        # initialize model
        # self._build_models()
