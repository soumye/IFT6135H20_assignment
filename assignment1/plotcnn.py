import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    t = np.arange(1,11,1)
    mlp = {'train_accuracy': [0.95414, 0.97714, 0.98098, 0.93198, 0.99166, 0.99422, 0.99504, 0.993, 0.9961, 0.9952], 'validation_accuracy': [0.953, 0.9682, 0.9685, 0.9247, 0.9776, 0.9798, 0.9782, 0.9759, 0.9789, 0.9788], 'train_loss': [0.14611843527482027, 0.07132097222743923, 0.056205069701519894, 0.3406747417199354, 0.025020832143308025, 0.017115700325237453, 0.015872456081713116, 0.022286399866280972, 0.01174395987227517, 0.013664906140044602], 'validation_loss': [0.15438071526558775, 0.10201827949830646, 0.09938103202843356, 0.36329022301764974, 0.08438816027548024, 0.08213235570807893, 0.08284839168127885, 0.09607678819562436, 0.08593169898030795, 0.09431417610033066]}
    cnn = {'test_loss':[0.0405,0.0345,0.0358,0.0364,0.0360,0.0363,0.0383,0.0379,0.0383,0.0384], 'train_loss':[0.3235179,0.0386971,0.0074444,0.005347,0.0005758,0.0010112,0.0193162,0.000363,0.0001787,0.0001308]}
    cnn_drop = {'test_loss':[0.0630,0.0402,0.0296,0.0342,0.0308,0.0353,0.0311,0.0284,0.0281,0.0280],'train_loss':[0.3984375,0.081228,0.0494224,0.0400789,0.0320605,0.0353155,0.0669841,0.0111027,0.0174986,0.0272274]}
    # plt.plot(t,mlp['train_loss'], linestyle='--', marker='o', color='b')
    # plt.plot(t,cnn['train_loss'], linestyle='--', marker='.', color='r')
    # plt.legend(('MLP','CNN'))
    # plt.xlabel('# Epochs')
    # plt.ylabel('Training loss')
    # plt.xscale('log')
    # plt.tight_layout()
    # plt.savefig('cnn1.png')
    # plt.plot(t,mlp['validation_loss'], linestyle='--', marker='o', color='b')
    # plt.plot(t,cnn['test_loss'], linestyle='--', marker='.', color='r')
    # plt.legend(('MLP','CNN'))
    # plt.xlabel('# Epochs')
    # plt.ylabel('Test loss')
    # plt.tight_layout()
    # plt.savefig('cnn2.png')
    
    # plt.plot(t,cnn_drop['train_loss'], linestyle='--', marker='o', color='b')
    # plt.plot(t,cnn['train_loss'], linestyle='--', marker='.', color='r')
    # plt.legend(('CNN(Dropout)','CNN'))
    # plt.xlabel('# Epochs')
    # plt.ylabel('Train loss')
    # plt.tight_layout()
    # plt.savefig('cnn3.png')
    plt.plot(t,cnn_drop['test_loss'], linestyle='--', marker='o', color='b')
    plt.plot(t,cnn['test_loss'], linestyle='--', marker='.', color='r')
    plt.legend(('CNN(Dropout)','CNN'))
    plt.xlabel('# Epochs')
    plt.ylabel('Test loss')
    plt.tight_layout()
    plt.savefig('cnn4.png')


