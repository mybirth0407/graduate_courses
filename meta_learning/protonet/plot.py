# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2022-04-14 02:12:57
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2022-04-14 17:19:07

import matplotlib.pyplot as plt

def plot(
    data1, data2,
    figsize=(15, 10),
):
    plt.figure(figsize=figsize)
    fig, ax1 = plt.subplots()

    t = range(1, len(data1)+1)
    plt.plot(t, data1, c='r')
    plt.plot(t, data2, c='b')
    plt.title('Train / Valid Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['Train Acc', 'Valid Acc'])
    plt.grid(True)