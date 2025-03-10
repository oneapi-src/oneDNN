#! /usr/bin/env python3

import numpy as np
import os
import torch
from sys import argv


def read_off(file):
    if file.readline().strip() != "OFF":
        raise ("Not a valid OFF header")
    file_contents = []
    for line in file:
        if len(tuple([np.float32(s) for s in line.strip().split(" ")])) != 3:
            break
        x, y, z = tuple([np.float32(s) for s in line.strip().split(" ")])
        file_contents.append([x, y, z])
    return file_contents


def normalize(points):
    norm_pointcloud = points - np.mean(points, axis=0)
    norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
    return norm_pointcloud


def add_noise(arr):
    noise = np.random.normal(0, 0.02, (arr.shape))
    noisy_points = arr + noise
    return noisy_points


def read_pc(file_name, out, index):
    with open(file_name) as f:
        pointcloud = np.array(read_off(f))
        pointcloud[pointcloud > 20] = 20
        pointcloud[pointcloud < -20] = -20
        sample_idx = np.random.randint(len(pointcloud), size=1024)
        pointcloud_sampled = pointcloud[sample_idx, :]
        pointcloud_sampled = add_noise(normalize(pointcloud_sampled))
        out[index, :, :] = pointcloud_sampled


if __name__ == "__main__":
    if len(argv) < 3:
        raise "USAGE: python3 prepareData.py <ModelNet10 directory> <Pretrained pth file>"

    modelnet_path = argv[1]
    pointnet_path = argv[2]

    all_dir = os.listdir(modelnet_path)
    for item in all_dir:
        # create oneDNN input binary from chair point clouds
        data = np.empty((32, 1024, 3), np.float32)
        # change this variable to run pointnet inference on a different furniture
        # item
        furniture_item = item 

        file_names = [
            modelnet_path
            + furniture_item
            + "/train/"
            + furniture_item
            + "_"
            + str(i + 1).zfill(4)
            + ".off"
            for i in range(32)
        ]

        for i in range(0, 32):
            read_pc(file_names[i], data, i)

        with open("%s_cloud.bin" % furniture_item, "wb") as f:
            data.tofile(f)

        # load pretrained model and write weights to binary files for oneDNN to
        # parse
        if torch.cuda.is_available():
            pretrained_model = torch.load(
                pointnet_path, map_location=torch.device("cuda"))
        else:
            pretrained_model = torch.load(
                pointnet_path, map_location=torch.device("cpu"), weights_only=False)

        layers = list(pretrained_model.keys())

        if not os.path.exists("./data"):
            os.mkdir("./data")

        for name in layers:
            dim = pretrained_model[name].shape
            to_write = pretrained_model[name].detach(
            ).numpy().reshape(dim).squeeze().T
            with open("data/" + name + ".bin", "wb") as f:
                to_write.tofile(f)
            print("data/" + name + ".bin")

        id = torch.eye(3).repeat(32, 1, 1).numpy()
        with open("data/transform.input_transform.id.bin", "wb") as f:
            id.tofile(f)
        print("data/transform.input_transform.id.bin")
        id = torch.eye(64).repeat(32, 1, 1).numpy()
        with open("data/transform.feature_transform.id.bin", "wb") as f:
            id.tofile(f)
        print("data/transform.feature_transform.id.bin")
