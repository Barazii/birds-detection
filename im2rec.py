#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# from __future__ import print_function

import os
import sys

# curr_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.join(curr_path, "../python"))
import argparse
import random
import time
import traceback

import cv2
import mxnet as mx


def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split("\t")]
            line_len = len(line)
            if line_len < 3:
                print(
                    "lst should at least has three parts, but only has %s parts for %s"
                    % (line_len, line)
                )
                continue
            try:
                item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            except Exception as e:
                print("Parsing lst met error for %s, detail: %s" % (line, e))
                continue
            yield item


def image_encode(args, i, item, q_out):
    fullpath = os.path.join(args.root, item[1])

    if len(item) > 3 and args.pack_label:
        header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
    else:
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)

    try:
        img = cv2.imread(fullpath, args.color)
    except:
        traceback.print_exc()
        print("imread error trying to load file: %s " % fullpath)
        q_out.put((i, None, item))
        return
    if img is None:
        print("imread read blank (None) image for file: %s" % fullpath)
        q_out.put((i, None, item))
        return
  
    try:
        s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
        q_out.put((i, s, item))
    except Exception as e:
        traceback.print_exc()
        print("pack_img error on file: %s" % fullpath, e)
        q_out.put((i, None, item))
        return


def read_worker(args, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(args, i, item, q_out)


def write_worker(q_out, fname, working_dir):
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + ".rec"
    fname_idx = os.path.splitext(fname)[0] + ".idx"
    record = mx.recordio.MXIndexedRecordIO(
        os.path.join(working_dir, fname_idx), os.path.join(working_dir, fname_rec), "w"
    )
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                record.write_idx(item[0], s)

            if count % 1000 == 0:
                cur_time = time.time()
                print("time:", cur_time - pre_time, " count:", count)
                pre_time = cur_time
            count += 1


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create an image list or \
        make a record database by reading from an image list",
    )
    parser.add_argument("--prefix", help="prefix of input/output lst and rec files.")
    parser.add_argument("--root", help="path to folder containing images.")

    rgroup = parser.add_argument_group("Options for creating database")
    rgroup.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9",
    )
    rgroup.add_argument(
        "--num-thread",
        type=int,
        default=1,
        help="number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.",
    )
    rgroup.add_argument(
        "--color",
        type=int,
        default=1,
        choices=[-1, 0, 1],
        help="specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.",
    )
    rgroup.add_argument(
        "--encoding",
        type=str,
        default=".jpg",
        choices=[".jpg", ".png"],
        help="specify the encoding of the images.",
    )
    rgroup.add_argument(
        "--pack-label",
        action="store_true",
        help="Whether to also pack multi dimensional label in the record file",
    )
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args


if __name__ == "__main__":
    args = parse_args()
    if os.path.isdir(args.prefix):
        working_dir = args.prefix
    else:
        working_dir = os.path.dirname(args.prefix)
    files = [
        os.path.join(working_dir, fname)
        for fname in os.listdir(working_dir)
        if os.path.isfile(os.path.join(working_dir, fname))
    ]
    count = 0
    for fname in files:
        if fname.startswith(args.prefix) and fname.endswith(".lst"):
            print("Creating .rec file from", fname, "in", working_dir)
            count += 1
            image_list = read_list(fname)
            # -- write_record -- #
            if args.num_thread > 1:
                import multiprocessing
                q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
                q_out = multiprocessing.Queue(1024)
                read_process = [
                    multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out))
                    for i in range(args.num_thread)
                ]
                for p in read_process:
                    p.start()
                write_process = multiprocessing.Process(
                    target=write_worker, args=(q_out, fname, working_dir)
                )
                write_process.start()

                for i, item in enumerate(image_list):
                    q_in[i % len(q_in)].put((i, item))
                for q in q_in:
                    q.put(None)
                for p in read_process:
                    p.join()

                q_out.put(None)
                write_process.join()
            else:
                print("multiprocessing not used, use single thread.")
                import queue
                q_out = queue.Queue()
                fname = os.path.basename(fname)
                fname_rec = os.path.splitext(fname)[0] + ".rec"
                fname_idx = os.path.splitext(fname)[0] + ".idx"
                record = mx.recordio.MXIndexedRecordIO(
                    os.path.join(working_dir, fname_idx),
                    os.path.join(working_dir, fname_rec),
                    "w",
                )
                cnt = 0
                pre_time = time.time()
                for i, item in enumerate(image_list):
                    image_encode(args, i, item, q_out)
                    if q_out.empty():
                        continue
                    _, s, _ = q_out.get()
                    record.write_idx(item[0], s)
                    if cnt % 1000 == 0:
                        cur_time = time.time()
                        print("time:", cur_time - pre_time, " count:", cnt)
                        pre_time = cur_time
                    cnt += 1
    if not count:
        print("Did not find and list file with prefix %s" % args.prefix)
