# -*- coding:utf-8 -*-
import pickle
import sys
import ngtpy

with open(sys.argv[1], "rb") as f:
    idolvecs = pickle.load(f)


def similar_v(index, v):
    result = index.search(v, 20)
    for j, score in result:
        print(f"{idolvecs[0][j]}({j}): {score}")


def main():
    dim = 100
    ngtpy.create(b"tmp", dim)
    index = ngtpy.Index(b"tmp")
    index.batch_insert(idolvecs[1])
    index.save()

    i = int(sys.argv[2])
    target_name = idolvecs[0][i]
    target_vec = idolvecs[1][i]
    print(target_name)

    similar_v(index, target_vec)


if __name__ == '__main__':
    main()
