import numpy as np


def gen_fake_data():
    fake_data = np.random.rand(1, 3, 256, 256).astype(np.float32) - 0.5
    fake_label = np.arange(1).astype(np.int64)
    np.save("./data/fake_data.npy", fake_data)
    np.save("./data/fake_label.npy", fake_label)


if __name__ == '__main__':
    gen_fake_data()
