import os
import numpy as np


class File:
    """
    _summary_
    """

    def __init__(self, filename, kind="file") -> None:
        self.path = filename
        self.kind = kind
        pass

    def check_for_file(self):
        print("Checking for {}s...".format(self.kind))
        if os.path.isfile(self.path):
            print("Found some.")
            return True
        else:
            print("None found.")
            return False


def check_for_file(name, kind="file"):
    print("Checking for {}s...".format(kind))
    if os.path.isfile(name):
        print("Found some.")
        return True
    else:
        print("None found.")
        return False


# load can become one function with a dictionary or list of what to load and return
def load_pdfs(name):
    print("Loading pdfs ", end="")
    mfile = np.load(name)
    print("with angles ", end="")
    print(mfile["angs"])
    return mfile["x"], mfile["pdf"], mfile["stats"], mfile["angs"]


def load_cfs(name):
    print("Loading cfs ", end="")
    mfile = np.load(name)
    print("with angles ", end="")
    print(mfile["angs"])
    return mfile["t"], mfile["cf_re"], mfile["cf_im"], mfile["ximax"], mfile["angs"]


def save_matrix(m, filename, kind="M"):
    print("Saving matrix.".format(kind))
    np.savez(filename, matrix=m)
