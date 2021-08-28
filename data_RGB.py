import os
from dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderValSubSet, DataLoaderTrainSubSet


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)


def get_validation_data_SubSet(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderValSubSet(rgb_dir, img_options)


def get_train_data_SubSet(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrainSubSet(rgb_dir, img_options)
