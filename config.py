import re

NEGATIVE_REVIEWS_FOLDER_PATH = '/Users/eugeneolkhovik/Desktop/master/text_mining/aclImdb/train/neg'
POSITIVE_REVIEWS_FOLDER_PATH = "/Users/eugeneolkhovik/Desktop/master/text_mining/aclImdb/train/pos"
UNSUPERVISED_FOLDER_PATH = '/Users/eugeneolkhovik/Desktop/master/text_mining/aclImdb/train/unsup'


REG_HASHTAG = re.compile('<.*?>')

SEED = 1863
MAX_LEN = 128
BATCH_SIZE = 32