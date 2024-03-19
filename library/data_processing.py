import os
import numpy as np

def get_num_trails(dataset):
    num_trails_dict = {'SEED': 15, 'SEED_IV': 24, 'DEAP': 40, 'FACED': 28}
    return num_trails_dict.get(dataset, 0)

def get_data_dir(dataset):
    return f'.\\PESD\\DATA\\{dataset}\\PSD'

def get_pretrained_path(dataset):
    dataset_pretrained_model = {
        'SEED': 'SEED_sub15_62_psd_feature_extraction_model_vit.pth',
        'SEED_IV': 'SEED_IV_sub15_62_psd_feature_extraction_model_vit.pth',
        'DEAP': 'DEAP_s17_62_psd_feature_extraction_model_vit.pth',
        'FACED': 'FACED_sub016_62_psd_feature_extraction_model_vit.pth'
    }
    pretrained_path = fr'.\PESD\DATA\pretrained_model\{dataset_pretrained_model.get(dataset, "default_pretrained_model.pth")}'
    return pretrained_path
def get_record_list(dataset):
    file_path = fr'\PESD\DATA\{dataset}_record_list.npy'
    return np.load(file_path)

def get_source_data(dataset):
    if dataset == 'SEED':
        record_source = np.array(['15_20130709.mat', '15_20131016.mat', '15_20131105.mat'])
        source_data, source_label,_ = load_data_trail_merge(record_source, get_data_dir(dataset), [], get_num_trails(dataset))
    elif dataset == 'SEED_IV':
        record_source = np.array(['15_20150508.mat', '15_20150514.mat', '15_20150527.mat'])
        source_data, source_label,_ = load_data_trail(record_source, get_data_dir(dataset), [], get_num_trails(dataset))
        source_data, source_label = filter_SEED_IV(source_data, source_label, 2)
    elif dataset == 'DEAP':
        record_source = 's16'
        source_data, source_label,_ = load_data_trail(record_source, get_data_dir(dataset), [], get_num_trails(dataset))
    elif dataset == 'FACED':
        record_source = 'sub061.pkl'
        source_data, source_label,_ = load_data_trail(record_source, get_data_dir(dataset), [], get_num_trails(dataset))
    return source_data, source_label

def merge_session_trail(record, trail_dir):
    x,y = [], []
    for session in range(record.shape[0]):
        record_name = record[session]
        x_session = np.load (os.path.join (trail_dir, record_name[:-4] + "_x_psd.npy"))
        y_session = np.load (os.path.join (trail_dir, record_name[:-4] + "_y.npy"))
        if x ==[]:
            x = x_session
            y = y_session
        else:
            x = np.concatenate((x,x_session),axis =0)
            y = np.concatenate((y,y_session),axis =0)
    x = x.reshape (x.shape[0], 1, x.shape[1], x.shape[2])
    return x,y

def load_data_trail_merge(record, data_dir, trails_test_target, num_train_source):
    x_train, y_train, x_test, y_test = [],[],[],[]
    for trail in range(num_train_source):
        if trail == trails_test_target:
            test_dir = os.path.join (data_dir, f'trail_{trail}')
            x_one_trail, y_one_trail = merge_session_trail (record, test_dir)
            if x_test == [] :
                x_test = x_one_trail
                y_test = y_one_trail
            else :
                x_test = np.concatenate ((x_test, x_one_trail), axis = 0)
                y_test = np.concatenate ((y_test, y_one_trail), axis = 0)
        else:
            train_dir = os.path.join (data_dir,f'trail_{trail}')
            x_one_trail, y_one_trail = merge_session_trail(record, train_dir)
            if x_train ==[]:
                x_train = x_one_trail
                y_train = y_one_trail
            else:
                x_train = np.concatenate ((x_train, x_one_trail), axis = 0)
                y_train = np.concatenate ((y_train, y_one_trail), axis = 0)
    return x_train, y_train, x_test, y_test

def load_data_trail(record, data_dir, trails_test_target, num_trails):
    x_train, y_train, x_test, y_test = [],[],[],[]
    for trail in range(num_trails):
        if trail == trails_test_target:
            test_dir = os.path.join (data_dir, f'trail_{trail}')
            x_one_trail = np.load (os.path.join (test_dir, record + "_x_psd.npy"))
            y_one_trail = np.load (os.path.join (test_dir, record + "_y.npy"))
            if x_test == [] :
                x_test = x_one_trail
                y_test = y_one_trail
            else :
                x_test = np.concatenate ((x_test, x_one_trail), axis = 0)
                y_test = np.concatenate ((y_test, y_one_trail), axis = 0)
        else:
            train_dir = os.path.join (data_dir,f'trail_{trail}')
            x_one_trail = np.load (os.path.join (train_dir, record + "_x_psd.npy"))
            y_one_trail = np.load (os.path.join (train_dir, record + "_y.npy"))
            if x_train ==[]:
                x_train = x_one_trail
                y_train = y_one_trail
            else:
                x_train = np.concatenate ((x_train, x_one_trail), axis = 0)
                y_train = np.concatenate ((y_train, y_one_trail), axis = 0)
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    if x_test:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
    return x_train, y_train, x_test, y_test

def filter_SEED_IV(x, y, to_remove) :
    indices_to_keep = y != to_remove
    x_filtered = x[indices_to_keep,:,:,:]
    y_filtered = y[indices_to_keep]
    y_filtered[y_filtered == 3] = to_remove
    return x_filtered, y_filtered
