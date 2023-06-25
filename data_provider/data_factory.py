from data_provider.data_loader import Taxi_data
from torch.utils.data import DataLoader

def data_provider(args, flag):
    Data = Taxi_data

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq

    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        split=flag,
        data_path=args.data_path,
        json_path=args.json_path,
        data_len=args.data_len,
        pre_len=args.pred_len,
        seq_len=args.seq_len,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
        
    return data_set, data_loader

