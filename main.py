
from torch.utils.data import DataLoader, TensorDataset
import argparse
from library.models import VisionTransformer
from library.train_loop import Trainer
from library.data_processing import load_data_trail, load_data_trail_merge
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args, x_train, y_train, x_test, y_test, source_data, source_label, pretrained_path):

    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)
    source_dataloader = DataLoader(TensorDataset(source_data, source_label), batch_size=args.batch_size, shuffle=True)

    model_pretrained = VisionTransformer(args.patch_size, args.image_size, args.channel_num, args.num_classes)
    checkpoint = torch.load(pretrained_path)
    model_pretrained.load_state_dict(checkpoint)

    trainer = Trainer(args.model_pretrained, args.embed_dim, args.num_classes, args.temperature, args.alpha, args.learning_rate)

    for epoch in range(epochs):
        trainer.train(train_dataloader, source_dataloader, args.model_name)
        test_loss, test_accuracy = trainer.eval(test_dataloader)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%')
    return eval_acc, train_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of dataset")
    parser.add_argument ("--pretrain_dataset", type = str, help = "Name of dataset for pretraining model")
    parser.add_argument("--record", type=str, help="Description of record")
    parser.add_argument("--record_source", type=str, help="Description of record_source")
    parser.add_argument("--data_dir", type=str, help="Description of data_dir")
    parser.add_argument("--num_test_trail", type=int, help="Description of num_test_trail")
    parser.add_argument("--num_trail_source", type=int, help="Description of num_trail_source")
    parser.add_argument("--pretrained_path", type=str, help="Description of pretrained_path")
    parser.add_argument("--model_name", type=str, help="Description of model_name")

    parser.add_argument("--patch_size", type=str, default="(1, 62)", help="Description of patch_size")
    parser.add_argument("--image_size", type=str, default="(6, 62)", help="Description of image_size")
    parser.add_argument("--channel_num", type=int, default=1, help="Description of channel_num")
    parser.add_argument("--batch_size", type=int, default=128, help="Description of batch_size")
    parser.add_argument("--embed_dim", type=int, default=256, help="Description of embed_dim")
    parser.add_argument("--temperature", type=float, default=0.7, help="Description of temperature")
    parser.add_argument("--alpha", type=float, default=0.25, help="Description of alpha")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Description of learning_rate")
    parser.add_argument("--num_classes", type=int, default=3, help="Description of num_classes")
    args = parser.parse_args()

    data_dir = get_data_dir (dataset)
    num_trails = get_num_trails(dataset)
    record_list = get_record_list(dataset)
    pretrained_path = get_pretrained_path (pretrain_dataset)

    for num_test_target in range (num_trails) :
        for record_num in range (record_list.shape[0]) :
            record = record_list[record_num]

            if dataset == 'SEED' :
                x_test, x_train, y_test, y_train = load_data_trail_merge(record, data_dir, num_test_trail, num_trails)
            elif dataset == 'SEED_IV' :
                for session in range (3) :
                    x_test, x_train, y_test, y_train = load_data_trail(record[session], data_dir, num_test_target, num_trails)
                    x_test, y_test = filter_SEED_IV(x_test, y_test, 2)
                    x_train, y_train = filter_SEED_IV(x_train, y_train, 2)
            elif dataset in ['DEAP', 'FACED'] :
                x_test, x_train, y_test, y_train = load_data_trail(record, data_dir, num_test_target, num_trails)

            source_data, source_label = get_source_data(pretrain_dataset)
            test_acc, train_acc = train(args, x_train, y_train, x_test, y_test, source_data, source_label, pretrained_path)
            print(f'Test Accuracy: {test_acc}, Train Accuracy: {train_acc}')