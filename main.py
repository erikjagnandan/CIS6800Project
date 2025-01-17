from utils import *
from train_and_val import *
from model_definition_cnn import CNNModel

def main():
    parser = argparse.ArgumentParser(description="SmoothDINOv2 Main Script")

    parser.add_argument('--model_string', type=str, default='cnn', choices=['cnn', 'cnn_regularized'], help="Model type")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size")
    parser.add_argument('--batches_per_backprop', type=int, default=1, help="Number of batches per backpropagation")
    parser.add_argument('--train', action='store_true', help="Run training if set, otherwise validation")
    parser.add_argument('--load_model', action='store_true', help="Load a saved model if set")
    parser.add_argument('--epoch_to_load', type=int, default=9, help="Epoch to load the model from (if loading model)")
    parser.add_argument('--segment_to_load', type=int, default=None, help="Segment to load (if loading model)")
    parser.add_argument('--val_checkpoint_ratios', type=float, nargs='+', default=[0.25, 0.5, 0.75], help="Validation checkpoint ratios")
    parser.add_argument('--backbone_size', type=str, default='small', choices=['small', 'base', 'large', 'giant'], help="Backbone size")

    args = parser.parse_args()

    data_directory = './datasets/nyu_data/data/nyu2_train'
    filename_train = './train_val_split/train_list.json'
    filename_val = './train_val_split/val_list.json'

    image_height = 480
    image_width = 640

    feature_height = 35
    feature_width = 46

    num_epochs = 10
    learning_rate = 1e-5

    backbone_name, backbone_model = load_backbone(backbone_size=args.backbone_size)
    model = load_dino_model(backbone_name, backbone_model, backbone_size=args.backbone_size, head_dataset="nyu", head_type="dpt")

    with open(filename_train, 'r') as f:
        train_list = json.load(f)

    with open(filename_val, 'r') as f:
        val_list = json.load(f)

    train_list = [sample for sample in train_list if sample[1] > 1]
    val_list = [sample for sample in val_list if sample[1] > 1]

    num_images_train = len(train_list)
    num_images_val = len(val_list)

    transform = make_depth_transform()

    if args.train:
        print("Training " + args.model_string)
    else:
        print("Running Validation for " + args.model_string)
        args.load_model = True

    adapter_model = CNNModel(1, 1)
    adapter_model.cuda()
    optimizer = torch.optim.Adam(adapter_model.parameters(), lr=learning_rate)

    num_batches_train = int(num_images_train / args.batch_size)
    val_checkpoints = (num_batches_train * np.array(args.val_checkpoint_ratios)).astype(int)

    if args.load_model:
        if args.segment_to_load is None:
            adapter_model.load_state_dict(torch.load(f'./models/{args.model_string}_model_epoch_{args.epoch_to_load}.pth', weights_only=True))
            start_epoch = args.epoch_to_load + 1
            start_batch_index = 0
            val_flag = 0
        else:
            adapter_model.load_state_dict(torch.load(f'./models/{args.model_string}_model_epoch_{args.epoch_to_load}_segment_{args.segment_to_load}.pth', weights_only=True))
            start_epoch = args.epoch_to_load
            if val_checkpoints[args.segment_to_load - 1] % args.batches_per_backprop == args.batches_per_backprop - 1:
                start_batch_index = val_checkpoints[args.segment_to_load - 1] + 1
            else:
                start_batch_index = val_checkpoints[args.segment_to_load - 1] - (val_checkpoints[args.segment_to_load - 1] % args.batches_per_backprop)
            val_flag = args.segment_to_load
    else:
        start_epoch = 0
        start_batch_index = 0
        val_flag = 0

    original_mse_train, updated_mse_train, original_mse_val, updated_mse_val = load_training_logs(num_epochs, len(val_checkpoints), args.model_string)

    difference_train = torch.zeros(num_epochs)
    difference_val = torch.zeros(num_epochs, len(val_checkpoints) + 1)

    if args.train:
        for epoch in range(start_epoch, num_epochs):
            if epoch == start_epoch:
                original_mse_current_epoch_train_mean, updated_mse_current_epoch_train_mean, original_mse_val, updated_mse_val = train_loop(
                    original_mse_val, updated_mse_val, train_list, val_list, data_directory, transform, model, adapter_model,
                    optimizer, image_height, image_width, feature_height, feature_width, num_images_train, num_images_val,
                    args.batch_size, args.batches_per_backprop, epoch, val_checkpoints, val_flag, args.model_string,
                    start_batch_index=start_batch_index)
            else:
                original_mse_current_epoch_train_mean, updated_mse_current_epoch_train_mean, original_mse_val, updated_mse_val = train_loop(
                    original_mse_val, updated_mse_val, train_list, val_list, data_directory, transform, model, adapter_model,
                    optimizer, image_height, image_width, feature_height, feature_width, num_images_train, num_images_val,
                    args.batch_size, args.batches_per_backprop, epoch, val_checkpoints, val_flag, args.model_string,
                    start_batch_index=0)

            original_mse_train[epoch] = original_mse_current_epoch_train_mean
            updated_mse_train[epoch] = updated_mse_current_epoch_train_mean

            torch.save(adapter_model.state_dict(), f'./models/{args.model_string}_model_epoch_{epoch}.pth')

            torch.save(original_mse_train, f'./training_logs/original_mse_train_{args.model_string}.pt')
            torch.save(updated_mse_train, f'./training_logs/updated_mse_train_{args.model_string}.pt')

            original_mse_current_epoch_val_mean, updated_mse_current_epoch_val_mean = val_loop(
                val_list, data_directory, transform, model, adapter_model, image_height, image_width, feature_height,
                feature_width, num_images_val, args.batch_size, epoch, args.model_string, end_of_epoch=True)
            original_mse_val[epoch, -1] = original_mse_current_epoch_val_mean
            updated_mse_val[epoch, -1] = updated_mse_current_epoch_val_mean

            torch.save(original_mse_val, f'./training_logs/original_mse_val_{args.model_string}.pt')
            torch.save(updated_mse_val, f'./training_logs/updated_mse_val_{args.model_string}.pt')

            val_flag = 0
    else:
        if args.segment_to_load is None:
            original_mse_current_epoch_val_mean, updated_mse_current_epoch_val_mean = val_loop(
                val_list, data_directory, transform, model, adapter_model, image_height, image_width, feature_height,
                feature_width, num_images_val, args.batch_size, args.epoch_to_load, args.model_string, end_of_epoch=True)
            original_mse_val[args.epoch_to_load, -1] = original_mse_current_epoch_val_mean
            updated_mse_val[args.epoch_to_load, -1] = updated_mse_current_epoch_val_mean
        else:
            original_mse_current_epoch_val_mean, updated_mse_current_epoch_val_mean = val_loop(
                val_list, data_directory, transform, model, adapter_model, image_height, image_width, feature_height,
                feature_width, num_images_val, args.batch_size, args.epoch_to_load, args.model_string, end_of_epoch=False,
                training_percentage=100 * args.segment_to_load / (len(val_checkpoints) + 1))
            original_mse_val[args.epoch_to_load, args.segment_to_load - 1] = original_mse_current_epoch_val_mean
            updated_mse_val[args.epoch_to_load, args.segment_to_load - 1] = updated_mse_current_epoch_val_mean

        torch.save(original_mse_val, f'./training_logs/original_mse_val_{args.model_string}.pt')
        torch.save(updated_mse_val, f'./training_logs/updated_mse_val_{args.model_string}.pt')

if __name__ == "__main__":
    main()
