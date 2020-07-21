# Example usage:
# python3 metrics_wrapper.py Shoes_0715_0 --nocuda --batchsize 10 --filecount 5

from metrics.fid_score import calculate_fid_given_paths
from metrics.inception_score import inception_score
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from generator import load_trained_model
from torchvision.utils import save_image
import os
import train
import torch
from tqdm import tqdm
import torchvision.transforms

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('modelnames', nargs='+', help='model names for which to calculate scores')
parser.add_argument('--nocuda', help='Disable CUDA', action='store_true')
parser.add_argument('--nogenerate', help='Disable generator of png files', action='store_true')
parser.add_argument('--nofid', help='Do not calculate FID', action='store_true')
parser.add_argument('--nois', help='Do not calculate IS', action='store_true')
parser.add_argument('--refshoes', help='Use shoe dataset as reference', action='store_true')
parser.add_argument('--refsketches', help='Use sketch dataset as reference', action='store_true')
parser.add_argument('--batchsize', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--filecount', type=int, default=-1,
                    help='Number of files to create max. -1 (default) means no limit.')


def generate_pngs(device, model_name, args):
    model, split, params = load_trained_model(os.path.join("saved_models", model_name))
    path = os.path.join('generator', model_name)
    save_path = os.path.join(path, 'pngs')
    if os.path.exists(os.path.join(path, 'ready_pngs')):
        print('ready_pngs folder already exists. Create scores with already generated images')
        return os.path.join(path, 'ready_pngs')
    try:
        os.makedirs(save_path)
    except:
        print("generate folder exists, so plots are overwritten")
    if not os.path.exists(path):
        raise (RuntimeError('Path to generated images could not be found {}'.format(path)))
    __, dataloader_test, ___, test_split = train.create_dataloaders(
        params["data_path"],
        args.batchsize,
        params["test_ratio"],
        only_classes=params.get("only_classes", None),
        split=split,
        only_one_sample=params.get("only_one_sample", False),
        load_on_request=True
    )
    model.to(device)
    count = 1
    with torch.set_grad_enabled(False):
        for batch_no, (batch_conditions, batch_inputs, batch_labels) in enumerate(
                tqdm(dataloader_test, "Visualization")):
            batch_conditions = batch_conditions.to(device)
            gauss_samples = torch.randn(batch_inputs.shape[0],
                                        batch_inputs.shape[1] * batch_inputs.shape[2] * batch_inputs.shape[3]).to(
                device)
            batch_output = model(x=gauss_samples, c=batch_conditions, rev=True)
            for i in range(batch_output.shape[0]):
                save_image(batch_output[i], os.path.join(save_path, 'img_b{}_i{}.png'.format(batch_no, i)))
                if count >= args.filecount and not args.filecount == -1:
                    try:
                        os.rename(save_path, os.path.join(path, 'ready_pngs'))
                    except:
                        raise (RuntimeError("Could not flag directory 'pngs' as ready"))
                    return os.path.join(path, 'ready_pngs')
                else:
                    count += 1

    try:
        os.rename(save_path, os.path.join(path, 'ready_pngs'))
    except:
        raise(RuntimeError("Could not flag directory 'pngs' as ready"))
    return os.path.join(path, 'ready_pngs')


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.nocuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA enabled.")
    else:
        device = torch.device('cpu')

    for model_name in args.modelnames:
        path = generate_pngs(device, model_name, args)
        if args.refshoes:
            reference_path = 'dataset/ShoeV2_F/photo/shoe'
        elif args.refsketches:
            reference_path = 'dataset/SketchyDatabase/256x256/photo'
        else:
            raise ValueError('Specify reference dataset with --refshoes or --refsketches')

        if not args.nofid:
            print("Calculating FID score for Model {}...".format(model_name))
            dims = 2048 # Pooling layer before last layer
            fid_value = calculate_fid_given_paths([path, reference_path], batch_size=args.batchsize,
                                                 cuda=device==torch.device('cuda'), dims=dims)
            print("FID: ", fid_value)
            with open(os.path.join('generator', model_name, 'metric_results.txt'), "a") as resultfile:
                resultfile.write("FID SCORE FOR N={} D={}: \n{}\n########\n\n".format(args.filecount, dims, fid_value))

        if not args.nois:
            dataset = torchvision.datasets.ImageFolder(root=os.path.join('generator', model_name),
                                                       transform=torchvision.transforms.ToTensor())
            print("Calculating inception score for Model {}...".format(model_name))
            is_value_mean, is_value_std = inception_score(dataset, device==torch.device('cuda'), args.batchsize, resize=True)
            print("IS: mean, std", is_value_mean, " ", is_value_std)
            with open(os.path.join('generator', model_name, 'metric_results.txt'), "a") as resultfile:
                resultfile.write(
                    "IS  SCORE FOR N={} MEAN:{}  STD:{}\n########\n\n".format(args.filecount, is_value_mean,
                                                                              is_value_std))

        print("DONE.")

