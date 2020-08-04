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
import numpy as np
from skimage import io, color

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
parser.add_argument('--combine', help="Whether to combine a bw and color model", action='store_true')
parser.add_argument('--filecount', type=int, default=-1,
                    help='Number of files to create max. -1 (default) means no limit.')

scale = (25.6, 11.2, 16.8)
bias =  (47.5, 2.4, 7.4)


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
        load_on_request=True,
        bw=params["model_params"].get("bw")
    )
    model.to(device)
    count = 0
    with torch.set_grad_enabled(False):
        for batch_no, (batch_conditions, batch_inputs, batch_labels) in enumerate(
                tqdm(dataloader_test, "Visualization")):
            batch_conditions = batch_conditions.to(device)
            gauss_samples = torch.randn(batch_inputs.shape[0],
                                        batch_inputs.shape[1] * batch_inputs.shape[2] * batch_inputs.shape[3]).to(
                device)
            batch_output = model(x=gauss_samples, c=batch_conditions, rev=True)
            gen = batch_output
            true = batch_inputs
            if params["model_params"].get("bw"):
                gen = torch.empty(batch_output.shape[0], 1, 64, 64)
                true = torch.empty(batch_output.shape[0], 1, 64, 64)

                gen[:,:,::2,::2] = batch_output[:,0,:,:].unsqueeze(1)
                gen[:,:,1::2,::2] = batch_output[:,1,:,:].unsqueeze(1)
                gen[:,:,::2,1::2] = batch_output[:,2,:,:].unsqueeze(1)
                gen[:,:,1::2,1::2] = batch_output[:,3,:,:].unsqueeze(1)
            elif params["model_params"].get("color"):
                gen = torch.cat((batch_conditions, batch_output), dim=1).cpu().data.numpy()
                for i in range(3):
                    gen[:, i] = gen[:, i] * scale[i] + bias[i]

                gen[:, 1:] = gen[:, 1:].clamp_(-128, 128)
                gen[:, 0] = gen[:, 0].clamp_(0, 100.)
                gen = torch.stack([torch.from_numpy(color.lab2rgb(np.transpose(l, (1, 2, 0))).transpose(2, 0, 1)) for l in gen], dim=0)

            for i in range(gen.shape[0]):
                save_image(gen[i], os.path.join(save_path, 'img_b{}_i{}.png'.format(batch_no, i)))
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


def generate_combined(device, args):
    model_list = args.modelnames
    print('Combining bw model {} and color model {}'.format(model_list[0], model_list[1]))
    model, split, params = load_trained_model(os.path.join("saved_models", model_list[0]))

    dataloader_train, dataloader_test, ___, test_split = train.create_dataloaders(
        params["data_path"], #"dataset/edges2shoes/",
        params["batch_size"],
        params["test_ratio"],
        only_classes=params.get("only_classes", None),
        split=split,
        only_one_sample=params.get("only_one_sample", False),
        load_on_request=True,
        bw=False,
        color=False,
    )
    model.to(device)
    path = os.path.join("generator", "combined_" + model_list[0].split("/")[0] + "_" + model_list[1].split("/")[0])
    save_path = os.path.join(path, 'pngs')
    if os.path.exists(os.path.join(path, 'ready_pngs')):
        print('ready_pngs folder already exists. Create scores with already generated images')
        return os.path.join(path, 'ready_pngs')
    try:
        os.makedirs(save_path)
    except:
        print("generate folder exists, so plots are overwritten")
    model.eval()
    images_bw = []
    gen_bw = []
    orig_cond = []
    with torch.set_grad_enabled(False):
        for batch_no, (batch_conditions, batch_inputs, batch_labels) in enumerate(
                tqdm(dataloader_test, "Visualization")):
            batch_conditions = batch_conditions.to(device)
            gauss_samples = torch.randn(batch_inputs.shape[0],
                                        1 * batch_inputs.shape[2] * batch_inputs.shape[3]).to(device)
            batch_output = model(x=gauss_samples, c=batch_conditions, rev=True)
            gen = torch.empty(batch_output.shape[0], 1, 64, 64)
            true = torch.empty(batch_output.shape[0], 1, 64, 64)

            gen[:,:,::2,::2] = batch_output[:,0,:,:].unsqueeze(1)
            gen[:,:,1::2,::2] = batch_output[:,1,:,:].unsqueeze(1)
            gen[:,:,::2,1::2] = batch_output[:,2,:,:].unsqueeze(1)
            gen[:,:,1::2,1::2] = batch_output[:,3,:,:].unsqueeze(1)
            gen_bw.append(gen)
            images_bw.append(batch_inputs)
            orig_cond.append(batch_conditions)
    #gen_bw, images_bw = torch.cat(gen_bw, dim = 0), torch.cat(images_bw, dim = 0)
    print("Coloring Images")
    model, split, params = load_trained_model(os.path.join("saved_models", model_list[1]))
    model.to(device)

    model.eval()
    count = 0
    scale = (25.6, 11.2, 16.8)
    bias =  (47.5, 2.4, 7.4)
    with torch.set_grad_enabled(False):
        for batch_no, (batch_inputs, batch_conditions, old_cond) in enumerate(
                tqdm(zip(images_bw, gen_bw, orig_cond), "Visualization")):
            batch_conditions = batch_conditions.to(device)
            gauss_samples = torch.randn(batch_inputs.shape[0],
                                        2 * batch_inputs.shape[2] * batch_inputs.shape[3]).to(device)
            for j in range(len(batch_conditions)):
                image = batch_conditions[j].cpu().numpy()

                image = np.transpose(image, (1,2,0))
                if image.shape[2] != 3:
                    image = np.stack([image[:,:,0]]*3, axis=2)
                image = color.rgb2lab(image).transpose((2, 0, 1))
                for i in range(3):
                    image[i] = (image[i] - bias[i]) / scale[i]
                image = torch.Tensor(image)
                batch_conditions[j] = image[0].to(device)
            batch_output = model(x=gauss_samples, c=batch_conditions, rev=True)
            gen = batch_output
            true = batch_inputs
            gen = torch.cat((batch_conditions, batch_output), dim=1)
            for i in range(3):
                gen[:, i] = gen[:, i] * scale[i] + bias[i]
            gen[:, 1:] = gen[:, 1:].clamp_(-128, 128)
            gen[:, 0] = gen[:, 0].clamp_(0, 100.)
            gen = gen.cpu().data.numpy()
            gen = torch.stack([torch.from_numpy(color.lab2rgb(np.transpose(l, (1, 2, 0))).transpose(2, 0, 1)) for l in gen], dim=0)
            for i in range(gen.shape[0]):
                save_image(gen[i], os.path.join(save_path, 'img_b{}_i{}.png'.format(batch_no, i)))
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
        if not args.combine:
            path = generate_pngs(device, model_name, args)
        else:
            path = generate_combined(device, args)
        if args.refshoes:
            reference_path = 'dataset/ShoeV2_F/photo/'
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
            dataset = torchvision.datasets.ImageFolder(root="/".join(path.split("/")[:-1]),
                                                       transform=torchvision.transforms.ToTensor())
            #Calculate IS of dataset as reference (generated IS should not exceed dataset IS)
            #dataset = torchvision.datasets.ImageFolder(root=os.path.join(reference_path),
            #                                           transform=torchvision.transforms.ToTensor())
            print("Calculating inception score for Model {}...".format(model_name))
            is_value_mean, is_value_std = inception_score(dataset, device==torch.device('cuda'), args.batchsize, resize=True)
            print("IS: mean, std", is_value_mean, " ", is_value_std)
            with open(os.path.join('generator', model_name, 'metric_results.txt'), "a") as resultfile:
                resultfile.write(
                    "IS  SCORE FOR N={} MEAN:{}  STD:{}\n########\n\n".format(args.filecount, is_value_mean,
                                                                              is_value_std))

        print("DONE.")
        if args.combine:
            break
