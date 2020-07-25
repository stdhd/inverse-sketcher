import torch
from PIL import Image

import train
# import model
import os
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from architecture import get_model_by_params
import numpy as np
import scipy.stats
from tqdm import tqdm
import torchvision
from torchvision.utils import save_image
import PIL

def load_trained_model(folder):
    """
    Provide model and test split from given parameter dictionary in file
    :param file: path to file
    :return: model, split, params
    """
    file = os.path.join(folder, [f for f in os.listdir(folder) if ".tar" in f ][0])

    try:
        state_dicts = torch.load(file, map_location=torch.device('cpu'))
    except Exception as e:
        raise (RuntimeError("Could not load training result from file " + file + ".\n" + str(e) ))

    if not "architecture" in state_dicts:
        state_dicts["architecture"] = 'glow'
        print('Field ARCHITECTURE not found in state dict. Will use glow...')

    mod = get_model_by_params(state_dicts)
    mod.model.load_state_dict(state_dicts["model_state_dict"])

    split = (state_dicts["train_split"], state_dicts["test_split"])
    if not state_dicts.get("data_path", False):
        state_dicts["data_path"] = "dataset/SketchyDatabase/256x256"
    return mod, split, state_dicts


def saliency_map(device, model_list):
    for model_name in model_list:
        print('Saliency map from model {}'.format(model_name))
        try:
            os.makedirs(os.path.join("generator", model_name))
        except:
            print("generate folder exists, so plot is overwritten")
        model, split, params = load_trained_model(os.path.join("saved_models", model_name))

        __, dataloader_test, ___, test_split = train.create_dataloaders(
            params["data_path"],
            params["batch_size"],
            params["test_ratio"],
            only_classes=params.get("only_classes", None),
            split=split,
            only_one_sample=params.get("only_one_sample", False),
            load_on_request=True
        )
        model.to(device)
        for i, (batch_condition, batch_inputs, batch_labels) in enumerate(dataloader_test):
            for j in range(batch_condition.size()[0]):
                condition = torch.tensor(batch_condition[j].unsqueeze(0))
                input = torch.tensor(batch_inputs[j].unsqueeze(0))
                model.eval()
                condition.requires_grad_()
                gauss_samples = torch.randn(input.shape[0], input.shape[1] * input.shape[2] * input.shape[3]).to(device)
                generated = model(x=gauss_samples, c=condition, rev=True)
                loss = torch.mean(generated ** 2 / 2) - torch.mean(model.log_jacobian()) / generated.shape[1]
                loss.backward()
                saliency = condition.grad.data.abs()
                # create heatmap
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(condition.squeeze().detach().numpy(), cmap='Greys')
                ax[0].imshow(saliency[0].squeeze().detach().numpy(), cmap='Reds', alpha=0.4)
                ax[1].imshow(generated.squeeze().permute(1, 2, 0).detach().numpy())
                ax[0].axis('off')
                ax[1].axis('off')
                plt.savefig(
                    os.path.join("generator", model_name, "gradient_batch{}_{}.png".format(i, j)),
                    bbox_inches='tight')
                plt.close(fig)


def latent_gauss(model_name, data, path, bins=50):
    plt.figure(figsize=[10., 5.])

    x = np.linspace(-5, 5, bins)
    y = scipy.stats.norm.pdf(x, 0, 1)
    plt.figure()
    print(data.shape)
    hist_data, _ = np.histogram()
    #plt.title('Model: ' + model_name + ' N= ' + str(data.shape[0]))
    plt.hist(data, bins, range=[-5., 5.], density=True, alpha=0.4)
    plt.boxplot(x, data)
    plt.plot(x, y)
    #plt.bar(x, bars, alpha=0.4)



    try:
        os.makedirs(os.path.join("generator", model_name))
    except:
        print("generate folder exists, so plot is overwritten")

    plt.savefig(os.path.join("generator", model_name, "GaussianLatent_all.pdf"))
    plt.close()

def generate_multiple_for_one(device, model_name, args):
    model, split, params = load_trained_model(os.path.join("saved_models", model_name))
    path = os.path.join('generate_multiple_per_sketch', model_name)
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

            for i in range(batch_conditions.shape[0]):
                save_image(batch_conditions[i], os.path.join(save_path, 'sk_img_b{}_i{}.png'.format(batch_no, i)))

            for j in range(5):
                gauss_samples = torch.randn(batch_inputs.shape[0],
                                            batch_inputs.shape[1] * batch_inputs.shape[2] * batch_inputs.shape[3]).to(
                    device)
                batch_output = model(x=gauss_samples, c=batch_conditions, rev=True)
                for ij in range(batch_output.shape[0]):
                    save_image(batch_output[ij], os.path.join(save_path, 'img_b{}_i{}_{}.png'.format(batch_no, ij, j)))


    try:
        os.rename(save_path, os.path.join(path, 'ready_pngs'))
    except:
        raise(RuntimeError("Could not flag directory 'pngs' as ready"))
    return os.path.join(path, 'ready_pngs')


def generate_from_image(image_name, model_list):

    for model_name in model_list:
        print('Generate from model {}'.format(model_name))
        model, split, params = load_trained_model(os.path.join("saved_models", model_name))
        sketch = Image.open(os.path.join('comparing_generator', image_name))

        sketch = sketch.convert("L")
        model.eval()
        with torch.set_grad_enabled(False):
            tensor_transform = torchvision.transforms.ToTensor()
            sketch = sketch.resize((64, 64))
            sketch = tensor_transform(sketch)
            #sketch = 1 - sketch
            sketch += 0.002 * torch.randn_like(sketch)
            sketch = torch.unsqueeze(sketch, 0)
            for i in range(10):
                noise = torch.randn(1, 3 * 64 * 64)
                result = model(x=noise, c=sketch, rev=True).squeeze()
                trans2 = transforms.ToPILImage()
                result_img = trans2(result)
                result_img.save(os.path.join('comparing_generator', "{}_model_{}-{}.png".format(image_name.replace(".", ""), model_name, i)), "PNG")


def generate_from_testset(device, model_list):
    for model_name in model_list:
        print('Generate from model {}'.format(model_name))
        model, split, params = load_trained_model(os.path.join("saved_models", model_name))

        __, dataloader_test, ___, test_split = train.create_dataloaders(
            params["data_path"],
            params["batch_size"],
            params["test_ratio"],
            only_classes=params.get("only_classes", None),
            split=split,
            only_one_sample=params.get("only_one_sample", False),
            load_on_request=True
        )
        model.to(device)

        try:
            os.makedirs(os.path.join("generator", model_name))
        except:
            print("'generate' folder exists, so plot is overwritten")
        model.eval()
        with torch.set_grad_enabled(False):
            for batch_no, (batch_conditions, batch_inputs, batch_labels) in enumerate(
                    tqdm(dataloader_test, "Visualization")):
                batch_conditions = batch_conditions.to(device)
                gauss_samples = torch.randn(batch_inputs.shape[0],
                                            batch_inputs.shape[1] * batch_inputs.shape[2] * batch_inputs.shape[3]).to(
                    device)
                batch_output = model(x=gauss_samples, c=batch_conditions, rev=True)
                subset = 0
                fig, axes = plt.subplots(nrows=3, ncols=3)
                for i in range(batch_inputs.shape[0]):


                    condition_image = transforms.ToPILImage()(batch_conditions[i].cpu().detach()).convert('L')
                    generated_image = transforms.ToPILImage()(batch_output[i].cpu().detach()).convert("RGB")
                    original = transforms.ToPILImage()(batch_inputs[i].cpu().detach()).convert("RGB")
                    axes[i % 3, 0].imshow(condition_image, cmap='gray')
                    axes[i % 3, 1].imshow(generated_image)
                    axes[i % 3, 2].imshow(original)

                    axes[i % 3, 0].axis('off')
                    axes[i % 3, 1].axis('off')
                    axes[i % 3, 2].axis('off')
                    if i % 3 == 2 or i == batch_inputs.shape[0] - 1:
                        plt.savefig(
                            os.path.join("generator", model_name, "out_batch{}_{}.pdf".format(batch_no, subset)),
                            bbox_inches='tight')
                        subset += 1
                        plt.close(fig)
                        fig, axes = plt.subplots(nrows=3, ncols=3)
                    if i == batch_inputs.shape[0]:
                        plt.close(fig)


def sanity_check(device, model_list):
    for model_name in model_list:
        print('Generate from model {}'.format(model_name))
        model, split, params = load_trained_model(os.path.join("saved_models", model_name))

        dataloader_train, dataloader_test, ___, test_split = train.create_dataloaders(
            params["data_path"],
            params["batch_size"],
            params["test_ratio"],
            only_classes=params.get("only_classes", None),
            split=split,
            only_one_sample=params.get("only_one_sample", False),
            load_on_request=True
        )
        model.to(device)

        with torch.set_grad_enabled(False):
            sanity_data = np.array([])

            for batch_no, (batch_conditions, batch_inputs, batch_labels) in enumerate(dataloader_train):
                batch_conditions, batch_inputs = batch_conditions.to(device), batch_inputs.to(device)
                sanity_check = model(x=batch_inputs, c=batch_conditions, rev=False)

                sanity_data = np.append(sanity_data, sanity_check.cpu().detach().numpy()[:, ..., 3])

            # Plot sanity check data
                break
            latent_gauss(model_name, sanity_data, "")


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def save__architecture_and_parameters(model, param_dict):
    checkpoint = {'model': get_model_by_params(param_dict),
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('modelnames', nargs='+', help='model names to generate from')
    parser.add_argument('--nocuda', help='Disable CUDA', action='store_true')
    parser.add_argument('--generate', help='Generate from test set', action='store_true')
    parser.add_argument('--sanity', help='Only sanity check, no generation', action='store_true')
    parser.add_argument('--saliencymap', help='Draw saliency map', action='store_true')
    parser.add_argument('--frompng', help='Generate images from various models to one sketch', action='store_true')
    parser.add_argument('--multiple', help='Generate multiple images from test set for each sketch', action='store_true')
    parser.add_argument('--batchsize', type=int, default=50,
                        help='Batch size to use')
    args = parser.parse_args()
    device = None

    if not args.nocuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA enabled.")
    else:
        device = torch.device('cpu')
        print("CUDA disabled.")
    if not args.modelnames is None:
        model_list = args.modelnames
    else:
        print("No model name specified in command line arguments. Will use hard-coded mode list...")
        model_list = ["default_0710_0g"]

    if args.multiple:
        generate_multiple_for_one(device, model_list[0], args)

    if args.saliencymap:
        saliency_map(device, model_list)

    elif args.generate:
        generate_from_testset(device, model_list)
    elif args.sanity:
        sanity_check(device, model_list)
    elif args.frompng:
        generate_from_image('2325150230_2.png', model_list)
    else:
        generate_from_testset(device, model_list)

