import torch
import train
#import model
import os
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from architecture import get_model_by_params
import numpy as np
import scipy.stats
from tqdm import tqdm

def load_trained_model(file):
    """
    Provide model and test split from given parameter dictionary in file
    :param file: path to file
    :return: model, split, params
    """

    try:
        state_dicts = torch.load(file, map_location=torch.device('cpu'))
    except:
        raise (RuntimeError("Could not load training result from file " + file + "."))
    if state_dicts.get("architecture", False):
        mod = get_model_by_params(state_dicts.get("architecture"))#model.cINN(**state_dicts.get("model_params")).to(device)
        mod.model.load_state_dict(state_dicts["model_state_dict"])

    else:
        try:
            mod = get_model_by_params("glow")
            mod.model.load_state_dict(state_dicts["model_state_dict"])

        except:
            mod = get_model_by_params("aio")
            mod.model.load_state_dict(state_dicts["model_state_dict"])

    split = (state_dicts["train_split"], state_dicts["test_split"])
    return mod, split, state_dicts


def latent_gauss(model_name, data, path, bins=50):
    plt.figure(figsize=[10., 5.])
    x = np.linspace(-5, 5, 1000)
    y = scipy.stats.norm.pdf(x, 0, 1)

    plt.figure()
    plt.title('Model: '+ model_name + ' N= ' + str(data.shape[0]))
    plt.hist(data, bins, range=[-5., 5.], density=True)
    plt.plot(x, y, color='coral')
    plt.tight_layout()

    try:
        os.makedirs(os.path.join("generator", model_name))
    except:
        print("generate folder exists, so plot is overwritten")

    plt.savefig(os.path.join("generator", model_name, "GaussianLatent.pdf"))


def generate_from_testset(device, model_list):
    for model_name in model_list:
        print('Generate from model {}'.format(model_name))

        model, split, params = load_trained_model(os.path.join("saved_models", model_name, "default.tar"))
        __, dataloader_test, ___, test_split = train.create_dataloaders(
            "dataset/SketchyDatabase"
            "/256x256",
            params["batch_size"],
            params["test_ratio"],
            only_classes=params.get("only_classes", None),
            split=split,
            only_one_sample=params.get("only_one_sample", False))
        model.to(device)

        try:
            os.makedirs(os.path.join("generator", model_name))
        except:
            print("generate folder exists, so plot is overwritten")

        with torch.set_grad_enabled(False):
            for batch_no, (batch_conditions, batch_inputs, batch_labels) in enumerate(tqdm(dataloader_test, "Visualization")):
                batch_conditions = batch_conditions.to(device)
                gauss_samples = torch.randn(batch_inputs.shape[0],
                                            batch_inputs.shape[1] * batch_inputs.shape[2] * batch_inputs.shape[3]).to(
                    device)
                batch_output = model(x=gauss_samples, c=batch_conditions, rev=True)
                subset = 0
                fig, axes = plt.subplots(nrows=3, ncols=2)
                for i in range(batch_inputs.shape[0]):

                    condition_image = transforms.ToPILImage()(batch_conditions[i].cpu().detach()).convert('L')
                    generated_image = transforms.ToPILImage()(batch_output[i].cpu().detach()).convert("RGB")
                    axes[i%3, 0].imshow(condition_image, cmap='gray')

                    axes[i%3, 1].imshow(generated_image)

                    axes[i%3, 0].axis('off')
                    axes[i%3, 1].axis('off')

                    if i % 3 == 0 or i == batch_inputs.shape[0] - 1:
                        plt.savefig(
                            os.path.join("generator", model_name, "out_batch{}_{}.pdf".format(batch_no, subset)),
                            bbox_inches='tight')
                        subset += 1
                        plt.close(fig)
                        fig, axes = plt.subplots(nrows=3, ncols=2)
                    if i == batch_inputs.shape[0]:
                        plt.close(fig)
                if batch_no > 10:
                    break


def sanity_check(device, model_list):
    for model_name in model_list:
        print('Generate from model {}'.format(model_name))
        model, split, params = load_trained_model(os.path.join("saved_models", model_name, "default.tar"))

        dataloader_train, dataloader_test, ___, test_split = train.create_dataloaders(
            "dataset/SketchyDatabase"
            "/256x256",
            params["batch_size"],
            params["test_ratio"],
            only_classes=params.get("only_classes", None),
            split=split,
            only_one_sample=params.get("only_one_sample", False)
        )
        model.to(device)

        with torch.set_grad_enabled(False):
            sanity_data = np.array([])
            for batch_no, (batch_conditions, batch_inputs, batch_labels) in enumerate(tqdm(dataloader_train, "Sanity Check")):
                batch_inputs, batch_conditions = batch_inputs.to(device), batch_conditions.to(device)
                sanity_check = model(x=batch_inputs, c=batch_conditions, rev=False)

                sanity_data = np.append(sanity_data, sanity_check.cpu().detach().numpy()[:, ..., 0])
            # Plot sanity check data
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


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--nocuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()
    device = None
    if not args.nocuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA enabled.")
    else:
        device = torch.device('cpu')
        print("CUDA disabled.")

    model_list = ["default_0709_0", "default_0709_8"]

    sanity_check(device, model_list)
    generate_from_testset(device, model_list)
