import torch
import train
import model
import os
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from architecture import get_model_by_name
import numpy as np
import scipy.stats

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

    mod = get_model_by_name('')#model.cINN(**state_dicts.get("model_params")).to(device)
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
            for batch_no, (batch_conditions, batch_inputs, batch_labels) in enumerate(dataloader_test):
                gauss_samples = torch.randn(batch_inputs.shape[0],
                                            batch_inputs.shape[1] * batch_inputs.shape[2] * batch_inputs.shape[3]).to(
                    device)
                batch_output = model(x=gauss_samples, c=batch_conditions, rev=True)

                subset = 0
                fig, axes = plt.subplots(nrows=3, ncols=2)
                for i in range(batch_inputs.shape[0]):
                    condition_image = transforms.ToPILImage()(batch_conditions[i]).convert('L')
                    generated_image = transforms.ToPILImage()(batch_output[i]).convert("RGB")
                    axes[i%3, 0].imshow(condition_image)
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
            for batch_no, (batch_conditions, batch_inputs, batch_labels) in enumerate(dataloader_train):
                sanity_check = model(x=batch_inputs, c=batch_conditions, rev=False)

                sanity_data = np.append(sanity_data, sanity_check.numpy()[:, ..., 0])
            # Plot sanity check data
            latent_gauss(model_name, sanity_data, "")


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

    model_list = ["default_0704_50"]

    #sanity_check(device, model_list)
    generate_from_testset(device, model_list)


