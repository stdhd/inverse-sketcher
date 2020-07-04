import torch
import train
#import model
import os
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
from architecture import get_model_by_name



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

    mod = get_model_by_name("") #model.cINN(**state_dicts.get("model_params")).to(device)
    mod.model.load_state_dict(state_dicts["model_state_dict"])
    split = (state_dicts["train_split"], state_dicts["test_split"])
    return mod, split, state_dicts


if __name__== "__main__":
    # Determine, whether cuda will be enabled
    # use x.to_device(args.device)
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


    evaluate_models = ["default_0703_0"]

    for model_name in evaluate_models:
        print('Generate from model {}'.format(model_name))

        model, split, params = load_trained_model(os.path.join("saved_models", model_name, "default.tar"))
        __, dataloader_test, ___, test_split = train.create_dataloaders(
            "dataset/SketchyDatabase"
            "/256x256",
            params["batch_size"],
            params["test_ratio"],
            split=split)
        model.to(device)
        with torch.set_grad_enabled(False):
            for batch_no, (batch_conditions, batch_inputs, batch_labels) in enumerate(dataloader_test):
                batch_conditions = batch_conditions.to(device)
                gauss_samples = torch.randn(batch_inputs.shape[0], batch_inputs.shape[1]*batch_inputs.shape[2]*batch_inputs.shape[3]).to(device)
                batch_output = model(gauss_samples, batch_conditions, rev = True)

                if batch_no == 0:
                    fig, axes = plt.subplots(nrows=1, ncols=2)
                    im = transforms.ToPILImage()(batch_conditions[0].cpu().detach()).convert('LA')
                    im2 = transforms.ToPILImage()(batch_output[0].cpu().detach()).convert("RGB")

                    axes[0].imshow(im)
                    axes[1].imshow(im2)
                    plt.show()
                    break
