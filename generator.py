import torch
import train
import os
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
from architecture import get_model_by_params
import numpy as np
import scipy.stats
from tqdm import tqdm
from torchvision.utils import save_image
import PIL
from skimage import io, color

scale = (25.6, 11.2, 16.8)
bias =  (47.5, 2.4, 7.4)


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
    #plt.hist(data, bins, range=[-5., 5.], density=True, alpha=0.4)
    #plt.boxplot(x, data)
    plt.plot(x, y)
    # plt.hist gives you the entries, edges
    # and drawables we do not need the drawables:
    entries, edges, _ = plt.hist(data.reshape(data.shape[0] * data.shape[1]), bins=bins, range=[-5, 5], density=True, color='gray')

    # create histograms for all indivisual pixels
    histograms = np.zeros((data.shape[1], bins))
    for i in range(data.shape[1]):
        histograms[i], _ = np.histogram(data[:, i], bins=bins, density=True)

    std_devs = np.std(histograms, axis=0)
    # calculate bin centers
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    plt.errorbar(bin_centers, entries, yerr=std_devs, fmt='.', elinewidth=0.2, markersize=0.2)

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

                save_image(1-batch_conditions[i], os.path.join(save_path, 'sk_img_b{}_i{}.png'.format(batch_no, i)))

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


def generate_from_testset(device, model_list):
    for model_name in model_list:
        print('Generate from model {}'.format(model_name))
        model, split, params = load_trained_model(os.path.join("saved_models", model_name))

        dataloader_train, dataloader_test, ___, test_split = train.create_dataloaders(
            params["data_path"], #"dataset/edges2shoes/",
            params["batch_size"],
            params["test_ratio"],
            only_classes=params.get("only_classes", None),
            split=split,
            only_one_sample=params.get("only_one_sample", False),
            load_on_request=True,
            bw=params["model_params"].get("bw"),
            color=params["model_params"].get("color")
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
                gen = batch_output
                true = batch_inputs

                if params["model_params"].get("bw"):
                    gen = torch.empty(batch_output.shape[0], 1, 64, 64)
                    true = torch.empty(batch_output.shape[0], 1, 64, 64)

                    gen[:,:,::2,::2] = batch_output[:,0,:,:].unsqueeze(1)
                    gen[:,:,1::2,::2] = batch_output[:,1,:,:].unsqueeze(1)
                    gen[:,:,::2,1::2] = batch_output[:,2,:,:].unsqueeze(1)
                    gen[:,:,1::2,1::2] = batch_output[:,3,:,:].unsqueeze(1)
                    true[:,:,::2,::2] = batch_inputs[:,0,:,:].unsqueeze(1)
                    true[:,:,1::2,::2] = batch_inputs[:,1,:,:].unsqueeze(1)
                    true[:,:,::2,1::2] = batch_inputs[:,2,:,:].unsqueeze(1)
                    true[:,:,1::2,1::2] = batch_inputs[:,3,:,:].unsqueeze(1)
                elif params["model_params"].get("color"):
                    gen = torch.cat((batch_conditions, batch_output), dim=1).cpu().data.numpy()
                    for i in range(3):
                        gen[:, i] = gen[:, i] * scale[i] + bias[i]

                    gen[:, 1:] = gen[:, 1:].clamp_(-128, 128)
                    gen[:, 0] = gen[:, 0].clamp_(0, 100.)
                    gen = torch.stack([torch.from_numpy(color.lab2rgb(np.transpose(l, (1, 2, 0))).transpose(2, 0, 1)) for l in gen], dim=0)
                    for i in range(3):
                        true[:, i] = true[:, i] * scale[i] + bias[i]
                    true = torch.cat((batch_conditions, batch_inputs.to(device)), dim=1).cpu().data.numpy()
                    true = torch.stack([torch.from_numpy(color.lab2rgb(np.transpose(l, (1, 2, 0))).transpose(2, 0, 1)) for l in true], dim=0)


                fig, axes = plt.subplots(nrows=3, ncols=3)
                for i in range(batch_inputs.shape[0]):
                    condition_image = transforms.ToPILImage()(1 - batch_conditions[i].cpu().detach()).convert('L')
                    generated_image = transforms.ToPILImage()(gen[i].cpu().detach()).convert("RGB")
                    original = transforms.ToPILImage()(true[i].cpu().detach()).convert("RGB")
                    axes[i % 3, 0].imshow(condition_image, cmap='gray')
                    axes[i % 3, 1].imshow(generated_image)#, cmap="gray")
                    axes[i % 3, 2].imshow(original)#, cmap="gray")

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

def generate_combined(device, model_list):
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
    try:
        os.makedirs(os.path.join("generator", "combined_" + model_list[0].split("/")[0] + "_" + model_list[1].split("/")[0]))
    except:
        print("'generate' folder exists, so plot is overwritten")
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

            if batch_no > 9:
                break
    #gen_bw, images_bw = torch.cat(gen_bw, dim = 0), torch.cat(images_bw, dim = 0)
    print("Coloring Images")
    model, split, params = load_trained_model(os.path.join("saved_models", model_list[1]))
    model.to(device)
    model.eval()
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
            subset = 0
            gen = batch_output
            true = batch_inputs
            gen = torch.cat((batch_conditions, batch_output), dim=1)
            for i in range(3):
                gen[:, i] = gen[:, i] * scale[i] + bias[i]

            gen[:, 1:] = gen[:, 1:].clamp_(-128, 128)
            gen[:, 0] = gen[:, 0].clamp_(0, 100.)

            gen = gen.cpu().data.numpy()
            gen = torch.stack([torch.from_numpy(color.lab2rgb(np.transpose(l, (1, 2, 0))).transpose(2, 0, 1)) for l in gen], dim=0)
            subset = 0
            fig, axes = plt.subplots(nrows=3, ncols=3)
            for i in range(batch_inputs.shape[0]):

                condition_image = transforms.ToPILImage()(old_cond[i].cpu().detach()).convert('L')
                generated_image = transforms.ToPILImage()(gen[i].cpu().detach()).convert("RGB")
                original = transforms.ToPILImage()(true[i].cpu().detach()).convert("RGB")
                axes[i % 3, 0].imshow(condition_image, cmap='gray')
                axes[i % 3, 1].imshow(generated_image)#, cmap="gray")
                axes[i % 3, 2].imshow(original)#, cmap="gray")

                axes[i % 3, 0].axis('off')
                axes[i % 3, 1].axis('off')
                axes[i % 3, 2].axis('off')
                if i % 3 == 2 or i == batch_inputs.shape[0] - 1:
                    plt.savefig(
                        os.path.join("generator", "combined_" + model_list[0].split("/")[0] + "_" + model_list[1].split("/")[0], "out_batch{}_{}.pdf".format(batch_no, subset)),
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
            sanity_data = np.array([]).reshape(0, 12288)

            for batch_no, (batch_conditions, batch_inputs, batch_labels) in enumerate(dataloader_train):
                batch_conditions, batch_inputs = batch_conditions.to(device), batch_inputs.to(device)
                sanity_check = model(x=batch_inputs, c=batch_conditions, rev=False)
                print(sanity_check.size())

                sanity_data = np.concatenate((sanity_data, sanity_check.cpu().detach().numpy()))

            # Plot sanity check data
                break

           # sanity_data = np.mean(sanity_data, axis=0)

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
    parser.add_argument('--multiple', help='Generate multiple images from test set for each sketch', action='store_true')
    parser.add_argument('--batchsize', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--combine', help='Whether to combine a bw and color model', action='store_true')
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
        raise(RuntimeError("No model name specified in command line arguments."))

    if args.multiple:
        generate_multiple_for_one(device, model_list[0], args)

    if args.saliencymap:
        saliency_map(device, model_list)

    elif args.generate:
        generate_from_testset(device, model_list)
    elif args.sanity:
        sanity_check(device, model_list)
    elif args.combine:
        generate_combined(device, model_list)
    else:
        generate_from_testset(device, model_list)
