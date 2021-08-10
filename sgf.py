from torch import nn
import torch
import torchvision


def translate(G, C, F, z0, c1, m=1, max_iteration=5):
    device = z0.device
    step = 1.0 / max_iteration

    with torch.no_grad():
        image, _ = G([z0], input_is_latent=True, randomize_noise=False)
        c0 = C(image).unsqueeze(0).to(device)

        d_c = step * (c1 - c0)

    images = [image]

    _, style_dim = z0.shape
    _, c_dim = c1.shape

    z = z0
    c = c0
    c_diff = -1

    for i in range(max_iteration):
        z.requires_grad = True
        c.requires_grad = True

        grad = torch.autograd.functional.jacobian(F, (z,c))
        z_grad = grad[0][0,:,0,:]
        c_grad = grad[1][0,:,0,:]
        
        with torch.no_grad():
            d_z = c_grad @ d_c[0]
    
            d_z_j = d_z
            for j in range(m):
                d_z_j = z_grad @ d_z_j
                d_z += d_z_j

            z += d_z
            image, _ = G([z], input_is_latent=True, randomize_noise=False)
            
            c = C(image).unsqueeze(0).to(device)

            c_diff_new = (c-c1).pow(2).mean()
            
            print(c_diff_new)

            if c_diff > 0 and c_diff < c_diff_new:
                break

            images.append(image)
            c_diff = c_diff_new

    return images, z


def translate_faster(G, C, F, z0, c1, m=1, max_iteration=5):
    device = z0.device
    step = 1.0 / max_iteration

    from tqdm import tqdm

    with torch.no_grad():
        image, _ = G([z0], input_is_latent=True, randomize_noise=False)
        c0 = C(image).unsqueeze(0).to(device)

        d_c = step * (c1 - c0)

    images = [image]

    _, style_dim = z0.shape
    _, c_dim = c1.shape

    z = z0
    c = c0

    z.requires_grad = True
    c.requires_grad = True


    for i in tqdm(range(max_iteration)):
        grad = torch.autograd.functional.jacobian(F, (z,c))
        z_grad = grad[0][0,:,0,:]
        c_grad = grad[1][0,:,0,:]
        
        with torch.no_grad():
            d_z = c_grad @ d_c[0]
    
            d_z_j = d_z
            for j in range(m):
                d_z_j = z_grad @ d_z_j
                d_z += d_z_j

            z += d_z
            c += d_c

    with torch.no_grad():
        image, _ = G([z], input_is_latent=True, randomize_noise=False)
        images.append(image)

    return images, z
