import torch
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import cv2
from random import randrange 
import time

plt.close('all')


def image_loader(path, plot = True):
    '''
    load image as tensor
    '''
    img = Image.open(path)
    transform = transforms.Compose([transforms.PILToTensor()])
    img = transform(img)
    img_shape = img.shape
    if plot:
        plt.figure()
        plt.imshow(img.permute(1,2,0))
        plt.title("Original image")
        plt.show()

    return img, img_shape

def get_motions_models(N, round = False, scale = 1):

    '''create N random affine motions'''
    A = torch.empty((N,2,3), dtype = torch.float64)

    for i in range(N):
    
        model = torch.rand(2,3)*scale
        model = torch.round(model) if round == True else model
        if N == 1:
            A = model
        elif N > 1:
            A[i,:,:] = model

    return A 

def get_alpha_maps(N,img, plot = True):
    ''' create N random alpha maps with shape of a circle'''
    
    radius = 25
    offset = round(radius * 2.1)
    W = img.shape[1]
    H = img.shape[0]
    
    alpha_maps = []
    for _ in range(N):
        mask = np.zeros_like(img)
        alpha_map = cv2.circle(mask,(randrange(radius,round(0.5*W)-radius, offset),randrange(radius,round(0.5*H)-radius, offset)),radius,(1,1,1),-1)
        alpha_map = torch.tensor(alpha_map)#.permute(2,0,1)
        if N == 1:
            alpha_maps = alpha_map
            
        elif N > 1:
            alpha_maps.append(alpha_map)
        
        if plot:
            plt.figure()
            plt.imshow(alpha_map.permute(1,2,0))
            plt.title("Alpha map")
            plt.show()
         
    return alpha_maps

def get_rgb_layers(N,img, alpha_maps, plot = True):
    '''take alpha maps as input, output rgb layers '''
    rgb_layers = []
    

    
    
    for i in range(N):

        rgb_layer = torch.clone(img)
        
        if N == 1:
            rgb_layer[alpha_maps == 0] = 0 
            rgb_layers = rgb_layer
            
        elif N > 1:
            rgb_layer[alpha_maps[i] == 0] = 0 
            rgb_layers.append(rgb_layer)
        
        if plot:
            plt.figure()
            plt.imshow(rgb_layer.type(torch.int32))
            plt.title("RGB layer")
            plt.show()
    #print(rgb_layers.dtype)
    return rgb_layers 

def get_optical_flow(A,rgb_layer, grid_sample = True):

    '''take as input the motion model and the rgb layer, output optic flow according to that rgb layer'''
    H = rgb_layer.shape[0]
    W = rgb_layer.shape[1]
    pixels_long = torch.nonzero(rgb_layer[:,:,0])
    pixels = pixels_long.type(torch.float32)
    flow = torch.zeros((H, W, 2)) 
    if grid_sample:
        print(A)
        row = torch.tensor([[1,0,0]])
        print(row)
        A_long = torch.cat((A,row),0)
        print(A_long)
        A_inv = torch.inverse(A_long)
        print(A_inv)
        A = A_inv[:2,:]
        print(A)

    for i in range(len(pixels)):
        
        u_x, u_y = torch.matmul(A,torch.tensor([1,pixels[i,0],pixels[i,1]]))
        flow[pixels[i,0].long(),pixels[i,1].long(),0] = u_x
        flow[pixels[i,0].long(),pixels[i,1].long(),1] = u_y
        
    
    
    return flow, pixels_long 

def forward(rgb_layer, flow, pixels, interpolation_mode = 'bilinear', grid_sample = True, plot = True):
        warped_rgb_layer = torch.zeros_like(rgb_layer)
        py = pixels[:,0]
        px = pixels[:,1]
        H = rgb_layer.shape[0]
        W = rgb_layer.shape[1]

        if interpolation_mode == 'bilinear':
            
            if grid_sample:
                grid = torch.unsqueeze(flow, dim = 0)
                sample = torch.unsqueeze(rgb_layer, dim = 0).permute(0,3,1,2)
                print()
                warped_rgb_layer = torch.nn.functional.grid_sample(sample, grid, mode = interpolation_mode, padding_mode = 'zeros', align_corners = False)
                warped_rgb_layer = torch.squeeze(warped_rgb_layer, dim = 0).permute(1,2,0)
                
            else:
                for h,w in zip(py,px):
                    x = w + flow[h, w, 0]
                    y = h + flow[h, w, 1]
                    nw = (int(torch.floor(x)), int(torch.floor(y)))
                    ne = (nw[0]+1, nw[1])
                    sw = (nw[0], nw[1]+1)
                    se = (nw[0]+1, nw[1]+1)
                    p = rgb_layer[h, w, :]
                    if nw[0] >= 0 and se[0] < W and nw[1] >= 0 and se[1] < H:
                        nw_k = (se[0]-x)*(se[1]-y)
                        ne_k = (x-sw[0])*(sw[1]-y)
                        sw_k = (ne[0]-x)*(y-ne[1])
                        se_k = (x-nw[0])*(y-nw[1])
                        warped_rgb_layer[nw[1], nw[0], :] += nw_k*p
                        warped_rgb_layer[ne[1], ne[0], :] += ne_k*p
                        warped_rgb_layer[sw[1], sw[0], :] += sw_k*p
                        warped_rgb_layer[se[1], se[0], :] += se_k*p

        
        
        if plot:
            plt.figure()
            plt.imshow(warped_rgb_layer.type(torch.int32))
            plt.title("Warped rgb layer")
            plt.show()
        return warped_rgb_layer

def get_warped_image(N, img, rgb_layers, A, plot = True, grid_sample = True):
    '''take as input the rgb layers, motion models, and alpha maps, output the warped rgb layers'''
    
    warped_image = torch.clone(img)

    for i in range(N):
        if N == 1:
            flow, pixels = get_optical_flow(A, rgb_layers)
            warped_rgb_layer = forward(rgb_layers, flow, pixels, plot = False, grid_sample = grid_sample)
            warped_image[warped_rgb_layer != 0] = warped_rgb_layer[warped_rgb_layer != 0]  
        elif N > 1:
            flow, pixels = get_optical_flow(A[i,:,:],rgb_layers[i]) 
            warped_rgb_layer = forward(rgb_layers, flow, pixels, grid_sample = grid_sample)
            warped_image[warped_rgb_layer != 0] = warped_rgb_layer[warped_rgb_layer != 0]
        if plot:
            plt.figure()
            plt.imshow(warped_rgb_layer.type(torch.int32))
            plt.title("Warped rgb layer")
            plt.show()
    if plot:
            
            plt.figure()
            plt.imshow(warped_image.type(torch.int32))
            plt.title("Warped image")
            plt.show()
    return warped_image

def backward(grad_output, im0, flow, pixels, interpolation_mode = 'bilinear', plot = True):
        
    py = pixels[:,0]
    px = pixels[:,1]
    H = grad_output.shape[0]
    W = grad_output.shape[1]
    C = grad_output.shape[2]
    im0_grad = torch.zeros_like(grad_output)
    flow_grad = torch.empty([H, W, 2])
    if interpolation_mode == 0:
        
        for h,w in zip(py,px):
                x = w + flow[h, w, 0]
                y = h + flow[h, w, 1]
                x_f = int(torch.floor(x))
                y_f = int(torch.floor(y))
                x_c = x_f+1
                y_c = y_f+1
                nw = (x_f, y_f)
                ne = (x_c, y_f)
                sw = (x_f, y_c)
                se = (x_c, y_c)
                p = im0[h, w, :]
                if nw[0] >= 0 and se[0] < W and nw[1] >= 0 and se[1] < H: # if coordinates fall into image domain HXW
                    nw_k = (se[0]-x)*(se[1]-y)
                    ne_k = (x-sw[0])*(sw[1]-y)
                    sw_k = (ne[0]-x)*(y-ne[1])
                    se_k = (x-nw[0])*(y-nw[1])
                    nw_grad = grad_output[nw[1], nw[0], :]
                    ne_grad = grad_output[ne[1], ne[0], :]
                    sw_grad = grad_output[sw[1], sw[0], :]
                    se_grad = grad_output[se[1], se[0], :]
                    im0_grad[h, w, :] += nw_k*nw_grad
                    im0_grad[h, w, :] += ne_k*ne_grad
                    im0_grad[h, w, :] += sw_k*sw_grad
                    im0_grad[h, w, :] += se_k*se_grad
                    flow_grad_x = torch.zeros(C)
                    flow_grad_y = torch.zeros(C)
                    flow_grad_x -= (y_c-y)*p*nw_grad
                    flow_grad_y -= (x_c-x)*p*nw_grad
                    flow_grad_x += (y_c-y)*p*ne_grad
                    flow_grad_y -= (x-x_f)*p*ne_grad
                    flow_grad_x -= (y-y_f)*p*sw_grad
                    flow_grad_y += (x_c-x)*p*sw_grad
                    flow_grad_x += (y-y_f)*p*se_grad
                    flow_grad_y += (x-x_f)*p*se_grad
                    flow_grad[h, w, 0] = torch.sum(flow_grad_x)
                    flow_grad[h, w, 1] = torch.sum(flow_grad_y)
    

    return im0_grad, flow_grad

if __name__ == '__main__':
    
    
    '''
    path = r'/Users/youssef/Desktop/MSc/Thesis-Code/event_segmentation/image_warping/image_folder/McS9K.png'
    N = 1
    im0 = cv2.imread(path)#[np.newaxis, :, :, :]
    im0 = torch.FloatTensor(im0).contiguous() #.permute( 2, 0, 1).contiguous()
    
    A = get_motions_models(N)
    alpha_maps = get_alpha_maps(N, im0, plot = False)
    rgb_layers = get_rgb_layers(N, im0, alpha_maps, plot = True)
    warped_image = get_warped_image(N, im0, rgb_layers, A, plot = True, grid_sample = True)
    #print(flow.shape)
    '''

    a = torch.rand(2,2)
    print(a)
    a = a.view(4,-1)
    print(a)
    

    event_list = torch.rand(2,2,4) # [batch size x N x 4]  (t, x, y, p)
    print(event_list)
    flow_idx = event_list[:, :, 1:3] # [batch size x N x 2] (x, y)
    print(flow_idx)
    flow_idx = torch.sum(flow_idx, dim = 2)
    print(flow_idx)
    flow = flow_idx.view(flow_idx.shape[0], 2, -1)
    print(flow)
    print(flow[:,1,:])
    print(flow_idx.long())
    
