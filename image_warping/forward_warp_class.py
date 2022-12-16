import torch 

class Forward_Warp:

    @staticmethod
    def forward(rgb_layer, flow, pixels, interpolation_mode = 'bilinear'):

        warped_rgb_layer = torch.zeros_like(rgb_layer)
        py = pixels[:,0]
        px = pixels[:,1]
        H = rgb_layer.shape[0]
        W = rgb_layer.shape[1]

        if interpolation_mode == 'bilinear':
            
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
        
        return warped_rgb_layer

    @staticmethod
    def backward(grad_output, im0, flow, pixels, interpolation_mode = 'bilinear'):
        
        py = pixels[:,0]
        px = pixels[:,1]
        H = grad_output.shape[0]
        W = grad_output.shape[1]
        C = grad_output.shape[2]
        im0_grad = torch.zeros_like(grad_output)
        flow_grad = torch.empty([H, W, 2])

        if interpolation_mode == 'bilinear':
            
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



