import numpy as np
import cv2
import os , time, pickle
from sklearn.model_selection import train_test_split
import random , json
import matplotlib.pyplot as plt
from skimage import measure
from skimage.draw import polygon
from tensorflow.keras.utils import to_categorical
from skimage.io import imread
import pandas as pd
from scipy import ndimage
from model import satellite_unet
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def custom_rotation_ppm(patch,poly,mask):
    degs = np.random.randint(low=1,high=4)
    patch = np.rot90(patch,k=degs)
    poly = np.rot90(poly,k=degs)
    mask = np.rot90(mask,k=degs)
    return patch, poly,  mask


def custom_rotation_pm(patch,mask):
    degs = np.random.randint(low=1,high=4)
    patch = np.rot90(patch,k=degs)
    mask = np.rot90(mask,k=degs)
    return patch, mask

def random_point_in_poly(polyext,img_size):
    #generates a random point for the polygon contour
    overlay = np.zeros((img_size+20,img_size+20))
    tmp_overlay = np.zeros((img_size+20,img_size+20))
    p_ctr = np.array(polyext).reshape((-1,1,2)).astype(np.int32)
    p_ctr_v2 = p_ctr.copy() 
    p_ctr_v2[:,:,0] = p_ctr[:,:,1]
    p_ctr_v2[:,:,1] = p_ctr[:,:,0]
    rr, cc = polygon(p_ctr_v2[:,:,1]+1, p_ctr_v2[:,:,0]+1)
    # cv2.fillPoly(overlay, [p_ctr_v2],(255,0,0))
    tmp_overlay[rr, cc]= 255 
    #for now return centroid
    y_pt , x_pt = int(np.sum(rr)/float(len(rr))) , int(np.sum(cc)/float(len(cc)))
    y_pt, x_pt = y_pt-10, x_pt-10
    if y_pt <0 :
        y_pt =0
    if x_pt <0:
        x_pt=0
    #return 0, 0, tmp_overlay[10:img_size+10,10:img_size+10]/255.0
    return y_pt, x_pt, tmp_overlay[10:img_size+10,10:img_size+10]/255.0

def random_point_in_poly_v2(polyext,img_size):
# def random_point_in_poly_v2(polyext,img_size,pidx):
    #generates a random point for the polygon contour
    overlay = np.zeros((img_size+20,img_size+20))
    tmp_overlay = np.zeros((img_size+20,img_size+20))
    p_ctr = np.array(polyext).reshape((-1,1,2)).astype(np.int32)
    p_ctr_v2 = p_ctr.copy() 
    p_ctr_v2[:,:,0] = p_ctr[:,:,1]
    p_ctr_v2[:,:,1] = p_ctr[:,:,0]
    tmp_rr, tmp_cc = polygon(p_ctr_v2[:,:,1]+1, p_ctr_v2[:,:,0]+1)
    # cv2.fillPoly(overlay, [p_ctr_v2],(255,0,0))
    tmp_overlay[tmp_rr, tmp_cc]= 255
    rr, cc = np.where(tmp_overlay==255) # need to confirm (x,y) vs (y,x)
    # np.random.seed(pidx)
    idx = np.random.randint(low=0,high=len(rr)) 
    y_pt , x_pt = rr[idx], cc[idx]
    #for now return centroid
    # y_pt , x_pt = int(np.sum(rr)/float(len(rr))) , int(np.sum(cc)/float(len(cc)))
    y_pt, x_pt = y_pt-10, x_pt-10
    if y_pt <0 :
        y_pt =0
    if x_pt <0:
        x_pt=0
    return y_pt, x_pt, tmp_overlay[10:img_size+10,10:img_size+10]/255.0


def get_point_outside_bbox(cur_indexes):
    if np.random.random() > 0.5:
        return np.max(cur_indexes) + np.random.randint(low=1,high=5)
    else:
        return np.min(cur_indexes) - np.random.randint(low=1,high=5)




def random_point_in_poly_v3(polyext,img_size):
    overlay = np.zeros((img_size+20,img_size+20))
    tmp_overlay_a = np.zeros((img_size+20,img_size+20))
    # import ipdb; ipdb.set_trace()
    p_ctr = np.array(polyext).reshape((-1,1,2)).astype(np.int32)
    p_ctr_v2 = p_ctr.copy() 
    p_ctr_v2[:,:,0] = p_ctr[:,:,1]
    p_ctr_v2[:,:,1] = p_ctr[:,:,0]
    tmp_rr, tmp_cc = polygon(p_ctr_v2[:,:,1]+1, p_ctr_v2[:,:,0]+1)
    # if len(tmp_rr)==0 | len(tmp_cc) == 0:
    #     import ipdb; ipdb.set_trace()
    # cv2.fillPoly(overlay, [p_ctr_v2],(255,0,0))
    tmp_overlay_a[tmp_rr, tmp_cc]= 255
    # if np.random.normal() > 0: 
    #     # 
    #     # need to confirm (x,y) vs (y,x)
    #     min_x = np.min(tmp_cc) - 5
    #     max_x = np.max(tmp_cc) + 5
    #     min_y = np.min(tmp_rr) - 5
    #     max_y = np.max(tmp_rr) + 5
    #     tmp_overlay_b = np.zeros((img_size+20,img_size+20))
    #     tmp_overlay_b[min_y:max_y,min_x:max_x] = 255
    #     # tmp_overlay_b[tmp_rr-5,tmp_cc+5] = 255
    #     # tmp_overlay_b[tmp_rr-5,tmp_cc-5] = 255
    #     # tmp_overlay_b[tmp_rr+5,tmp_cc-5] = 255
    #     # tmp_overlay_b[tmp_rr+5,tmp_cc+5] = 255
    #     tmp_overlay_b[tmp_overlay_a==255] = 0
    #     rr, cc = np.where(tmp_overlay_b==255)
    #     # np.random.seed(pidx)
    # else:
    rr, cc = np.where(tmp_overlay_a==255) 

    idx = np.random.randint(low=0,high=len(rr))
    y_pt , x_pt = rr[idx], cc[idx] 
    y_pt, x_pt = y_pt-10, x_pt-10
    if y_pt <0 :
        y_pt =0
    if x_pt <0:
        x_pt=0

    if y_pt >127 :
        y_pt =127
    if x_pt >127:
        x_pt=127
    
    return y_pt, x_pt, tmp_overlay_a[10:img_size+10,10:img_size+10]/255.0


def misaligned_polygon(polyext,img_size):
    #generates a random point for the polygon contour
    overlay = np.zeros((img_size+20,img_size+20))
    misaligned_overlay = np.zeros((img_size+20,img_size+20))
    # import ipdb; ipdb.set_trace()
    # overlay = np.zeros((img_size,img_size))
    # misaligned_overlay = np.zeros((img_size,img_size))
    p_ctr = np.array(polyext).reshape((-1,1,2)).astype(np.int32)
    p_ctr_v2 = p_ctr.copy() 
    p_ctr_v2[:,:,0] = p_ctr[:,:,1]
    p_ctr_v2[:,:,1] = p_ctr[:,:,0]
    tmp_rr, tmp_cc = polygon(p_ctr_v2[:,:,1]+1, p_ctr_v2[:,:,0]+1)
    # shift_row = np.random.randint(low=-10,high=10)
    # shift_col= np.random.randint(low=-10,high=10)

    if np.random.normal() >0:
        shift_row = np.random.randint(low=0,high=10)
    else:
        shift_row = np.random.randint(low=-10,high=0)

    if np.random.normal() >0:
        shift_col= np.random.randint(low=0,high=10)
    else:
        shift_col= np.random.randint(low=-10,high=0)

    overlay[tmp_rr,tmp_cc] = 1.0

    shifted_rr = tmp_rr + shift_row
    shifted_cc = tmp_cc + shift_col
    # shifted_rr = tmp_rr + 0
    # shifted_cc = tmp_cc + 0

    shifted_cc[shifted_cc>((img_size+20)-1)]=(img_size+20)-1
    shifted_cc[shifted_cc<0]=0

    shifted_rr[shifted_rr>((img_size+20)-1)]=(img_size+20)-1
    shifted_rr[shifted_rr<0]=0

    misaligned_overlay[shifted_rr, shifted_cc] = 1.0
    # return misaligned_overlay, overlay
    return misaligned_overlay[10:img_size+10,10:img_size+10], overlay[10:img_size+10,10:img_size+10]

def misaligned_polygon_v2(polyext,img_size,idx):
    #generates a random point for the polygon contour
    overlay = np.zeros((img_size+20,img_size+20))
    misaligned_overlay = np.zeros((img_size+20,img_size+20))
    p_ctr = np.array(polyext).reshape((-1,1,2)).astype(np.int32)
    p_ctr_v2 = p_ctr.copy() 
    p_ctr_v2[:,:,0] = p_ctr[:,:,1]
    p_ctr_v2[:,:,1] = p_ctr[:,:,0]
    tmp_rr, tmp_cc = polygon(p_ctr_v2[:,:,1]+1, p_ctr_v2[:,:,0]+1)
    np.random.seed(idx)
    shift_row = np.random.randint(low=-10,high=10)
    shift_col= np.random.randint(low=-10,high=10)
    overlay[tmp_rr,tmp_cc] = 1.0
    misaligned_overlay[tmp_rr + shift_row, tmp_cc+shift_col] = 1.0
    return misaligned_overlay[10:img_size+10,10:img_size+10], overlay[10:img_size+10,10:img_size+10]


def randomCrop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask

def custom_transform(transform,patch):
    if np.random.normal() > 0.12:
        degs = np.random.randint(low=1,high=4)
        patch = np.rot90(patch,k=degs)
    else: 
        patch = transform.random_transform(patch)
    return patch


def check_img_dims(center,step_size,dim_size):
    if dim_size == 128:
        start_loc =  center - step_size
        end_loc = center + step_size
    else:
        if (center < 64) & (dim_size >=(step_size*2)):
            start_loc = 0
            end_loc = step_size*2
        else:
            start_loc = 0
            end_loc =  dim_size
    return start_loc, end_loc
            
def get_XY_partial_polygons_with_points(imagelist_batch,root_path, batch_size,img_size,channels,mode ='train',fraction_polys=1):
    X = np.zeros((batch_size,img_size, img_size,channels), dtype=np.float32)
    y = np.zeros((batch_size,img_size,img_size,1), dtype=np.float32)
    struct2 = ndimage.generate_binary_structure(2, 2) 
    
    # import ipdb; ipdb.set_trace()
    for k, pt in enumerate(imagelist_batch):
        # patch = imread(os.path.join('../../v2_formulation_oldnewdg/',mode,pt))/255.0
        patch = cv2.imread(os.path.join(root_path,'images',pt))/255.0
        # mask = generate_label_from_json(pt,root_path,img_size)
        # mask = imread(os.path.join(root_path,mode,pt),0)/255.0
        mask = cv2.imread(os.path.join(root_path,'labels',pt),0)/255.0
        if pt.startswith('coords'):
            patch = np.dstack((patch[:,:,-1],patch[:,:,1],patch[:,:,0]))
        patch = cv2.resize(patch,(img_size,img_size))
        mask = cv2.resize(mask,(img_size,img_size))
        # new_mask = ndimage.binary_dilation(mask, structure=struct2).astype(mask.dtype)
        # new_mask = ndimage.binary_dilation(new_mask, structure=struct2).astype(new_mask.dtype)
        # patch , mask =  randomCrop(patch,mask,img_size,img_size)

        # mask[mask>0.75]=1.0
        # mask[mask<=0.75]=0

        # p_img_size = patch.shape[0]

        # pos_no_mask =  np.where(new_mask==0)

        padded_mask = np.zeros((img_size+20,img_size+20))
        padded_mask[10:img_size+10,10:img_size+10]=mask

        polygon_mask = np.zeros((img_size,img_size))
        points_mask = np.zeros((img_size,img_size))

        t_contours = measure.find_contours(padded_mask,0.8)
        t_contours = [cnt for cnt in t_contours if len(cnt)>5]
        if len(t_contours)==0:
            X[k] = X[k-1]
            y[k] = y[k-1]
        else:
            random.shuffle(t_contours)
            num_polys = 1
            # num_polys = int(np.ceil(fraction_polys* len(t_contours)))
            for p_idx in range(num_polys):
                cur_y,cur_x, tmp_overlay = random_point_in_poly_v3(t_contours[p_idx],img_size)
                # cur_y,cur_x, tmp_overlay = random_point_in_poly(t_contours[p_idx],img_size,p_idx)
                points_mask[cur_y-1:cur_y+1,cur_x-1:cur_x+1] = 1
                polygon_mask[tmp_overlay==1]=1.0

            patch, polygon_mask,points_mask = custom_rotation_ppm(patch,polygon_mask,points_mask)
            
            y[k,:,:,0] = polygon_mask
            X[k,:,:,1:] = patch[:,:,:]

            X[k,:,:,0] = points_mask
        # num_polys = len(t_contours)

        # cur_locs =  np.arange(len(pos_no_mask[0]))
        # cur_locs = np.random.permutation(cur_locs)

        # for i in range(num_polys):
        #     y_idx = pos_no_mask[0][cur_locs[i]].astype('int')
        #     x_idx = pos_no_mask[1][cur_locs[i]].astype('int')
        #     points_mask[y_idx-1:y_idx+1,x_idx-1:x_idx+1] = 1

            

        ## need to chose how many polygons to include
        
        # if len(t_contours)> 1:
        #     # tmp_fractions =  np.random.uniform(low=0,high=1.0)
        #     num_polys =  np.random.randint(low=1,high=len(t_contours)) 
        #     # num_polys = int(fraction_polys* tmp_fractions * len(t_contours))
        # else:
        #     num_polys = 1
        # indexes = np.random.randint(0, len(t_contours),num_polys)
        # for p_idx in indexes:


            # polygon_mask[cur_y,cur_x]=0.5
        # 
        
        

        # patch = cv2.resize(patch,(img_size,img_size))
        # polygon_mask = cv2.resize(polygon_mask,(img_size,img_size))
        # points_mask = cv2.resize(points_mask,(img_size,img_size))
        # polygon_mask[polygon_mask>0.75]=1.0
        # polygon_mask[polygon_mask<=0.75]=0
        # points_mask[points_mask>0.1]=1.0
    # import ipdb; ipdb.set_trace()
    return X, y

def get_XY_partial_polygons_with_points_validation(imagelist_batch,root_path, batch_size,img_size,channels,fraction_polys=1):
    X = np.zeros((batch_size,img_size, img_size,channels), dtype=np.float32)
    y = np.zeros((batch_size,img_size,img_size,1), dtype=np.float32)
    true_y = np.zeros((batch_size,img_size,img_size,1), dtype=np.float32)
    
    # import ipdb; ipdb.set_trace()
    for k, pt in enumerate(imagelist_batch):
        patch = cv2.imread(os.path.join(root_path,'images',pt))/255.0
        patch = cv2.resize(patch,(img_size,img_size))
        mask = cv2.imread(os.path.join(root_path,'realigned_labels',pt),0)/255.0
        true_mask = generate_label_from_json(pt, root_path,img_size)
        # true_mask = cv2.imread(os.path.join('../dataset/modified_airs/train/','true',pt),0)/255.0

        padded_mask = np.zeros((img_size+20,img_size+20))
        padded_mask[10:img_size+10,10:img_size+10]=mask

        # padded_mask_true = np.zeros((img_size+20,img_size+20))
        # padded_mask_true[10:img_size+10,10:img_size+10]=tmp_true_mask

        polygon_mask = np.zeros((img_size,img_size))
        points_mask = np.zeros((img_size,img_size))

        # true_mask = np.zeros((img_size,img_size))
        t_contours = measure.find_contours(padded_mask,0.8)
        t_indexes = [i for i in range(len(t_contours)) if len(t_contours[i])>5]

        # real_contours = measure.find_contours(padded_mask_true,0.8)
        # real_idx = [con for con in real_contours if len(con)>10]
        # if len(real_contours) < len(t_indexes):satellite_unet_segment-2020-06-01-15-07-44
        #     print(pt)

        ## need to chose how many polygons to include
        for p_idx in t_indexes:
            cur_y,cur_x, tmp_overlay = random_point_in_poly(t_contours[p_idx],img_size)
            points_mask[cur_y,cur_x] = 1.0
            polygon_mask[tmp_overlay==1]=1.0
            # tmp_outputs = np.array(real_contours[p_idx]).astype(np.int32) - 9
            # tmp_r_true, tmp_c_true = polygon(tmp_outputs[:,0],tmp_outputs[:,1])
            # true_mask[tmp_r_true, tmp_c_true]=1.0
        # locations = [[np.array((row,col)) for col in range(0,128,32)]
        # for row in range(0,128,32)]
        
        # # locations = [k for k in range(0,128,8)]
        # locations = np.array(locations)
        # locations = np.reshape(locations,(16,2)) 
        # points_mask[locations[:,0], locations[:,1]]=1
        # for hmm in range(0,128,32):
        #     points_mask[hmm,127]=1
        #     points_mask[127,hmm] =1
        # points_mask[127,127] =1
        y[k,:,:,0] = polygon_mask
        X[k,:,:,1:] = patch
        X[k,:,:,0] = points_mask
        true_y[k,:,:,0] = true_mask
    return X, y, true_y

def get_XY_partial_polygons_no_points_validation(imagelist_batch,root_path, batch_size,img_size,channels,fraction_polys=1):
    X = np.zeros((batch_size,img_size, img_size,channels), dtype=np.float32)
    y = np.zeros((batch_size,img_size,img_size,1), dtype=np.float32)
    true_y = np.zeros((batch_size,img_size,img_size,1), dtype=np.float32)
    # import ipdb; ipdb.set_trace()
    for k, pt in enumerate(imagelist_batch):
        patch = cv2.imread(os.path.join(root_path,'images',pt))/255.0
        patch = cv2.resize(patch,(img_size,img_size))
        mask = cv2.imread(os.path.join('../dataset/modified_airs/train/','realigned',pt),0)/255.0
        true_mask = cv2.imread(os.path.join('../dataset/modified_airs/train/','true',pt),0)/255.0

        # padded_mask = np.zeros((img_size+20,img_size+20))
        # padded_mask[10:img_size+10,10:img_size+10]=mask

        # polygon_mask = np.zeros((img_size,img_size))
        # # points_mask = np.zeros((img_size,img_size))
        # t_contours = measure.find_contours(padded_mask,0.8)
        # t_contours = [con for con in t_contours if len(con)>10]

        # ## need to chose how many polygons to include
        # for p_idx in range(len(t_contours)):
        #     cur_y,cur_x, tmp_overlay = random_point_in_poly(t_contours[p_idx],img_size)
        #     # points_mask[cur_y,cur_x] = 1.0
        #     polygon_mask[tmp_overlay==1]=1.0

        y[k,:,:,0] = mask
        true_y[k,:,:,0] = true_mask
        X[k] = patch
    return X, y, true_y


def get_XY_partial_polygons_no_points(imagelist_batch,root_path, batch_size,img_size,channels,fraction_polys=1):
    X = np.zeros((batch_size,img_size, img_size,channels), dtype=np.float32)
    y = np.zeros((batch_size,img_size,img_size,1), dtype=np.float32)

    for k, pt in enumerate(imagelist_batch):
        patch = cv2.imread(os.path.join(root_path,'images',pt))/255.0
        mask = cv2.imread(os.path.join(root_path,'labels',pt),0)/255.0
        patch = cv2.resize(patch,(img_size,img_size))
        mask = cv2.resize(mask,(img_size,img_size))
        mask[mask>0.75]=1.0
        mask[mask<=0.75]=0
        #import ipdb; ipdb.set_trace()
        # p_img_size = patch.shape[0]
        padded_mask = np.zeros((img_size+20,img_size+20))
        padded_mask[10:img_size+10,10:img_size+10]=mask

        polygon_mask = np.zeros((img_size,img_size))
        #points_mask = np.zeros((img_size,img_size))
        t_contours = measure.find_contours(padded_mask,0.8)
        t_contours = [con for con in t_contours if len(con)>10]
        random.shuffle(t_contours)

        ## need to chose how many polygons to include
        if len(t_contours)> 1:
            # tmp_fractions =  np.random.uniform(low=0,high=1.0) 
            num_polys = int(fraction_polys * 1.0 * len(t_contours))
        else:
            num_polys = 1
        for p_idx in range(num_polys):
            cur_y,cur_x, tmp_overlay = random_point_in_poly(t_contours[p_idx],img_size)
            #points_mask[cur_y,cur_x] = 1.0
            polygon_mask[tmp_overlay==1]=1.0
        # patch = cv2.resize(patch,(img_size,img_size))
        # polygon_mask = cv2.resize(polygon_mask,(img_size,img_size))
        # polygon_mask[polygon_mask>0.1]=1.0
        patch, polygon_mask = custom_rotation_pm(patch,polygon_mask)
        y[k,:,:,0] = polygon_mask
        X[k] = patch

    return X, y



def get_XY_all_polygons_no_mask(imagelist_batch,root_path, batch_size,img_size,channels):
    X = np.zeros((batch_size,img_size, img_size,channels), dtype=np.float32)
    y = np.zeros((batch_size,img_size,img_size,1), dtype=np.float32)
    # import ipdb; ipdb.set_trace()
    for k, pt in enumerate(imagelist_batch):
        # patch = cv2.imread(os.path.join(root_path,'images',pt))/255.0
        patch = imread(os.path.join(root_path,'patches',pt))/255.0
        patch =  patch[:,:,:3]
        mask = cv2.imread(os.path.join(root_path,'labels',pt),0)/255.0
        # if pt.startswith('coords'):
            # patch = np.dstack((patch[:,:,-1],patch[:,:,1],patch[:,:,0]))
        # patch = patch[16:112,16:112,:]
        # mask = mask[16:112,16:112]
        # mask = generate_label_from_json(pt,root_path,img_size)
        # mask = cv2.imread(os.path.join(root_path,'labels',pt),0)/255.0
        patch = cv2.resize(patch,(img_size,img_size))
        mask =  cv2.resize(mask,(img_size,img_size))
        # mask[mask>0.75]=1.0
        # mask[mask<=0.75]=0
        patch,mask = custom_rotation_pm(patch,mask)
        y[k,:,:,0] = mask
        X[k] = patch
    # import ipdb; ipdb.set_trace()
    # return X, to_categorical(y,num_classes=2)
    return X, y

def get_XY_misaligned(imagelist_batch,root_path,batch_size,img_size,channels):
    X = np.zeros((batch_size,img_size, img_size,channels), dtype=np.float32)
    y = np.zeros((batch_size,img_size,img_size,1), dtype=np.float32)

    for k, pt in enumerate(imagelist_batch):
        patch = cv2.imread(os.path.join(root_path,'images',pt))/255.0
        mask = cv2.imread(os.path.join(root_path,'labels',pt),0)/255.0
        patch = cv2.resize(patch,(img_size,img_size))
        mask = cv2.resize(mask,(img_size,img_size))
        mask[mask>0.75]=1.0
        mask[mask<=0.75]=0
        #import ipdb; ipdb.set_trace()
    
        padded_mask = np.zeros((img_size+20,img_size+20))
        padded_mask[10:img_size+10,10:img_size+10]=mask

        polygon_mask = np.zeros((img_size,img_size))
        misaligned_polygon_mask = np.zeros((img_size,img_size))

        t_contours = measure.find_contours(padded_mask,0.8)
        t_contours = [con for con in t_contours if len(con)>40]
        random.Random(4).shuffle(t_contours)
        # random.shuffle(t_contours)
        # select only one polygon per image
        misaligned_overlay, polygon_overlay  = misaligned_polygon(t_contours[0],img_size)
        # misaligned_overlay, polygon_overlay  = misaligned_polygon(t_contours[-1],img_size)
        polygon_mask[polygon_overlay==1]=1.0
        misaligned_polygon_mask[misaligned_overlay==1]=1.0

        patch, polygon_mask, misaligned_polygon_mask = custom_rotation_ppm(patch,polygon_mask,misaligned_polygon_mask)
        y[k,:,:,0]  = polygon_mask
        X[k,:,:,1:] = patch
        X[k,:,:,0]  = misaligned_polygon_mask

    return X, y


def get_XY_misaligned_validation(imagename,root_path,batch_size,img_size,channels):


    # for k, pt in enumerate(imagelist_batch):AOP_AF20_Q417_V0_508_308_126_9_R7C3_18816_14080_
    mask[mask<=0.75]=0
    padded_mask = np.zeros((img_size+20,img_size+20))
    padded_mask[10:img_size+10,10:img_size+10]=mask

    t_contours = measure.find_contours(padded_mask,0.8)
    t_contours = [con for con in t_contours if len(con)>40]
    X = np.zeros((len(t_contours),img_size, img_size,channels), dtype=np.float32)
    y = np.zeros((len(t_contours),img_size,img_size,1), dtype=np.float32)

        # select only one polygon per image
    for k, pt in enumerate(t_contours):
        misaligned_overlay, polygon_overlay  = misaligned_polygon_v2(pt,img_size,k)
        # polygon_mask[polygon_overlay==1]=1.0
        # misaligned_polygon_mask[misaligned_overlay==1]=1.0

        y[k,:,:,0]  = polygon_overlay
        X[k,:,:,1:] = patch
        X[k,:,:,0]  = misaligned_overlay

    return X, y

class DataGenerator():
    'This selects and prepares test, training and validation data'
    def __init__(self,args):
        self.args = args
        
        # self.imglist = os.listdir(os.path.join('../dataset/modified_airs/val/','realigned'))
        # self.imglist = os.listdir(os.path.join(args.root_path,'realigned_labels'))
        # self.imglist = os.listdir('../image_analysis_coords_model_allimages/high')
        self.imglist = os.listdir(os.path.join(args.root_path,'images'))
        # self.imglist =[k for k in self.imglist if k.startswith('coords')]
        # self.trainlist = os.listdir(os.path.join(args.root_path,'train'))
        # self.vallist = os.listdir(os.path.join(args.root_path,'val'))
        # hand_labelled = os.listdir(os.path.join(args.root_path,'hand_labelled'))
        # self.imglist = [k for k in self.imglist if k.strip('png')+'json' in hand_labelled]
        random.Random(4).shuffle(self.imglist)
        # self.imglist =  self.imglist[:500]
        self.trainlist , self.vallist =  train_test_split(self.imglist,train_size=self.args.train_fraction ,random_state=self.args.trainval_seed)
        self.trainlist =  self.trainlist #[:100]
        # self.trainlist = self.imglist
        self.train_steps_per_epoch = len(self.trainlist) //self.args.batch_size
        self.val_steps_per_epoch = len(self.vallist)// self.args.batch_size
        print('{} training  bldg points '.format(len(self.trainlist)))
        print('{} validation  bldg points '.format(len(self.vallist)))


    def generate_train(self):
        while 1:
            indexes = np.arange(len(self.trainlist))
            np.random.shuffle(indexes)
            #import ipdb; ipdb.set_trace()
            mode = 'train'
            for idx in range(0, len(indexes)- self.args.batch_size, self.args.batch_size):
                imagelist_batch = [self.trainlist[indexes[k+idx]] for k in range(self.args.batch_size)]
                if self.args.ours==2:
                    patch, polygon_pts =  get_XY_partial_polygons_with_points(imagelist_batch,self.args.root_path, self.args.batch_size,self.args.img_size,self.args.channels,mode,self.args.fraction_polys)
                elif self.args.ours==1:
                    patch, polygon_pts =  get_XY_partial_polygons_no_points(imagelist_batch,self.args.root_path, self.args.batch_size,self.args.img_size,self.args.channels,self.args.fraction_polys)
                else:
                    patch, polygon_pts = get_XY_all_polygons_no_mask(imagelist_batch,self.args.root_path, self.args.batch_size,self.args.img_size,self.args.channels)
                yield patch, polygon_pts

    def generate_valid(self):
        while 1:
            mode = 'val'
            # import ipdb; ipdb.set_trace()
            for idx in range(0, len(self.vallist)- self.args.batch_size, self.args.batch_size):
                imagelist_batch = [self.vallist[k+idx] for k in range(self.args.batch_size)]
                if self.args.ours==2:
                    patch, polygon_pts =  get_XY_partial_polygons_with_points(imagelist_batch,self.args.root_path, self.args.batch_size,self.args.img_size,self.args.channels,mode, self.args.fraction_polys)
                elif self.args.ours==1:
                    patch, polygon_pts =  get_XY_partial_polygons_no_points(imagelist_batch,self.args.root_path, self.args.batch_size,self.args.img_size,self.args.channels,self.args.fraction_polys)
                else:
                    patch, polygon_pts = get_XY_all_polygons_no_mask(imagelist_batch,self.args.root_path, self.args.batch_size,self.args.img_size,self.args.channels)
                yield patch, polygon_pts

    def generate_visual_data(self):
        mode = 'val'
        num=3
        imagelist_batch = [self.vallist[k] for k in range(self.args.batch_size *num)]
        if self.args.ours==2:
            patch, polygon_pts =  get_XY_partial_polygons_with_points(imagelist_batch,self.args.root_path, self.args.batch_size*num,self.args.img_size,self.args.channels,mode,self.args.fraction_polys)
        elif self.args.ours==1:
            patch, polygon_pts =  get_XY_partial_polygons_no_points(imagelist_batch,self.args.root_path, self.args.batch_size*num,self.args.img_size,self.args.channels,self.args.fraction_polys)
        else:
            patch, polygon_pts = get_XY_all_polygons_no_mask(imagelist_batch,self.args.root_path, self.args.batch_size*num,self.args.img_size,self.args.channels)
        return patch, polygon_pts




class misaligned_DataGenerator():
    'This selects and prepares test, training and validation data'
    def __init__(self,args):
        self.args = args
        self.imglist = os.listdir(os.path.join(args.root_path,'images'))
        random.Random(4).shuffle(self.imglist)
        self.trainlist , self.vallist =  train_test_split(self.imglist,train_size=self.args.train_fraction ,random_state=self.args.trainval_seed)
        self.train_steps_per_epoch = len(self.trainlist) //self.args.batch_size
        self.val_steps_per_epoch = len(self.vallist)// self.args.batch_size
        print('{} training  bldg points '.format(len(self.trainlist)))
        print('{} validation  bldg points '.format(len(self.vallist)))


    def generate_train(self):
        while 1:
            indexes = np.arange(len(self.trainlist))
            np.random.shuffle(indexes)
            #import ipdb; ipdb.set_trace()
            for idx in range(0, len(indexes)- self.args.batch_size, self.args.batch_size):
                imagelist_batch = [self.trainlist[indexes[k+idx]] for k in range(self.args.batch_size)]
                patch, polygon_pts =  get_XY_misaligned(imagelist_batch,self.args.root_path, self.args.batch_size,self.args.img_size,self.args.channels)
                yield patch, polygon_pts

    def generate_valid(self):
        while 1:
            # import ipdb; ipdb.set_trace()
            for idx in range(0, len(self.vallist)- self.args.batch_size, self.args.batch_size):
                imagelist_batch = [self.vallist[k+idx] for k in range(self.args.batch_size)]
                patch, polygon_pts =  get_XY_misaligned(imagelist_batch,self.args.root_path, self.args.batch_size,self.args.img_size,self.args.channels)
                yield patch, polygon_pts

    def generate_visual_data(self):
        num=3
        imagelist_batch = [self.vallist[k] for k in range(self.args.batch_size *num)]
        patch, polygon_pts =  get_XY_misaligned(imagelist_batch,self.args.root_path, self.args.batch_size*num,self.args.img_size,self.args.channels)
        return patch, polygon_pts
