from dataloader import DataGenerator
import tensorflow as tf
import time, os, yaml, argparse
from tensorflow.keras.optimizers import Adam
from model import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from metrics import custom_iou, custom_dice


my_best = np.Inf

def unet_finetune_vis(epoch,logs=None):
    global my_best
    if logs.get('val_loss') < my_best:
        # import ipdb; ipdb.set_trace()
        predicted_y = model.predict(visual_sample_x)
        images = []
        alpha = 0.4
        padded_visual_y =  np.zeros((visual_sample_y.shape[0],visual_sample_y.shape[1]+20,visual_sample_y.shape[2]+20,visual_sample_y.shape[3] ))
        padded_visual_y[:,10:138,10:138,:] = visual_sample_y

        padded_predicted_y =  np.zeros((predicted_y.shape[0],predicted_y.shape[1]+20,predicted_y.shape[2]+20,predicted_y.shape[3] ))
        padded_predicted_y[:,10:138,10:138,:] = predicted_y

        if visual_sample_x.shape[-1] ==4:
            for k in range(visual_sample_x.shape[0]):
                zcur_img = visual_sample_x[k,:,:,1:] *255
                zcur_point= visual_sample_x[k,:,:,0] *255
                cur_img = zcur_img.copy()
                overlay = zcur_img.copy()

                t_contours = measure.find_contours(padded_visual_y[k,:,:,0],0.8)
                for tm in t_contours: 
                    t_ctr = np.array(tm).reshape((-1,1,2)).astype(np.int32)
                    t_ctr_v2 = t_ctr.copy()
                    t_ctr_v2[:,:,0] = t_ctr[:,:,1]
                    t_ctr_v2[:,:,1] = t_ctr[:,:,0]
                    cv2.fillPoly(overlay, [t_ctr_v2-10],(0,0,255))

                p_output = padded_predicted_y[k] > 0.5

                p_contours = measure.find_contours(p_output[:,:,0],0.8) 
                for pm in p_contours: 
                    p_ctr = np.array(pm).reshape((-1,1,2)).astype(np.int32)
                    p_ctr_v2 = p_ctr.copy()
                    p_ctr_v2[:,:,0] = p_ctr[:,:,1]
                    p_ctr_v2[:,:,1] = p_ctr[:,:,0]
                    cv2.fillPoly(overlay, [p_ctr_v2-10],(0,255,0))
                # import ipdb; ipdb.set_trace()
                
                new_image = cv2.addWeighted(overlay, alpha, cur_img,1-alpha, 0) 
                new_image[zcur_point==255] = 255
                images.append(new_image/255.0)
        else:
            for k in range(visual_sample_x.shape[0]):
                zcur_img = visual_sample_x[k] *255
                cur_img = zcur_img.copy()
                overlay = zcur_img.copy()
                

                t_contours = measure.find_contours(padded_visual_y[k,:,:,0],0.8)
                for tm in t_contours: 
                    t_ctr = np.array(tm).reshape((-1,1,2)).astype(np.int32)
                    t_ctr_v2 = t_ctr.copy()
                    t_ctr_v2[:,:,0] = t_ctr[:,:,1]
                    t_ctr_v2[:,:,1] = t_ctr[:,:,0]
                    cv2.fillPoly(overlay, [t_ctr_v2-10],(0,0,255))

                p_output = padded_predicted_y[k] > 0.5

                p_contours = measure.find_contours(p_output[:,:,0],0.8) 
                for pm in p_contours: 
                    p_ctr = np.array(pm).reshape((-1,1,2)).astype(np.int32)
                    p_ctr_v2 = p_ctr.copy()
                    p_ctr_v2[:,:,0] = p_ctr[:,:,1]
                    p_ctr_v2[:,:,1] = p_ctr[:,:,0]
                    cv2.fillPoly(overlay, [p_ctr_v2-10],(0,255,0))
                
                new_image = cv2.addWeighted(overlay, alpha, cur_img,1-alpha, 0) 
                images.append(new_image/255.0)
        
        images = np.stack(images)
        with file_writer.as_default():
            tf.summary.image("Sample Images", images, max_outputs=24, step=epoch)
        my_best = logs.get('val_loss')


def get_args():
    parser = argparse.ArgumentParser(
        description= 'Predict building segments from points')

    parser.add_argument('--training_params_filename',
                        type=str,
                        default='segment.yaml',
                        help='Filename defining learning configuration')

    args = parser.parse_args()
    config = yaml.load(open(args.training_params_filename))
    for k, v in config.items():
        args.__dict__[k] = v


    args.training_time_stamp = "{}_{}".format(args.create_model_function,
                                              time.strftime("segment-%Y-%m-%d-%H-%M-%S"))
    args.trained_models_dir = os.path.join('../results/segment_trained_models', args.training_time_stamp)
    args.tensorboard_log_dir = os.path.join('../results/segment_logs', args.training_time_stamp)

    if not os.path.exists(args.trained_models_dir):
        os.makedirs(args.trained_models_dir)

    args.create_model_function = eval(args.create_model_function)

    return args

if __name__ == '__main__':
    args = get_args()
    datagenerator = DataGenerator(args)
    steps_per_epoch =  datagenerator.train_steps_per_epoch
    validation_steps = datagenerator.val_steps_per_epoch
    
    # load patches for training
    # import ipdb; ipdb.set_trace()
    training_data = datagenerator.generate_train()
    validation_data = datagenerator.generate_valid()
    visual_sample_x, visual_sample_y = datagenerator.generate_visual_data()
    yaml.dump(args.__dict__, open(os.path.join(args.trained_models_dir,args.training_params_filename),'w'))

    optimizer= Adam()
    model = args.create_model_function(image_size=args.img_size, channels=args.channels)
    
    # model.compile(loss='mse', optimizer=optimizer,metrics=args.metrics)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=[custom_iou, custom_dice,tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    print(model.summary())
    
    model_checkpoint_filepath =  os.path.join(args.trained_models_dir, 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint = ModelCheckpoint(model_checkpoint_filepath, monitor=args.checkpoint_monitor, verbose=1,save_best_only=args.checkpoint_save_best_only, mode=args.checkpoint_mode)

    # tensorboard
    file_writer = tf.summary.create_file_writer(args.tensorboard_log_dir)

    tensorboard = TensorBoard(log_dir=args.tensorboard_log_dir, write_graph=True, write_images=True)
    vis_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end = unet_finetune_vis)
    
    callback_list = [checkpoint, tensorboard, vis_cb ]

    history = model.fit_generator(generator = training_data,
                                  steps_per_epoch = steps_per_epoch,
                                  validation_data = validation_data,
                                  validation_steps = validation_steps,
                                  epochs =  args.epochs, verbose=1, callbacks=callback_list)