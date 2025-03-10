import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html_1
import util.util as util
#nani mo nai  45  final/ttk2
#ari  30
#hidotsu   

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    for i in range(79,90, 1):#muddy2clean 220 muddy2cleanori 275
        print(i)
        i=45
        opt.epoch = str(i)
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        train_dataset = create_dataset(util.copyconf(opt, phase="train"))
        model = create_model(opt)      # create a model given opt.model and other options
        # create a webpage for viewing the results
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        print('creating web directory', web_dir)
        webpage = html_1.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            if i == 0:
                model.data_dependent_initialize(opt.epoch)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
                if opt.eval:
                    model.eval()
            # if i >= opt.num_test:  # only apply our model to opt.num_test images.
            #     break
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, width=opt.display_winsize)
        webpage.save()  # save the HTML
        exit(0)