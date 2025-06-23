import os
import main_loop as main_loop

from config import config

os.environ['CUDA_VISIBLE_DEVICES'] = config.args.CUDA_VISIBLE_DEVICES

if __name__ == '__main__':
    restart = os.path.join('./experiments', config.args.s_experiment_name, 'restart.key')
    # while True:
    main_loop.main()